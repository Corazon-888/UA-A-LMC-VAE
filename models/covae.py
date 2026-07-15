import torch
from torch import Tensor, nn
from lightning.pytorch.utilities.seed import isolate_rng
import operator
from functools import reduce


from models.covae_base import CoVAEBase

class CoVAE(CoVAEBase):
    def __init__(self,
                 time_scale,
                 rec_weight_mode,
                 kl_weight_mode,
                 lambda_denoiser,
                 latent_type,
                 latent_shape,
                 lambda_latent_consistency=0.1,
                 ua_lmc_kappa=1.,
                 ua_lmc_gamma=2.,
                 ua_lmc_eta=0.5,
                 ua_lmc_umin=0.5,
                 ua_lmc_umax=2.,
                 ua_lmc_eps=1e-6,
                 ua_lmc_warmup_frac=0.1,
                 **cm_kwargs
                 ):
        super().__init__(**cm_kwargs)
        self.time_scale = time_scale
        self.rec_weight_mode = rec_weight_mode
        self.kl_weight_mode = kl_weight_mode
        self.lambda_denoiser = lambda_denoiser
        self.lambda_latent_consistency = lambda_latent_consistency
        self.ua_lmc_kappa = ua_lmc_kappa
        self.ua_lmc_gamma = ua_lmc_gamma
        self.ua_lmc_eta = ua_lmc_eta
        self.ua_lmc_umin = ua_lmc_umin
        self.ua_lmc_umax = ua_lmc_umax
        self.ua_lmc_eps = ua_lmc_eps
        self.ua_lmc_warmup_frac = ua_lmc_warmup_frac
        self.latent_type = latent_type
        self.latent_shape = latent_shape
        assert latent_type in ['gaussian', 'categorical']
        if lambda_latent_consistency > 0 and latent_type != 'gaussian':
            raise NotImplementedError('UA-A-LMC requires gaussian posterior uncertainty.')
        if latent_type == 'categorical':
            assert self.num_elements(self.latent_shape) == self.num_elements(self.noise_shape)

    def num_elements(self, shape):
        return reduce(operator.mul, shape, 1)

    def _get_distribution(self, mu, std):
        if self.latent_type == 'gaussian':
            return torch.distributions.Normal(mu, std)
        elif self.latent_type == 'categorical':
            mu = mu.view([mu.shape[0]] + self.latent_shape[::-1]).transpose(1,2)
            return torch.distributions.Categorical(logits=mu)

    def gumbel_softmax_sample(self, logits, noise, tau=1.0):
        gumbel_noise = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
        y = logits + gumbel_noise
        return nn.functional.softmax(y / tau, dim=-1)

    def gumbel_softmax(self, logits, noise, tau=1.0, hard=False):
        y = self.gumbel_softmax_sample(logits, noise, tau)
        if hard:
            y_hard = torch.zeros_like(y)
            y_hard.scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
            y = (y_hard - y).detach() + y  # Straight-through estimator
        return y

    def sample_noise(self, batch_size, device):
        if self.latent_type == 'gaussian':
            return torch.randn([batch_size] + self.noise_shape, dtype=torch.float32).to(device)
        elif self.latent_type == 'categorical':
            return torch.rand([batch_size] + self.latent_shape, dtype=torch.float32).to(device)


    def _reparametrized_sample(self, mu, std, noise):
        if self.latent_type == 'gaussian':
            return mu + std * noise
        elif self.latent_type == 'categorical':
            mu_shape = mu.shape
            mu = mu.view([mu.shape[0]] + self.latent_shape[::-1]).transpose(1,2)
            sample = self.gumbel_softmax(mu, noise, hard=True)
            return sample.transpose(1,2).view(mu_shape)


    def _get_time_steps(self, num_timesteps, device):
        assert self.time_scale in ['linear', 'log', 'karras']
        if self.time_scale == 'linear':
            time_steps = torch.linspace(self.sigma_min, self.sigma_max, num_timesteps, device=device)
        elif self.time_scale == 'log':
            time_steps = torch.exp(torch.linspace(
                torch.log(torch.tensor(self.sigma_min)),
                torch.log(torch.tensor(self.sigma_max)),
                num_timesteps,
                device=device
            ))
        elif self.time_scale == 'karras':
            time_steps = self._get_sigmas_karras(num_timesteps, device)

        time_steps = torch.cat([torch.zeros(1).to(torch.float32).to(device), time_steps], dim=0)

        return time_steps

    def _get_rec_loss_weights(self, t):
        return self._get_loss_weights(t, self.rec_weight_mode)

    def _get_kl_loss_weights(self, t):
        return 1 / self._get_loss_weights(t, self.kl_weight_mode)

    def _get_loss_weights(self, t, mode):
        assert mode in ['linear', 'square', 'ones']
        if mode == 'linear':
            return 1 / t
        elif mode == 'square':
            return 1 / (t ** 2)
        elif mode == 'ones':
            return torch.ones_like(t)

    def _get_latent_consistency_weight(self, step):
        if self.ua_lmc_warmup_frac <= 0:
            return self.lambda_latent_consistency

        warmup_steps = max(int(self.total_training_steps * self.ua_lmc_warmup_frac), 1)
        warmup = min((step + 1) / warmup_steps, 1.)
        return self.lambda_latent_consistency * warmup

    def _get_ua_lmc_loss(self, mu_h, mu_l, std_h, std_l, t_h, t_l):
        snr_h = 1 / (t_h.square() + self.ua_lmc_eps)
        snr_l = 1 / (t_l.square() + self.ua_lmc_eps)
        w_conf = (snr_h / (snr_h + self.ua_lmc_kappa)).pow(self.ua_lmc_gamma)
        w_gap = torch.exp(-self.ua_lmc_eta * (torch.log(snr_l) - torch.log(snr_h)).abs())
        w_noise = self._append_dims(w_conf * w_gap, mu_h.ndim)

        var_h = std_h.square()
        var_l = std_l.square()
        w_unc = 1 / (var_l + var_h + self.ua_lmc_eps)
        w_unc = w_unc.clamp(min=self.ua_lmc_umin, max=self.ua_lmc_umax).detach()
        w_unc = w_unc / (w_unc.mean() + self.ua_lmc_eps)

        return (w_noise * w_unc * (mu_h - mu_l).square()).mean()

    def _decode_fn(self, z, t, emb):
        x = self.model.decoder(z, emb)
        if self.denoiser_loss_mode:
            x, denoiser_x = torch.chunk(x, 2, dim=1)
            x = denoiser_x.detach() + (t - self.sigma_min) / (self.sigma_max - self.sigma_min) * x.to(torch.float32)
        else:
            denoiser_x = None
        return x, denoiser_x

    def precond(self, x, t, noise, class_labels):
        x = x.to(torch.float32)
        t = self._append_dims(t, x.ndim).to(x).to(torch.float32)
        c_noise = t.log() / 4
        emb = self.model.time_embedding(c_noise.flatten(), class_labels=class_labels)
        mu, std = self.model.encoder(x, emb)
        z = self._reparametrized_sample(mu, std, noise)
        x, denoiser_x = self._decode_fn(z, t, emb)
        return x, mu, std, denoiser_x

    def decode(self, z, t, class_labels):
        z = z.to(torch.float32)
        t = self._append_dims(t, z.ndim).to(z).to(torch.float32)
        c_noise = t.log() / 4
        emb = self.model.time_embedding(c_noise.flatten(), class_labels)
        x, denoiser_x = self._decode_fn(z, t, emb)
        return x, denoiser_x

    def encode(self, x, t, class_labels):
        x = x.to(torch.float32)
        t = self._append_dims(t, x.ndim).to(x).to(torch.float32)
        c_noise = t.log() / 4
        emb = self.model.time_embedding(c_noise.flatten(), class_labels)
        mu, std = self.model.encoder(x, emb)
        return mu, std

    def encode_decode(self, x, idx=1, class_labels=None, noise=None):
        device = x.device
        batch_size = x.shape[0]
        time_steps = self._get_time_steps(self.end_scales + 1, device=device)
        t = time_steps[idx].to(device)
        if not torch.is_tensor(noise):
            noise = self.sample_noise(batch_size, device)
        x, _, _, _ = self.precond(x, t, noise, class_labels)
        return x

    def loss(self, x, step, labels=None):

        log_dict = {}
        dims = x.ndim  # keeps track of data dimensionality to work with both images and tabular
        device = x.device
        batch_size = x.shape[0]
        num_timesteps = self._step_schedule(step)
        time_steps = self._get_time_steps(num_timesteps, device=device)
        idxs = torch.randint(0, len(time_steps) - 1, (batch_size,))
        t = time_steps[idxs + 1].to(device)
        r = time_steps[idxs].to(device)
        noise = self.sample_noise(batch_size, device)

        with isolate_rng():
            x_t, mu, std, denoiser_x = self.precond(x, t, noise, labels)

        if (idxs == 0).all():
            # save time when training simple vae
            x_r = x
            mu_r = None
            std_r = None
        else:
            with torch.no_grad():
                x_r, mu_r, std_r, _ = self.precond(x, r, noise, labels)

        if self.lambda_latent_consistency > 0 and mu_r is not None:
            latent_mask = (idxs > 0).to(device)
            latent_consistency_loss = self._get_ua_lmc_loss(
                mu[latent_mask],
                mu_r[latent_mask],
                std[latent_mask],
                std_r[latent_mask],
                t[latent_mask],
                r[latent_mask],
            )
            log_dict['latent_consistency_loss'] = latent_consistency_loss.detach()
        else:
            latent_consistency_loss = mu.new_zeros(())

        if self.loss_mode == 'bce':
            x_r = torch.where(self._append_dims(idxs > 0, dims).to(device), nn.functional.sigmoid(x_r), x)
        else:
            # boundary condition
            x_r = torch.where(self._append_dims(idxs > 0, dims).to(device), x_r, x)

        rec_loss = self._loss_fn(x_t, x_r.detach(), self.loss_mode)
        log_dict['rec_loss'] = rec_loss.detach().view(batch_size, -1).sum(1).mean()
        rec_loss_weights = self._get_rec_loss_weights(t)
        kl_loss_weights = self._get_kl_loss_weights(t)

        if self.denoiser_loss_mode:
            denoiser_loss = self._loss_fn(denoiser_x, x, self.denoiser_loss_mode)
            log_dict['denoiser_loss'] = denoiser_loss.detach().view(batch_size, -1).sum(1).mean()
            denoiser_skip = self.lambda_denoiser + (1 - self.lambda_denoiser) * (1 - (t - self.sigma_min)/(self.sigma_max - self.sigma_min))
            denoiser_loss = denoiser_loss * self._append_dims(denoiser_skip, dims)
            denoiser_loss = denoiser_loss.view(batch_size, -1).sum(1) * rec_loss_weights
        else:
            denoiser_loss = 0.

        if self.use_gan and step >= self.gan_warmup_steps:
            if self.denoiser_loss_mode:
                gan_input = torch.where(self._append_dims(idxs > 1, dims).to(device), x_t, denoiser_x)
            else:
                gan_input = x_t
            gan_loss = -torch.mean(self.discriminator(torch.clamp(gan_input, -1, 1).contiguous()).view(batch_size, -1), dim=1)
            mask_idx = int(self.end_scales / ((self.total_training_steps - self.gan_warmup_steps)  / (step + 1 - self.gan_warmup_steps)))
            all_timesteps = self._get_time_steps(self.end_scales + 1, device=device)
            mask = torch.where(t > all_timesteps[mask_idx], 0., 1.).to(device)
            gan_loss = gan_loss * mask
            log_dict['generator_loss'] = gan_loss.detach().mean()
            gan_loss = gan_loss * rec_loss_weights * ((step + 1 - self.gan_warmup_steps)/ (self.total_training_steps - self.gan_warmup_steps)) * self.gan_lambda
            gan_loss = gan_loss.mean()
        else:
            gan_loss = 0.

        rec_loss = rec_loss.view(batch_size, -1).sum(1) * rec_loss_weights
        posterior = self._get_distribution(mu, std + 1e-8)
        prior = self._get_distribution(torch.zeros_like(mu), torch.ones_like(std))
        kl_loss = torch.distributions.kl_divergence(posterior, prior)
        log_dict['kl_loss'] = kl_loss.detach().reshape(batch_size, -1).sum(1).mean()
        kl_loss = kl_loss.reshape(batch_size, -1).sum(-1) * kl_loss_weights

        loss = (rec_loss + denoiser_loss + kl_loss).mean() + gan_loss
        loss = loss + self._get_latent_consistency_weight(step) * latent_consistency_loss
        return loss, log_dict, x_t

    @torch.no_grad()
    def sample(self, sample_shape, n_iters, device, class_labels=None, idx=None, temperature=1):
        time_steps = self._get_time_steps(self.end_scales + 1, device)
        t = torch.ones(sample_shape[0], device=device) * time_steps[-1]
        noise = self.sample_noise(sample_shape[0], device) * temperature
        mu = torch.zeros([sample_shape[0]] + self.noise_shape, dtype=torch.float32, device=device)
        std = torch.ones([sample_shape[0]] + self.noise_shape, dtype=torch.float32, device=device)
        z = self._reparametrized_sample(mu, std, noise)
        # z = torch.randn([sample_shape[0]] + self.noise_shape).to(device)
        x, _ = self.decode(z, t, class_labels)
        if idx is None:
            if time_steps.shape[0] > 2:
                idx = round((self.end_scales + 1) * 0.5)
            else:
                idx = 1
        for i in range(1, n_iters):
            t = torch.ones(sample_shape[0], device=device) * time_steps[idx].to(device)
            noise = self.sample_noise(sample_shape[0], device)
            x, _, _, _ = self.precond(x, t, noise, class_labels)
        return x
