import argparse
from pathlib import Path

import torch
import wandb
from torchmetrics.image.fid import FrechetInceptionDistance

from lightning_modules.lightning_cm import LightningConsistencyModel
from utils.datamodule_utils import get_datamodule
from utils.training_steps import get_model_step
from utils.utils import adjust_channels, rescaling_inv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FID metrics from a completed checkpoint without training.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-run-id", help="Optionally append metrics to this W&B run.")
    parser.add_argument("--project", help="W&B project; defaults to the checkpoint configuration.")
    return parser.parse_args()


def batch_data(batch):
    return batch[0] if isinstance(batch, (list, tuple)) else batch


def prepare_images(images, rescale):
    if rescale:
        images = rescaling_inv(images.clamp(-1, 1))
    else:
        images = torch.nn.functional.sigmoid(images)
    return adjust_channels(images)


@torch.inference_mode()
def add_real_features(metric, dataloader, device):
    seen = 0
    for batch in dataloader:
        images = adjust_channels(batch_data(batch).to(device))
        metric.update(images, real=True)
        seen += images.shape[0]
        if seen % 10000 == 0:
            print(f"real images: {seen}", flush=True)
    print(f"real images: {seen} (complete)", flush=True)


@torch.inference_mode()
def generated_fid(metric, model, sample_shape, n_samples, n_iters, rescale):
    metric.reset()
    torch.manual_seed(32)
    seen = 0
    while seen < n_samples:
        samples = model.sample(sample_shape, n_iters, use_ema=True)
        samples = samples[:n_samples - seen]
        metric.update(prepare_images(samples, rescale), real=False)
        seen += samples.shape[0]
        if seen % 10000 == 0 or seen == n_samples:
            print(f"generated images ({n_iters} step): {seen}/{n_samples}", flush=True)
    return float(metric.compute().cpu())


@torch.inference_mode()
def reconstruction_fid(metric, model, dataloader, rescale):
    metric.reset()
    torch.manual_seed(32)
    seen = 0
    for batch in dataloader:
        data = batch_data(batch).to(model.device)
        samples = model.ema.eval().encode_decode(data)
        metric.update(prepare_images(samples, rescale), real=False)
        seen += samples.shape[0]
        if seen % 10000 == 0:
            print(f"reconstructed images: {seen}", flush=True)
    print(f"reconstructed images: {seen} (complete)", flush=True)
    return float(metric.compute().cpu())


def main():
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    checkpoint_data = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    global_step = checkpoint_data["global_step"]
    model = LightningConsistencyModel.load_from_checkpoint(
        checkpoint, map_location="cpu", weights_only=False
    )
    cfg = model.cfg
    if args.data_dir is not None:
        cfg.dataset.data_dir = str(args.data_dir.expanduser().resolve())

    model.step = get_model_step(
        global_step,
        cfg.model.gan_warmup_steps,
        cfg.model.use_gan,
    )
    device = torch.device(args.device)
    model = model.to(device).eval()
    model.ema.eval()

    datamodule = get_datamodule(cfg)
    datamodule.prepare_data()
    datamodule.setup("fit")

    rescale = "binary" not in cfg.dataset.name
    metric = FrechetInceptionDistance(reset_real_features=False, normalize=True).to(device)
    add_real_features(metric, datamodule.fid_dataloader(), device)

    sample_shape = tuple(cfg.dataset.fid_sample_shape)
    n_samples = cfg.dataset.n_dataset_samples
    results = {
        "FID_1_iters": generated_fid(metric, model, sample_shape, n_samples, 1, rescale),
        "FID_2_iters": generated_fid(metric, model, sample_shape, n_samples, 2, rescale),
        "rec_FID_1_iters": reconstruction_fid(
            metric, model, datamodule.train_dataloader(), rescale
        ),
    }

    print(f"checkpoint: {checkpoint}")
    print(f"global_step: {global_step}")
    print(f"model iterations: {model.step}")
    for name, value in results.items():
        print(f"{name}: {value:.6f}")

    if args.wandb_run_id:
        run = wandb.init(
            project=args.project or cfg.project,
            id=args.wandb_run_id,
            resume="allow",
        )
        run.log(results, step=global_step)
        run.finish()


if __name__ == "__main__":
    main()
