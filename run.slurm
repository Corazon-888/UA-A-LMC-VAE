#!/bin/bash
#SBATCH --job-name=lmcvae-CelebA-64
#SBATCH --output=/home/lyy/VAE/UA-A-LMC-VAE/logs/%j.out
#SBATCH --error=/home/lyy/VAE/UA-A-LMC-VAE/logs/%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=gpu3,gpu4


# 修正后的 conda 初始化路径
source ~/software/miniconda3/etc/profile.d/conda.sh
conda activate covae

unset HTTP_PROXY
unset HTTPS_PROXY
unset http_proxy
unset https_proxy
unset ALL_PROXY
unset all_proxy

cd /home/lyy/VAE/UA-A-LMC-VAE

echo "===== Start ====="
which python
python --version
nvidia-smi
pwd

# Ablation Experiment for MNIST:

# /home/lyy/software/miniconda3/envs/covae/bin/python main.py \
#   --config-path conf \
#   --config-name config \
#   log_samples=True \
#   compute_fid=True \
#   log_rec=True \
#   compute_rec_fid=True \
#   dataset=mnist \
#   dataset.name=mnist \
#   model=covae \
#   project=ua-a-lmc-mnist-gamma \
#   dataset.num_workers=16 \
#   dataset.batch_size=128 \
#   model.total_training_steps=400000 \
#   model.step_schedule=exp \
#   model.start_scales=2 \
#   model.end_scales=256 \
#   model.sigma_min=0.05 \
#   model.sigma_max=3 \
#   model.time_scale=karras \
#   model.rho=7 \
#   network=autoencoder \
#   gradient_clip_val=200 \
#   model.rec_weight_mode=linear \
#   model.kl_weight_mode=square \
#   network.attn_resolutions=[14] \
#   deterministic=True \
#   network.z_channels=1 \
#   network.model_channels=64 \
#   network.channel_mult_enc=[2,2,2] \
#   network.channel_mult_dec=[2,2,2] \
#   model.denoiser_loss_mode=l2 \
#   dataset.out_channels=2 \
#   model.loss_mode=huber \
#   model.lambda_denoiser=0.1 \
#   model.use_gan=False \
#   model.lambda_latent_consistency=0.05 \
#   model.ua_lmc_gamma=3
  # model.ua_lmc_warmup_frac=0.05
  # model.ua_lmc_warmup_frac=0.01
  # model.ua_lmc_kappa=0 \
  # model.ua_lmc_eta=0 \
  # model.ua_lmc_umin=1 model.ua_lmc_umax=1


# MNIST:
# CUDA_VISIBLE_DEVICES=0 \
# /home/lyy/software/miniconda3/envs/covae/bin/python main.py \
#   --config-path conf \
#   --config-name config \
#   log_samples=True \
#   compute_fid=True \
#   log_rec=True \
#   compute_rec_fid=True \
#   dataset=mnist \
#   dataset.name=mnist \
#   model=covae \
#   project=ua-a-lmc-mnist \
#   dataset.num_workers=16 \
#   dataset.batch_size=128 \
#   model.total_training_steps=400000 \
#   model.step_schedule=exp \
#   model.start_scales=2 \
#   model.end_scales=256 \
#   model.sigma_min=0.05 \
#   model.sigma_max=3 \
#   model.time_scale=karras \
#   model.rho=7 \
#   network=autoencoder \
#   gradient_clip_val=200 \
#   model.rec_weight_mode=linear \
#   model.kl_weight_mode=square \
#   network.attn_resolutions=[14] \
#   deterministic=True \
#   network.z_channels=1 \
#   network.model_channels=64 \
#   network.channel_mult_enc=[2,2,2] \
#   network.channel_mult_dec=[2,2,2] \
#   model.denoiser_loss_mode=l2 \
#   dataset.out_channels=2 \
#   model.loss_mode=huber \
#   model.lambda_denoiser=0.1 \
#   model.use_gan=False \
#   model.lambda_latent_consistency=0.005

# CIFAR-10:

# /home/lyy/software/miniconda3/envs/covae/bin/python main.py \
#   --config-path conf \
#   --config-name config \
#   dataset=cifar10 \
#   strategy=ddp_find_unused_parameters_true \
#   model=covae \
#   log_frequency=10000 \
#   project=ua-a-lmc-cifar \
#   dataset.num_workers=16 \
#   dataset.batch_size=256 \
#   model.total_training_steps=400000 \
#   compute_rec_fid=True \
#   model.step_schedule=exp \
#   model.start_scales=2 \
#   model.end_scales=256 \
#   model.sigma_min=0.05 \
#   model.sigma_max=3 \
#   model.time_scale=karras \
#   model.rho=7 \
#   network=autoencoder \
#   gradient_clip_val=200 \
#   model.rec_weight_mode=linear \
#   model.kl_weight_mode=square \
#   log_rec=True \
#   network.attn_resolutions=[16,8] \
#   deterministic=True \
#   network.z_channels=16 \
#   network.model_channels=128 \
#   network.channel_mult_enc=[2,2,4] \
#   network.channel_mult_dec=[2,2,4] \
#   network.num_blocks_enc=4 \
#   network.num_blocks_dec=4 \
#   model.denoiser_loss_mode=l2 \
#   dataset.out_channels=6 \
#   model.loss_mode=huber \
#   model.lambda_denoiser=0.1 \
#   model.use_gan=False \
#   model.lambda_latent_consistency=0.05
  # model.ua_lmc_kappa=0 \
  # model.ua_lmc_eta=0 \
  # model.ua_lmc_umin=1 model.ua_lmc_umax=1


# CelebA-64:

# /home/lyy/software/miniconda3/envs/covae/bin/python main.py \
#   --config-path conf \
#   --config-name config \
#   dataset=celeba64 \
#   dataset.data_dir=/home/lyy/datasets \
#   strategy=ddp_find_unused_parameters_true \
#   model=covae \
#   project=ua-a-lmc-celeba64 \
#   dataset.num_workers=16 \
#   dataset.batch_size=128 \
#   model.total_training_steps=400000 \
#   compute_rec_fid=True \
#   model.step_schedule=exp \
#   model.start_scales=2 \
#   model.end_scales=256 \
#   model.sigma_min=0.05 \
#   model.sigma_max=3 \
#   model.time_scale=karras \
#   model.rho=7 \
#   network=autoencoder \
#   gradient_clip_val=200 \
#   model.rec_weight_mode=linear \
#   model.kl_weight_mode=square \
#   log_rec=True \
#   network.attn_resolutions=[16,8] \
#   deterministic=True \
#   network.z_channels=64 \
#   network.model_channels=128 \
#   network.channel_mult_enc=[1,2,2,4] \
#   network.channel_mult_dec=[1,2,2,4] \
#   network.num_blocks_enc=2 \
#   network.num_blocks_dec=2 \
#   model.denoiser_loss_mode=l2 \
#   dataset.out_channels=6 \
#   model.loss_mode=huber \
#   model.lambda_denoiser=0.1 \
#   model.use_gan=True \
#   model.gan_warmup_steps=200000 \
#   model.gan_lambda=0.05 \
#   model.lambda_latent_consistency=0.05
  # model.ua_lmc_kappa=0 \
  # model.ua_lmc_eta=0 \
  # model.ua_lmc_umin=1 model.ua_lmc_umax=1


# Resume CIFAR-10 from step 20000 (W&B run: vurfi4a5):
/home/lyy/software/miniconda3/envs/covae/bin/python main.py \
    ckpt_path=/home/lyy/VAE/UA-A-LMC-VAE/ua-a-lmc-cifar/vurfi4a5/checkpoints/last.ckpt \
    run_path=vurfi4a5 \
    root_dir=/home/lyy/VAE/UA-A-LMC-VAE \
    compute_fid=False \
    compute_rec_fid=False

# Resume CelebA-64 from step 80000 (W&B run: pyxyr1gv):
# /home/lyy/software/miniconda3/envs/covae/bin/python main.py \
#     ckpt_path=/home/lyy/VAE/UA-A-LMC-VAE/ua-a-lmc-celeba64/pyxyr1gv/checkpoints/last.ckpt \
#     run_path=pyxyr1gv \
#     root_dir=/home/lyy/VAE/UA-A-LMC-VAE \
#     dataset.data_dir=/home/lyy/datasets \
#     compute_fid=False \
#     compute_rec_fid=False
