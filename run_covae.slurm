#!/bin/bash
#SBATCH --job-name=lmcvae-mnist
#SBATCH --output=/home/lyy/VAE/covae/logs/%j.out
#SBATCH --error=/home/lyy/VAE/covae/logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=gpu3


# 修正后的 conda 初始化路径
source ~/software/miniconda3/etc/profile.d/conda.sh
conda activate covae

unset HTTP_PROXY
unset HTTPS_PROXY
unset http_proxy
unset https_proxy
unset ALL_PROXY
unset all_proxy

cd /home/lyy/VAE/covae


# 恢复训练
# python main.py \
#     ckpt_path=/home/lyy/VAE/covae/lmcvae-mnist/wqb9qdfu/checkpoints/last.ckpt \
#     run_path=wqb9qdfu \
#     root_dir=/home/lyy/VAE/covae

# 训练
# python main.py log_samples=True compute_fid=True log_rec=True compute_rec_fid=True \
#     dataset=mnist dataset.name=mnist model=covae project=lmcvae-mnist \
#     dataset.num_workers=16 dataset.batch_size=128 model.total_training_steps=400000 \
#     model.step_schedule=exp model.start_scales=2 model.end_scales=256 \
#     model.sigma_min=0.05 model.sigma_max=3 model.time_scale=karras model.rho=7 \
#     network=autoencoder gradient_clip_val=200 model.rec_weight_mode=linear \
#     model.kl_weight_mode=square network.attn_resolutions=[14] deterministic=True \
#     network.z_channels=1 network.model_channels=64 network.channel_mult_enc=[2,2,2] \
#     network.channel_mult_dec=[2,2,2] model.denoiser_loss_mode=l2 dataset.out_channels=2 \
#     model.loss_mode=huber model.lambda_denoiser=0.1 model.use_gan=False model.lambda_latent_consistency=0.01