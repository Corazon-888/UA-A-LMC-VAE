#!/bin/bash
# Usage: sbatch run4.slurm [false|true]
# false (default): full UA-A-LMC-VAE; true: unweighted LMC ablation.
#SBATCH --job-name=ua-a-lmc-cifar-4gpu
#SBATCH --output=/home/lyy/VAE/UA-A-LMC-VAE/logs/%j.out
#SBATCH --error=/home/lyy/VAE/UA-A-LMC-VAE/logs/%j.err
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=gpu1,gpu3,gpu4


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
echo "SLURM_NTASKS=${SLURM_NTASKS:-unset} SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [[ "${SLURM_NTASKS:-0}" -ne 4 ]]; then
  echo "ERROR: This job requires exactly 4 Slurm tasks; got ${SLURM_NTASKS:-unset}." >&2
  exit 1
fi

DISABLE_DYNAMIC_WEIGHT="${1:-false}"
if [[ "$DISABLE_DYNAMIC_WEIGHT" != "true" && "$DISABLE_DYNAMIC_WEIGHT" != "false" ]]; then
  echo "ERROR: disable_dynamic_weight must be 'true' or 'false'; got '$DISABLE_DYNAMIC_WEIGHT'." >&2
  exit 1
fi
echo "UA-A-LMC dynamic weight disabled: $DISABLE_DYNAMIC_WEIGHT"

# CIFAR-10: 256 images/GPU x 4 GPUs = global batch size 1024.
srun --ntasks=4 --ntasks-per-node=4 --kill-on-bad-exit=1 \
  /home/lyy/software/miniconda3/envs/covae/bin/python main.py \
  --config-path conf \
  --config-name config \
  dataset=cifar10 \
  devices=auto \
  strategy=ddp \
  model=covae \
  log_frequency=10000 \
  log_model=all \
  project=ua-a-lmc-cifar \
  dataset.num_workers=16 \
  dataset.batch_size=256 \
  model.total_training_steps=400000 \
  compute_rec_fid=True \
  compute_fid=False \
  model.step_schedule=exp \
  model.start_scales=2 \
  model.end_scales=256 \
  model.sigma_min=0.05 \
  model.sigma_max=3 \
  model.time_scale=karras \
  model.rho=7 \
  network=autoencoder \
  gradient_clip_val=200 \
  model.rec_weight_mode=linear \
  model.kl_weight_mode=square \
  log_rec=True \
  network.attn_resolutions=[16,8] \
  deterministic=True \
  network.z_channels=16 \
  network.model_channels=128 \
  network.channel_mult_enc=[2,2,4] \
  network.channel_mult_dec=[2,2,4] \
  network.num_blocks_enc=4 \
  network.num_blocks_dec=4 \
  model.denoiser_loss_mode=l2 \
  dataset.out_channels=6 \
  model.loss_mode=huber \
  model.lambda_denoiser=0.1 \
  model.use_gan=False \
  model.lambda_latent_consistency=0.01 \
  model.ua_lmc_gamma=1 \
  model.ua_lmc_disable_dynamic_weight="$DISABLE_DYNAMIC_WEIGHT"
