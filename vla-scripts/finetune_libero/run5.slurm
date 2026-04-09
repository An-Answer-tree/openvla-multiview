#!/bin/bash
#SBATCH --job-name=openvla-libero-backview
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=50
#SBATCH --mem=90G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/liutong/Projects/openvla-multiview/logs/%x-%j.log

set -eo pipefail

source /home/liutong/.bashrc
conda activate openvla

export http_proxy="http://localhost:7878"
export https_proxy="http://localhost:7878"

cd /home/liutong/Projects/openvla-multiview

echo "[$(date '+%F %T')] job=${SLURM_JOB_ID:-N/A} node=$(hostname) gpus=2 camera=backview"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=2 \
  vla-scripts/finetune_libero/finetune.py \
  --data_root_dir /opt/public/liutong/LIBERO-datasets-multiview-cameraINFO-double-backsideview \
  --camera_names "[backview]" \
  --wandb_project openvla-libero-backview \
  --run_root_dir /opt/liutong/openvla-runs/backview \
  --adapter_tmp_dir /opt/liutong/openvla-adapter-tmp/backview \
  --batch_size 4 \
  --grad_accumulation_steps 2 \
  --max_steps 40000
