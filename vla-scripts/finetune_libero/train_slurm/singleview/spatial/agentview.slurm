#!/bin/bash
#SBATCH --job-name=openvla-libero-spatial-agentview-seed8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=50
#SBATCH --mem=90G
#SBATCH --time=10-00:00:00
#SBATCH --qos=high_priority
#SBATCH --output=/home/liutong/Projects/openvla-multiview/logs/%x-%j.out

set -eo pipefail

source /home/liutong/.bashrc
conda activate openvla

export http_proxy="http://localhost:7878"
export https_proxy="http://localhost:7878"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd /home/liutong/Projects/openvla-multiview

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=2 \
  vla-scripts/finetune_libero/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /opt/public/liutong/LIBERO-datasets-multiview-cameraINFO-double-backsideview \
  --dataset_name libero_spatial \
  --camera_names "[agentview]" \
  --run_root_dir /opt/liutong/openvla-runs/singleview/spatial/agentview-seed8 \
  --adapter_tmp_dir /opt/liutong/openvla-adapter-tmp/singleview/spatial/agentview-seed8 \
  --batch_size 4 \
  --max_steps 40000 \
  --save_steps 5000 \
  --learning_rate 5e-4 \
  --use_lr_scheduler True \
  --lr_scheduler_type linear-warmup+cosine-decay \
  --warmup_ratio 0.03 \
  --grad_accumulation_steps 4 \
  --image_aug True \
  --shuffle_buffer_size 100000 \
  --seed 8 \
  --global_shuffle_across_ranks True \
  --save_latest_checkpoint_only False \
  --use_lora True \
  --lora_rank 32 \
  --lora_dropout 0.0 \
  --use_quantization False \
  --wandb_project openvla-libero-spatial-agentview \
  --wandb_entity szliutong-wuhan-university
