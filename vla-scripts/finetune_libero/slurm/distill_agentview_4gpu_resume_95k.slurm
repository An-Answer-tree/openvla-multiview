#!/bin/bash
#SBATCH --job-name=openvla-distill-agentview-r95k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=50
#SBATCH --mem=96G
#SBATCH --time=10-00:00:00
#SBATCH --output=/home/liutong/Projects/openvla-multiview/logs/%x-%j.log

set -e

source /home/liutong/.bashrc
conda activate openvla

export http_proxy="http://localhost:7878"
export https_proxy="http://localhost:7878"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="/home/liutong/Projects/openvla-multiview"
mkdir -p "/opt/liutong/distillation-runs/agentview-teacher"
mkdir -p "/opt/liutong/distillation-adapter-tmp/agentview-teacher"

cd "${REPO_ROOT}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=2 \
  vla-scripts/finetune_libero/distillation.py \
  --teacher_path "/opt/liutong/Experiments/MultiviewVLA/teacher_ckpt/openvla-7b+libero_spatial+cam-agentview+b4+lr-0.0005+seed-7+lora-r32+dropout-0.0--image_aug--40000_chkpt" \
  --student_path "openvla/openvla-7b" \
  --resume_checkpoint "/opt/liutong/distillation-runs/agentview-teacher/distillation+libero_spatial+teacher-agentview+student-6v+0411-1800--95000_chkpt" \
  --data_root_dir "/opt/public/liutong/LIBERO-datasets-multiview-cameraINFO-double-backsideview" \
  --dataset_name "libero_spatial" \
  --teacher_camera_names "[agentview]" \
  --student_target_camera_names "[topview,leftview,rightview,backview,leftbackview,rightbackview]" \
  --student_gpu_ids "[0,2]" \
  --teacher_gpu_ids "[1,3]" \
  --run_root_dir "/opt/liutong/distillation-runs/agentview-teacher" \
  --adapter_tmp_dir "/opt/liutong/distillation-adapter-tmp/agentview-teacher" \
  --wandb_project "distillation-agentview-teacher" \
  --wandb_entity "szliutong-wuhan-university" \
  --batch_size 2 \
  --grad_accumulation_steps 4 \
  --max_steps 300000 \
  --save_steps 5000 \
  --learning_rate 5e-4 \
  --seed 7
