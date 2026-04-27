#!/bin/bash
#SBATCH --job-name=distill-2views-action-projector-40k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=10-00:00:00
#SBATCH --qos=high_priority
#SBATCH --output=/home/liutong/Projects/openvla-multiview/logs/distillation_log/%x-%j.log

set -eo pipefail

source /home/liutong/.bashrc
conda activate openvla

export http_proxy="http://localhost:7878"
export https_proxy="http://localhost:7878"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [[ ! -d "/opt/liutong/openvla-runs/multiview/spatial/2views/openvla-7b+libero_spatial+cam-agentview+topview+b16+lr-0.0005+seed-7+lora-r32+dropout-0.0+sched-linear-warmup+cosine-decay+warmup-0.03--image_aug--40000_chkpt" ]]; then
  echo "Teacher checkpoint not found: /opt/liutong/openvla-runs/multiview/spatial/2views/openvla-7b+libero_spatial+cam-agentview+topview+b16+lr-0.0005+seed-7+lora-r32+dropout-0.0+sched-linear-warmup+cosine-decay+warmup-0.03--image_aug--40000_chkpt" >&2
  exit 1
fi

mkdir -p \
  "/opt/liutong/distill_ckpt/spatial/action_projector/2views_teacher/40k" \
  "/opt/liutong/distill_tmp/spatial/action_projector/2views_teacher/40k"
cd /home/liutong/Projects/openvla-multiview

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=2 \
  vla-scripts/finetune_libero/distillation.py \
  --teacher_path "/opt/liutong/openvla-runs/multiview/spatial/2views/openvla-7b+libero_spatial+cam-agentview+topview+b16+lr-0.0005+seed-7+lora-r32+dropout-0.0+sched-linear-warmup+cosine-decay+warmup-0.03--image_aug--40000_chkpt" \
  --student_path "openvla/openvla-7b" \
  --resume_checkpoint "null" \
  --data_root_dir "/opt/public/liutong/LIBERO-datasets-multiview-cameraINFO-double-backsideview" \
  --dataset_name "libero_spatial" \
  --teacher_camera_names "[agentview,topview]" \
  --student_target_camera_names "[agentview,topview,leftview,rightview,backview,leftbackview,rightbackview]" \
  --distill_camera_name "null" \
  --run_root_dir "/opt/liutong/distill_ckpt/spatial/action_projector/2views_teacher/40k" \
  --adapter_tmp_dir "/opt/liutong/distill_tmp/spatial/action_projector/2views_teacher/40k" \
  --batch_size "2" \
  --max_steps "200000" \
  --save_steps "5000" \
  --learning_rate "3e-4" \
  --grad_accumulation_steps "4" \
  --image_aug "True" \
  --shuffle_buffer_size "100000" \
  --seed "7" \
  --global_shuffle_across_ranks "True" \
  --save_latest_checkpoint_only "False" \
  --use_lora "True" \
  --lora_rank "32" \
  --lora_dropout "0.0" \
  --use_quantization "False" \
  --adapter_hidden_scale "4.0" \
  --action_loss_weight "1.0" \
  --distill_patch_loss_weight "0.0" \
  --distill_projector_loss_weight "1.0" \
  --use_balanced_view_sampling "True" \
  --use_dual_lr_scheduler "True" \
  --adapter_learning_rate "1e-3" \
  --lora_learning_rate "3e-4" \
  --use_lr_scheduler "True" \
  --lr_scheduler_type "linear-warmup+cosine-decay" \
  --warmup_ratio "0.03" \
  --min_lr_ratio "0.1" \
  --max_grad_norm "1.0" \
  --student_gpu_ids "[0,2]" \
  --teacher_gpu_ids "[1,3]" \
  --wandb_project "distillation-spatial" \
  --wandb_entity "szliutong-wuhan-university" \
  --run_id_note "2views-teacher-action-projector-40k"
