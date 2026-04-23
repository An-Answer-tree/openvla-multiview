#!/bin/bash
#SBATCH --job-name=distill-7views-80k
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
conda activate dwt

export http_proxy="http://localhost:7878"
export https_proxy="http://localhost:7878"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

REPO_ROOT="/home/liutong/Projects/openvla-multiview"
TEACHER_PATH="/opt/liutong/openvla-runs/multiview/spatial/7views/openvla-7b+libero_spatial+cam-agentview+topview+leftview+rightview+backview+leftbackview+rightbackview+b16+lr-0.0005+seed-7+lora-r32+dropout-0.0+sched-linear-warmup+cosine-decay+warmup-0.03--image_aug--80000_chkpt"
STUDENT_PATH="openvla/openvla-7b"
DATA_ROOT_DIR="/opt/public/liutong/LIBERO-datasets-multiview-cameraINFO-double-backsideview"
RUN_ROOT_DIR="/opt/liutong/distill_ckpt/spatial/7views/80k"
ADAPTER_TMP_DIR="/opt/liutong/distill_tmp/spatial/7views/80k"
DATASET_NAME="libero_spatial"
TEACHER_CAMERA_NAMES="[agentview,topview,leftview,rightview,backview,leftbackview,rightbackview]"
STUDENT_TARGET_CAMERA_NAMES="[agentview,topview,leftview,rightview,backview,leftbackview,rightbackview]"
WANDB_PROJECT="distillation-spatial"
WANDB_ENTITY="szliutong-wuhan-university"
RUN_ID_NOTE="7views-teacher-80k"
BATCH_SIZE=2
MAX_STEPS=200000
SAVE_STEPS=5000
LEARNING_RATE=3e-4
GRAD_ACCUMULATION_STEPS=4
IMAGE_AUG=True
SHUFFLE_BUFFER_SIZE=100000
SEED=7
GLOBAL_SHUFFLE_ACROSS_RANKS=True
SAVE_LATEST_CHECKPOINT_ONLY=False
USE_LORA=True
LORA_RANK=32
LORA_DROPOUT=0.0
USE_QUANTIZATION=False
ADAPTER_HIDDEN_SCALE=4.0
ACTION_LOSS_WEIGHT=1.0
DISTILL_PATCH_LOSS_WEIGHT=1.0
DISTILL_PROJECTOR_LOSS_WEIGHT=1.0
USE_BALANCED_VIEW_SAMPLING=True
USE_DUAL_LR_SCHEDULER=True
ADAPTER_LEARNING_RATE=1e-3
LORA_LEARNING_RATE=3e-4
USE_LR_SCHEDULER=True
LR_SCHEDULER_TYPE="linear-warmup+cosine-decay"
WARMUP_RATIO=0.03
MIN_LR_RATIO=0.1
MAX_GRAD_NORM=1.0
STUDENT_GPU_IDS="[0,2]"
TEACHER_GPU_IDS="[1,3]"

if [[ ! -d "${TEACHER_PATH}" ]]; then
  echo "Teacher checkpoint not found: ${TEACHER_PATH}" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT_DIR}" "${ADAPTER_TMP_DIR}"
cd "${REPO_ROOT}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=2 \
  vla-scripts/finetune_libero/distillation.py \
  --teacher_path "${TEACHER_PATH}" \
  --student_path "${STUDENT_PATH}" \
  --data_root_dir "${DATA_ROOT_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --teacher_camera_names "${TEACHER_CAMERA_NAMES}" \
  --student_target_camera_names "${STUDENT_TARGET_CAMERA_NAMES}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --adapter_tmp_dir "${ADAPTER_TMP_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --max_steps "${MAX_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --grad_accumulation_steps "${GRAD_ACCUMULATION_STEPS}" \
  --image_aug "${IMAGE_AUG}" \
  --shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE}" \
  --seed "${SEED}" \
  --global_shuffle_across_ranks "${GLOBAL_SHUFFLE_ACROSS_RANKS}" \
  --save_latest_checkpoint_only "${SAVE_LATEST_CHECKPOINT_ONLY}" \
  --use_lora "${USE_LORA}" \
  --lora_rank "${LORA_RANK}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --use_quantization "${USE_QUANTIZATION}" \
  --adapter_hidden_scale "${ADAPTER_HIDDEN_SCALE}" \
  --action_loss_weight "${ACTION_LOSS_WEIGHT}" \
  --distill_patch_loss_weight "${DISTILL_PATCH_LOSS_WEIGHT}" \
  --distill_projector_loss_weight "${DISTILL_PROJECTOR_LOSS_WEIGHT}" \
  --use_balanced_view_sampling "${USE_BALANCED_VIEW_SAMPLING}" \
  --use_dual_lr_scheduler "${USE_DUAL_LR_SCHEDULER}" \
  --adapter_learning_rate "${ADAPTER_LEARNING_RATE}" \
  --lora_learning_rate "${LORA_LEARNING_RATE}" \
  --use_lr_scheduler "${USE_LR_SCHEDULER}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --min_lr_ratio "${MIN_LR_RATIO}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --student_gpu_ids "${STUDENT_GPU_IDS}" \
  --teacher_gpu_ids "${TEACHER_GPU_IDS}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --run_id_note "${RUN_ID_NOTE}"
