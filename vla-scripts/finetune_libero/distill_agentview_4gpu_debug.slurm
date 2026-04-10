#!/bin/bash
#SBATCH --job-name=openvla-distill-agentview-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00-01:00:00
#SBATCH --output=/home/liutong/Projects/openvla-multiview/logs/%x-%j.log
#SBATCH --qos=debug

set -e

source /home/liutong/.bashrc
conda activate openvla

export http_proxy="http://localhost:7878"
export https_proxy="http://localhost:7878"

REPO_ROOT="/home/liutong/Projects/openvla-multiview"

TEACHER_PATH="/opt/liutong/Experiments/MultiviewVLA/teacher_ckpt/openvla-7b+libero_spatial+cam-agentview+b4+lr-0.0005+seed-7+lora-r32+dropout-0.0--image_aug--40000_chkpt"
STUDENT_PATH="openvla/openvla-7b"
DATA_ROOT_DIR="/opt/public/liutong/LIBERO-datasets-multiview-cameraINFO-double-backsideview"
RUN_ROOT_DIR="/opt/liutong/distillation-runs/agentview-teacher"
ADAPTER_TMP_DIR="/opt/liutong/distillation-adapter-tmp/agentview-teacher"

DATASET_NAME="libero_spatial"
WANDB_PROJECT="distillation-agentview-teacher"
WANDB_ENTITY="szliutong-wuhan-university"

BATCH_SIZE=4
GRAD_ACCUMULATION_STEPS=4
MAX_STEPS=40000
SAVE_STEPS=5000
LEARNING_RATE=5e-4
SEED=7

STUDENT_GPU_IDS="[0]"
TEACHER_GPU_IDS="[1]"
TEACHER_CAMERA_NAMES="[agentview]"
STUDENT_TARGET_CAMERA_NAMES="[topview,leftview,rightview,backview,leftbackview,rightbackview]"

mkdir -p "${RUN_ROOT_DIR}"
mkdir -p "${ADAPTER_TMP_DIR}"

cd "${REPO_ROOT}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=1 \
  vla-scripts/finetune_libero/distillation.py \
  --teacher_path "${TEACHER_PATH}" \
  --student_path "${STUDENT_PATH}" \
  --data_root_dir "${DATA_ROOT_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --teacher_camera_names "${TEACHER_CAMERA_NAMES}" \
  --student_target_camera_names "${STUDENT_TARGET_CAMERA_NAMES}" \
  --student_gpu_ids "${STUDENT_GPU_IDS}" \
  --teacher_gpu_ids "${TEACHER_GPU_IDS}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --adapter_tmp_dir "${ADAPTER_TMP_DIR}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accumulation_steps "${GRAD_ACCUMULATION_STEPS}" \
  --max_steps "${MAX_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --seed "${SEED}"
