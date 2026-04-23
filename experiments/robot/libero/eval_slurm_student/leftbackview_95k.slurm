#!/bin/bash
#SBATCH --job-name=eval-student-spatial-leftbackview-95k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=25
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/liutong/Projects/openvla-multiview/logs/%x-%j.log

set -eo pipefail

source /home/liutong/.bashrc
conda activate openvla

export http_proxy="http://localhost:7878"
export https_proxy="http://localhost:7878"
export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_ONEDNN_OPTS=0

cd /home/liutong/Projects/openvla-multiview

python experiments/robot/libero/run_libero_eval_student.py \
    --pretrained_checkpoint "/opt/liutong/distillation-runs/agentview-teacher/distillation+libero_spatial+teacher-agentview+student-6v+0411-1800--95000_chkpt" \
    --camera_names "[leftbackview]" \
    --local_log_dir "/opt/liutong/multiview_test/student_leftbackview_95k" \
    --rollout_root_dir "/opt/liutong/multiview_test/student_leftbackview_95k/videos"
