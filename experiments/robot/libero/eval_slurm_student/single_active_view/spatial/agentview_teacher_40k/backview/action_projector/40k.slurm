#!/bin/bash
#SBATCH --job-name=eval-student-spatial-backview-40k-action-projector
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=25
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --qos=high_priority
#SBATCH --exclude=gpu03
#SBATCH --output=/home/liutong/Projects/openvla-multiview/logs/distillation_log/spatial/agentview_teacher_40k/backview/action_projector/%x-%j.log

set -eo pipefail

source /home/liutong/.bashrc
conda activate openvla

export http_proxy="http://localhost:7878"
export https_proxy="http://localhost:7878"
export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_ONEDNN_OPTS=0

cd /home/liutong/Projects/openvla-multiview

python experiments/robot/libero/run_libero_eval_student.py \
    --pretrained_checkpoint "/opt/liutong/distill_ckpt/spatial/single_active_view/agentview_teacher_40k/backview/action_projector/distillation+libero_spatial+teacher-agentview+student-backview+0428-0138--40000_chkpt" \
    --camera_names "[backview]" \
    --local_log_dir "/opt/liutong/multiview_test/student/single_active_view/spatial/agentview_teacher_40k/backview/action_projector/40k" \
    --rollout_root_dir "/opt/liutong/multiview_test/student/single_active_view/spatial/agentview_teacher_40k/backview/action_projector/40k/videos"
