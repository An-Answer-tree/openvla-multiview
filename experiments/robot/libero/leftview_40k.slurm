#!/bin/bash
#SBATCH --job-name=eval-spatial-leftview-40k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=25
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/liutong/Projects/openvla/logs/%x-%j.log

set -eo pipefail

source /home/liutong/.bashrc
conda activate openvla

export http_proxy="http://localhost:7878"
export https_proxy="http://localhost:7878"
export LIBERO_ROOT="/home/liutong/Projects/lerobot-libero-NewViewCameras"

cd /home/liutong/Projects/openvla

python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "/opt/liutong/openvla-runs/leftview/openvla-7b+libero_spatial+cam-leftview+b8+lr-0.0005+seed-7+lora-r32+dropout-0.0--image_aug--40000_chkpt" \
    --camera_names "[leftview]" \
    --local_log_dir "/opt/liutong/multiview_test/leftview_40k" \
    --rollout_root_dir "/opt/liutong/multiview_test/leftview_40k/videos"
