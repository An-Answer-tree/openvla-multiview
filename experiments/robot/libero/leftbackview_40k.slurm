#!/bin/bash
#SBATCH --job-name=eval-spatial-leftbackview-40k
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

cd /home/liutong/Projects/openvla-multiview

python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "/opt/liutong/openvla-runs/leftbackview/openvla-7b+libero_spatial+cam-leftbackview+b8+lr-0.0005+seed-7+lora-r32+dropout-0.0--image_aug--40000_chkpt" \
    --camera_names "[leftbackview]" \
    --local_log_dir "/opt/liutong/multiview_test/leftbackview_40k" \
    --rollout_root_dir "/opt/liutong/multiview_test/leftbackview_40k/videos"
