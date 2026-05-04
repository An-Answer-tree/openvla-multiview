#!/bin/bash
#SBATCH --job-name=eval-spatial-7views-40k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=25
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --qos=high_priority
#SBATCH --exclude=gpu03
#SBATCH --output=/home/liutong/Projects/openvla-multiview/logs/eval/spatial/%x-%j.log

set -eo pipefail

source /home/liutong/.bashrc
conda activate openvla

export http_proxy="http://localhost:7878"
export https_proxy="http://localhost:7878"
export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_ONEDNN_OPTS=0

cd /home/liutong/Projects/openvla-multiview

python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "/opt/liutong/openvla-runs/multiview/spatial/7views/openvla-7b+libero_spatial+cam-agentview+topview+leftview+rightview+backview+leftbackview+rightbackview+b16+lr-0.0005+seed-7+lora-r32+dropout-0.0+sched-linear-warmup+cosine-decay+warmup-0.03--image_aug--40000_chkpt" \
    --camera_names "[agentview,topview,leftview,rightview,backview,leftbackview,rightbackview]" \
    --local_log_dir "/opt/liutong/multiview_test/multiview/spatial/7views/40k" \
    --rollout_root_dir "/opt/liutong/multiview_test/multiview/spatial/7views/40k/videos"
