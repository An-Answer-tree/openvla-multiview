#!/usr/bin/env bash

set -eo pipefail

source /home/szliutong/miniconda3/etc/profile.d/conda.sh
conda create -n tmp python=3.10 -y
conda activate tmp
conda install nvidia::cuda==13.0.0 -y

OPENVLA_ROOT="/home/szliutong/Projects/openvla"
LIBERO_ROOT="/home/szliutong/Projects/LIBERO"
MAX_JOBS="4"

export TOKENIZERS_PARALLELISM=false

cd "${OPENVLA_ROOT}"
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install -e "${OPENVLA_ROOT}"
uv pip install packaging ninja
env MAX_JOBS="${MAX_JOBS}" uv pip install "flash-attn==2.8.3" --no-build-isolation --no-deps --verbose
uv pip install -r "${OPENVLA_ROOT}/experiments/robot/libero/libero_requirements.txt"
uv pip install -e "${LIBERO_ROOT}/third_party/robosuite"
uv pip install -e "${LIBERO_ROOT}"
python -c "import experiments.robot.libero.run_libero_eval as m; print(m.GenerateConfig)"
