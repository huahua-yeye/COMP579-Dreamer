#!/bin/bash
#SBATCH --job-name=dreamerv3
#SBATCH --output=logs/out_%j.txt
#SBATCH --error=logs/err_%j.txt
#SBATCH --time=12:00:00
#SBATCH --partition=comp579-1gpu-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail

# 1) 进入项目目录
cd ~/dreamerv3

# 2) 准备日志目录
mkdir -p logs
mkdir -p ~/logdir/dreamer

# 3) 加载 CUDA（可选：cuda/cuda-11.8）
module load cuda/cuda-12.6

# 4) 激活本地 venv
source .venv/bin/activate

# 5) 打印环境信息，方便排错
echo "===== JOB INFO ====="
echo "HOSTNAME: $(hostname)"
echo "DATE: $(date)"
echo "PWD: $(pwd)"
echo "PYTHON: $(which python)"
python --version
nvidia-smi || true
echo "===================="

# 6) 检查 JAX 是否看到 GPU
python - <<'PY'
import jax
print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("JAX backend:", jax.default_backend())
PY

# 7) 开始训练
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/run1 \
  --configs atari100k,size1m \
  --run.steps 100000 \
  --run.train_ratio 64 \
  --run.log_every 60 \
  --run.report_every 120
