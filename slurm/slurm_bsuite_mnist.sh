#!/bin/bash
#SBATCH --job-name=dreamer-bsuite
#SBATCH --output=logs/bsuite-mnist/out_%j.txt
#SBATCH --error=logs/bsuite-mnist/err_%j.txt
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

# Behavior Suite MNIST. Extra dep: pip install dm_env bsuite
JOB_TAG="bsuite_mnist"
SLURM_LOG_DIR="logs/bsuite-mnist"
LOGDIR="logdir/dreamer/bsuite_mnist_0_size1m"
CONFIGS="bsuite,size1m"
TASK="bsuite_mnist/0"
STEPS="1.1e6"
TRAIN_RATIO="1024"
LOG_EVERY="60"
REPORT_EVERY="120"
SAVE_EVERY="-1"

source "$(dirname "${BASH_SOURCE[0]}")/slurm_common.sh"

"$VENV_PY" -c "import bsuite" 2>/dev/null || {
  echo "Installing bsuite..."
  "$VENV_PY" -m pip install bsuite
}

"$VENV_PY" dreamerv3/main.py \
  --logdir "$LOGDIR" \
  --configs "$CONFIGS" \
  --task "$TASK" \
  --run.steps "$STEPS" \
  --run.train_ratio "$TRAIN_RATIO" \
  --run.log_every "$LOG_EVERY" \
  --run.report_every "$REPORT_EVERY" \
  --run.save_every "$SAVE_EVERY"
