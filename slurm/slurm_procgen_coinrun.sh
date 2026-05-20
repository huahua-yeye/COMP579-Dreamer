#!/bin/bash
#SBATCH --job-name=dreamer-procgen
#SBATCH --output=logs/procgen-coinrun/out_%j.txt
#SBATCH --error=logs/procgen-coinrun/err_%j.txt
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32

# ProcGen coinrun. Extra dep: pip install procgen_mirror
JOB_TAG="procgen_coinrun"
SLURM_LOG_DIR="logs/procgen-coinrun"
LOGDIR="logdir/dreamer/procgen_coinrun_size12m"
CONFIGS="procgen,size12m"
TASK="procgen_coinrun"
STEPS="1.1e8"
TRAIN_RATIO="64"
LOG_EVERY="120"
REPORT_EVERY="300"
SAVE_EVERY="900"

source "$(dirname "${BASH_SOURCE[0]}")/slurm_common.sh"

"$VENV_PY" -c "import procgen" 2>/dev/null || {
  echo "Installing procgen_mirror..."
  "$VENV_PY" -m pip install procgen_mirror
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
