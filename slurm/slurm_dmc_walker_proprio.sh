#!/bin/bash
#SBATCH --job-name=dreamer-dmc-proprio
#SBATCH --output=logs/dmc-walker-proprio/out_%j.txt
#SBATCH --error=logs/dmc-walker-proprio/err_%j.txt
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

JOB_TAG="dmc_walker_walk_proprio"
SLURM_LOG_DIR="logs/dmc-walker-proprio"
LOGDIR="logdir/dreamer/dmc_walker_walk_proprio_size1m"
CONFIGS="dmc_proprio,size1m"
TASK="dmc_walker_walk"
STEPS="1.1e6"
TRAIN_RATIO="1024"
LOG_EVERY="120"
REPORT_EVERY="300"
SAVE_EVERY="900"

source "$(dirname "${BASH_SOURCE[0]}")/slurm_common.sh"

"$VENV_PY" -c "import dm_control" 2>/dev/null || {
  echo "Installing dm_control..."
  "$VENV_PY" -m pip install dm_control
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
