#!/bin/bash
#SBATCH --job-name=dreamer-dmlab
#SBATCH --output=logs/dmlab-goal/out_%j.txt
#SBATCH --error=logs/dmlab-goal/err_%j.txt
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32

# DeepMind Lab. Requires DMLab install (see Dockerfile install-dmlab.sh).
# Long run (~26M steps).
JOB_TAG="dmlab_explore_goal"
SLURM_LOG_DIR="logs/dmlab-goal"
LOGDIR="logdir/dreamer/dmlab_explore_goal_locations_small_size12m"
CONFIGS="dmlab,size12m"
TASK="dmlab_explore_goal_locations_small"
STEPS="2.6e7"
TRAIN_RATIO="32"
LOG_EVERY="120"
REPORT_EVERY="300"
SAVE_EVERY="900"

source "$(dirname "${BASH_SOURCE[0]}")/slurm_common.sh"

"$VENV_PY" -c "import deepmind_lab" 2>/dev/null || {
  echo "ERROR: deepmind_lab not installed. See Dockerfile / install-dmlab.sh"
  exit 1
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
