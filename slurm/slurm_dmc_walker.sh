#!/bin/bash
#SBATCH --job-name=dreamer-dmc-walker
#SBATCH --output=logs/dmc-walker/out_%j.txt
#SBATCH --error=logs/dmc-walker/err_%j.txt
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32

# DeepMind Control Suite (vision). Extra dep: pip install dm_control
# Use slurm_dmc_walker_proprio.sh for proprio-only (size1m, faster).
JOB_TAG="dmc_walker_walk"
SLURM_LOG_DIR="logs/dmc-walker"
LOGDIR="logdir/dreamer/dmc_walker_walk_size12m"
CONFIGS="dmc_vision,size12m"
TASK="dmc_walker_walk"
STEPS="1.1e6"
TRAIN_RATIO="256"
LOG_EVERY="120"
REPORT_EVERY="300"
SAVE_EVERY="900"


set -euo pipefail

echo "Home directory: $HOME"
echo "Current working directory: $PWD"
mkdir -p "$PWD/logdir/dreamer"

module unload cuda || true
module load cuda/12.6

export TMPDIR="$PWD/tmp/$SLURM_JOB_ID"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

VENV_PY="$PWD/.venv/bin/python3"
if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: venv python not found at $VENV_PY"
  echo "Please run: python3 -m venv .venv && .venv/bin/python3 -m pip install -U -r requirements.txt"
  exit 1
fi

echo "===== JOB INFO ====="
echo "HOSTNAME: $(hostname)"
echo "DATE: $(date)"
echo "PWD: $(pwd)"
echo "PYTHON: $VENV_PY"
"$VENV_PY" -c "import os; print('TMPDIR:', os.environ.get('TMPDIR'))"
"$VENV_PY" --version
nvidia-smi || true
echo "===================="

# "$VENV_PY" -m pip --version
# "$VENV_PY" -m pip show jax >/dev/null
"$VENV_PY" - <<'PY'
import jax
print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("JAX backend:", jax.default_backend())
PY

"$VENV_PY" dreamerv3/main.py \
  --logdir "$LOGDIR" \
  --configs "$CONFIGS" \
  --task "$TASK" \
  --run.steps "$STEPS" \
  --run.train_ratio "$TRAIN_RATIO" \
  --run.log_every "$LOG_EVERY" \
  --run.report_every "$REPORT_EVERY" \
  --run.save_every "$SAVE_EVERY"
