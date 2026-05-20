#!/bin/bash
#SBATCH --job-name=dreamer-atari-pong
#SBATCH --output=logs/atari-pong/out_%j.txt
#SBATCH --error=logs/atari-pong/err_%j.txt
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32

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

install_atari_roms_if_needed() {
  local rom
  rom="$("$VENV_PY" -c "import os, ale_py; print(os.path.join(os.path.dirname(ale_py.__file__), 'roms', 'pong.bin'))")"
  if [ -f "$rom" ]; then
    echo "Atari ROMs OK ($rom)"
    return
  fi
  echo "Atari ROMs missing; installing with AutoROM (requires network once)..."
  if ! "$VENV_PY" -m pip show autorom >/dev/null 2>&1; then
    echo "Installing autorom..."
    "$VENV_PY" -m pip install 'autorom[accept-rom-license]==0.6.1'
  fi
  AUTO_ROM="$PWD/.venv/bin/AutoROM"
  if [ ! -x "$AUTO_ROM" ]; then
    echo "ERROR: AutoROM not found at $AUTO_ROM"
    exit 1
  fi
  ALE_ROMS="$("$VENV_PY" -c "import os, ale_py; print(os.path.join(os.path.dirname(ale_py.__file__), 'roms'))")"
  "$AUTO_ROM" --accept-license -d "$ALE_ROMS"
  if [ ! -f "$rom" ]; then
    echo "ERROR: ROM install failed; pong.bin still missing at $rom"
    exit 1
  fi
  echo "Atari ROMs installed."
}
install_atari_roms_if_needed

# Full Atari training (~51M steps). Needs: ale_py + AutoROM ROMs.
JOB_TAG="atari_pong"
SLURM_LOG_DIR="logs/atari-pong"
LOGDIR="logdir/dreamer/atari_pong_size1m"
CONFIGS="atari,size1m"
TASK="atari_pong"
STEPS="5.1e7"
TRAIN_RATIO="32"
LOG_EVERY="120"
REPORT_EVERY="300"
SAVE_EVERY="900"

"$VENV_PY" dreamerv3/main.py \
  --logdir "$LOGDIR" \
  --configs "$CONFIGS" \
  --task "$TASK" \
  --run.steps "$STEPS" \
  --run.train_ratio "$TRAIN_RATIO" \
  --run.log_every "$LOG_EVERY" \
  --run.report_every "$REPORT_EVERY" \
  --run.save_every "$SAVE_EVERY"