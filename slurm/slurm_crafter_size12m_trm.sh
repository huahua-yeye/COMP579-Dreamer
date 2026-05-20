#!/bin/bash
#SBATCH --job-name=dreamer-trm12m
#SBATCH --output=logs/trm-12m/out_%j.txt
#SBATCH --error=logs/trm-12m/err_%j.txt
#SBATCH --time=1-00:00:00
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

"$VENV_PY" - <<'PY'
import jax
print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("JAX backend:", jax.default_backend())
PY

# ============================================================
# Crafter ablation: GRU vs Transformer vs TRM (size12m)
# 任选其一运行
# ============================================================
# # ---- (3) TRM ----
"$VENV_PY" dreamerv3/main.py \
  --logdir ./logdir/dreamer/crafter_size12m_trm \
  --configs crafter,size12m \
  --run.steps 1.1e6 \
  --run.log_every 60 \
  --run.report_every 300 \
  --run.save_every 2000 \
  --agent.dyn.rssm.core trm \
  --agent.dyn.rssm.attn_heads 4 \
  --agent.dyn.rssm.trm_steps 4 \
  --jax.compute_dtype float32