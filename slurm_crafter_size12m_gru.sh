#!/bin/bash
#SBATCH --job-name=dreamer-gru
#SBATCH --output=logs/gru-12m/out_%j.txt
#SBATCH --error=logs/gru-12m/err_%j.txt
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32

set -euo pipefail

echo "Home directory: $HOME"
echo "Current working directory: $PWD"
mkdir -p "$PWD/logdir/dreamer"

module unload cuda || true
# module load StdEnv/2023 intel/2023.2.1 cuda/11.8
# module load cuda/12.2
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

# 7) 开始训练
# "$VENV_PY" dreamerv3/main.py \
#   --logdir ~/logdir/dreamer/run1 \
#   --configs crafter \
#   #--jax.platform cpu
#   --run.steps 10000 \
#   --run.train_ratio 64 \
#   --run.log_every 60 \
#   --run.report_every 120


# "$VENV_PY" dreamerv3/main.py \
#   --logdir ~/logdir/dreamer/atari100k_pong_6h \
#   --configs atari100k,size1m \
#   --task atari100k_pong \
#   --run.steps 100000 \
#   --run.train_ratio 128 \
#   --run.steps 1.1e5 \
#   --run.train_ratio 256\
#   --run.log_every 60 \
#   --run.report_every 120 \
#   --run.save_every 600

# ============================================================
# Crafter ablation: GRU vs Transformer vs TRM (size12m)
# 任选其一运行
# ============================================================

# ---- (1) GRU baseline ----
"$VENV_PY" dreamerv3/main.py \
  --logdir ./logdir/dreamer/crafter_size12m_gru \
  --configs crafter,size12m \
  --run.report_every 300 \
  --run.steps 1.1e6 \
  --run.train_ratio 64 \
  --run.log_every 60 \
  --run.report_every 120 \
  --run.save_every 2000

# ---- (2) Transformer ----
# "$VENV_PY" dreamerv3/main.py \
#   --logdir ./logdir/dreamer/crafter_size12m_tx \
#   --configs crafter,size12m \
#   --run.log_every 60 \
#   --run.report_every 300 \
#   --run.save_every 1800 \
#   --agent.dyn.rssm.core transformer \
#   --agent.dyn.rssm.attn_heads 4 \
#   --agent.dyn.rssm.attn_layers 1

# ---- (3) TRM ----
# "$VENV_PY" dreamerv3/main.py \
#   --logdir ./logdir/dreamer/crafter_size12m_trm \
#   --configs crafter,size12m \
#   --run.log_every 60 \
#   --run.report_every 300 \
#   --run.save_every 1800 \
#   --agent.dyn.rssm.core trm \
#   --agent.dyn.rssm.attn_heads 4 \
#   --agent.dyn.rssm.trm_steps 4 \
#   --jax.compute_dtype float32