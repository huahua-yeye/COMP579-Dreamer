#!/bin/bash
#SBATCH --job-name=dreamerv3
#SBATCH --account=winter2026-comp579
#SBATCH --output=logs/out_%j.txt
#SBATCH --error=logs/err_%j.txt
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# --partition=gpu-grad-02
# darcos2
# cyu2
# sacctmgr list association where user=cyu2 format=Account,User,Description,Partition,Cluster

set -euo pipefail

# 1) 进入项目目录
cd ~/dreamerv3

# 2) 准备日志目录
mkdir -p logs
mkdir -p ~/logdir/dreamer
mkdir -p ~/tmp/$SLURM_JOB_ID
mkdir -p ~/tmp/jax_cache

# 3) 加载 CUDA（可选：cuda/cuda-11.8）
module unload cuda || true
module load cuda/cuda-11.8


# 让 JAX/XLA 不使用系统 /tmp（该目录常被其他作业占满）
export TMPDIR="$HOME/tmp/$SLURM_JOB_ID"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

# 4) 使用本地 venv 的绝对路径 Python，避免 PATH 问题
VENV_PY="$HOME/dreamerv3/.venv/bin/python3"
if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: venv python not found at $VENV_PY"
  echo "Please run: python3 -m venv .venv && .venv/bin/python3 -m pip install -U -r requirements.txt"
  exit 1
fi

# 5) 打印环境信息，方便排错
echo "===== JOB INFO ====="
echo "HOSTNAME: $(hostname)"
echo "DATE: $(date)"
echo "PWD: $(pwd)"
echo "PYTHON: $VENV_PY"
"$VENV_PY" -c "import os; print('TMPDIR:', os.environ.get('TMPDIR'))"
"$VENV_PY" --version
nvidia-smi || true
echo "===================="

# 6) 检查 JAX 是否看到 GPU
"$VENV_PY" -m pip --version
"$VENV_PY" -m pip show jax >/dev/null
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
# "$VENV_PY" dreamerv3/main.py \
#   --logdir ~/logdir/dreamer/crafter_size12m_gru \
#   --configs crafter,size12m \
#   --run.log_every 60 \
#   --run.report_every 300 \
#   --run.save_every 1800

# ---- (2) Transformer ----
# "$VENV_PY" dreamerv3/main.py \
#   --logdir ~/logdir/dreamer/crafter_size12m_tx \
#   --configs crafter,size12m \
#   --run.log_every 60 \
#   --run.report_every 300 \
#   --run.save_every 1800 \
#   --agent.dyn.rssm.core transformer \
#   --agent.dyn.rssm.attn_heads 4 \
#   --agent.dyn.rssm.attn_layers 1

# ---- (3) TRM ----
"$VENV_PY" dreamerv3/main.py \
  --logdir ~/logdir/dreamer/crafter_size12m_trm \
  --configs crafter,size12m \
  --run.log_every 60 \
  --run.report_every 300 \
  --run.save_every 1800 \
  --agent.dyn.rssm.core trm \
  --agent.dyn.rssm.attn_heads 4 \
  --agent.dyn.rssm.trm_steps 4 \
  --jax.compute_dtype float32