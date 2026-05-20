#!/bin/bash
#SBATCH --job-name=arc-trm
#SBATCH --output=logs/arc-trm/out_%j.txt
#SBATCH --error=logs/arc-trm/err_%j.txt
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

set -euo pipefail

echo "Home directory: $HOME"
echo "Current working directory: $PWD"
mkdir -p "$PWD/logdir/arc_transformer" "$PWD/logs/arc-trm"

module unload cuda || true
module load cuda/12.6

export TMPDIR="$PWD/tmp/$SLURM_JOB_ID"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export WANDB_DIR="$PWD/logdir/arc_transformer/wandb"
export PYTHONUNBUFFERED=1

VENV_PY="$PWD/.venv/bin/python3"
if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: venv python not found at $VENV_PY"
  echo "Please run: python3 -m venv .venv && .venv/bin/python3 -m pip install -U pip torch wandb tqdm"
  exit 1
fi

echo "===== JOB INFO ====="
echo "HOSTNAME: $(hostname)"
echo "DATE: $(date)"
echo "PWD: $(pwd)"
echo "PYTHON: $VENV_PY"
"$VENV_PY" --version
nvidia-smi || true
echo "===================="

# Optional: set your W&B entity outside this script if needed.
# export WANDB_ENTITY=your_team_or_user

"$VENV_PY" -m arc_agi.train_transformer \
  --arc-agi-1-root "$PWD/ARC-AGI" \
  --arc-agi-2-root "$PWD/ARC-AGI-2" \
  --batch-size 32 \
  --epochs 40 \
  --lr 3e-4 \
  --d-model 256 \
  --nhead 8 \
  --num-layers 6 \
  --ffn-dim 1024 \
  --outdir "$PWD/logdir/arc_transformer/run_$SLURM_JOB_ID" \
  --project arc-agi-transformer \
  --run-name "arc-trm-${SLURM_JOB_ID}" \
  --wandb

