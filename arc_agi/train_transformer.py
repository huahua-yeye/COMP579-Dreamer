from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from arc_agi.data import ArcGridDataset, MAX_GRID_SIZE, load_arc_samples, split_train_val
from arc_agi.model import ArcTransformer

try:
  import wandb
except ImportError:
  wandb = None


def parse_args():
  parser = argparse.ArgumentParser(description="Train ARC-AGI transformer baseline.")
  parser.add_argument("--arc-agi-1-root", type=Path, default=Path("ARC-AGI"))
  parser.add_argument("--arc-agi-2-root", type=Path, default=Path("ARC-AGI-2"))
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--epochs", type=int, default=40)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  parser.add_argument("--val-ratio", type=float, default=0.1)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--num-workers", type=int, default=4)
  parser.add_argument("--d-model", type=int, default=256)
  parser.add_argument("--nhead", type=int, default=8)
  parser.add_argument("--num-layers", type=int, default=6)
  parser.add_argument("--ffn-dim", type=int, default=1024)
  parser.add_argument("--dropout", type=float, default=0.1)
  parser.add_argument("--log-every", type=int, default=20)
  parser.add_argument("--outdir", type=Path, default=Path("logdir/arc_transformer"))
  parser.add_argument("--project", type=str, default="arc-agi-transformer")
  parser.add_argument("--run-name", type=str, default="")
  parser.add_argument("--wandb", action="store_true")
  parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
  return parser.parse_args()


def set_seed(seed: int):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


def _shape_mask(out_h: torch.Tensor, out_w: torch.Tensor) -> torch.Tensor:
  b = out_h.shape[0]
  rows = torch.arange(MAX_GRID_SIZE, device=out_h.device).view(1, MAX_GRID_SIZE, 1)
  cols = torch.arange(MAX_GRID_SIZE, device=out_h.device).view(1, 1, MAX_GRID_SIZE)
  return (rows <= out_h.view(b, 1, 1)) & (cols <= out_w.view(b, 1, 1))


def _compute_batch_metrics(color_logits, h_logits, w_logits, batch) -> Dict[str, float]:
  target_grid = batch["output_grid"]
  target_h = batch["output_hw"][:, 0]
  target_w = batch["output_hw"][:, 1]

  color_pred = color_logits.argmax(dim=-1)
  h_pred = h_logits.argmax(dim=-1)
  w_pred = w_logits.argmax(dim=-1)

  full_match = (color_pred == target_grid).all(dim=(-1, -2)).float().mean().item()
  shape_match = ((h_pred == target_h) & (w_pred == target_w)).float().mean().item()

  pred_mask = _shape_mask(h_pred, w_pred)
  true_mask = _shape_mask(target_h, target_w)
  overlap = pred_mask & true_mask
  union = pred_mask | true_mask
  iou = (overlap.float().sum(dim=(-1, -2)) / union.float().sum(dim=(-1, -2)).clamp(min=1.0)).mean().item()
  return {
      "pixel_full_match": full_match,
      "shape_exact_match": shape_match,
      "shape_iou": iou,
  }


def run_epoch(model, loader, optimizer, device, train: bool):
  model.train(train)
  total_loss = 0.0
  total_batches = 0
  aggregate = {"pixel_full_match": 0.0, "shape_exact_match": 0.0, "shape_iou": 0.0}

  for batch in loader:
    input_grid = batch["input_grid"].to(device)
    target_grid = batch["output_grid"].to(device)
    target_h = batch["output_hw"][:, 0].to(device)
    target_w = batch["output_hw"][:, 1].to(device)

    with torch.set_grad_enabled(train):
      color_logits, h_logits, w_logits = model(input_grid)
      color_loss = F.cross_entropy(color_logits.permute(0, 3, 1, 2), target_grid)
      h_loss = F.cross_entropy(h_logits, target_h)
      w_loss = F.cross_entropy(w_logits, target_w)
      loss = color_loss + 0.3 * (h_loss + w_loss)

      if train:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    metrics = _compute_batch_metrics(color_logits.detach(), h_logits.detach(), w_logits.detach(), {
        "output_grid": target_grid,
        "output_hw": torch.stack([target_h, target_w], dim=-1),
    })
    total_loss += loss.item()
    total_batches += 1
    for key, value in metrics.items():
      aggregate[key] += value

  denom = max(total_batches, 1)
  out = {"loss": total_loss / denom}
  out.update({k: v / denom for k, v in aggregate.items()})
  return out


def main():
  args = parse_args()
  set_seed(args.seed)
  args.outdir.mkdir(parents=True, exist_ok=True)

  if args.wandb and wandb is None:
    raise ImportError("wandb is not installed. Install it with `pip install wandb`.")

  samples = load_arc_samples(args.arc_agi_1_root, args.arc_agi_2_root)
  train_samples, val_samples = split_train_val(samples["train"], val_ratio=args.val_ratio, seed=args.seed)
  eval_samples = samples["eval"]

  print(f"Loaded train={len(train_samples)} val={len(val_samples)} eval={len(eval_samples)} pairs")

  train_loader = DataLoader(
      ArcGridDataset(train_samples),
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      pin_memory=True,
  )
  val_loader = DataLoader(
      ArcGridDataset(val_samples),
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=True,
  )
  eval_loader = DataLoader(
      ArcGridDataset(eval_samples),
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=True,
  )

  device = torch.device(args.device)
  model = ArcTransformer(
      d_model=args.d_model,
      nhead=args.nhead,
      num_layers=args.num_layers,
      dim_feedforward=args.ffn_dim,
      dropout=args.dropout,
  ).to(device)

  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

  wb = None
  if args.wandb:
    wb = wandb.init(
        project=args.project,
        name=args.run_name or None,
        config=vars(args),
    )

  best_val = -math.inf
  best_ckpt = args.outdir / "best.pt"

  for epoch in range(1, args.epochs + 1):
    print(f"\nEpoch {epoch}/{args.epochs}")
    train_metrics = run_epoch(model, tqdm(train_loader, desc="train"), optimizer, device, train=True)
    val_metrics = run_epoch(model, tqdm(val_loader, desc="val"), optimizer, device, train=False)
    eval_metrics = run_epoch(model, tqdm(eval_loader, desc="eval"), optimizer, device, train=False)
    scheduler.step()

    log_data = {
        "epoch": epoch,
        "lr": scheduler.get_last_lr()[0],
        **{f"train/{k}": v for k, v in train_metrics.items()},
        **{f"val/{k}": v for k, v in val_metrics.items()},
        **{f"eval/{k}": v for k, v in eval_metrics.items()},
    }
    print(log_data)
    if wb is not None:
      wb.log(log_data, step=epoch)

    if val_metrics["pixel_full_match"] > best_val:
      best_val = val_metrics["pixel_full_match"]
      torch.save(
          {
              "model_state_dict": model.state_dict(),
              "optimizer_state_dict": optimizer.state_dict(),
              "args": vars(args),
              "epoch": epoch,
              "val_metrics": val_metrics,
          },
          best_ckpt,
      )
      print(f"Saved best checkpoint: {best_ckpt}")

  if wb is not None:
    wb.summary["best_val_pixel_full_match"] = best_val
    wb.finish()


if __name__ == "__main__":
  main()

