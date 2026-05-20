from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

MAX_GRID_SIZE = 30
NUM_COLORS = 10


@dataclass(frozen=True)
class ArcSample:
  source: str
  split: str
  task_id: str
  pair_id: str
  input_grid: List[List[int]]
  output_grid: List[List[int]]


def _read_task(path: Path, source: str, split: str) -> List[ArcSample]:
  with path.open("r", encoding="utf-8") as f:
    task = json.load(f)
  samples: List[ArcSample] = []
  for key in ("train", "test"):
    for idx, pair in enumerate(task.get(key, [])):
      if "output" not in pair:
        continue
      samples.append(
          ArcSample(
              source=source,
              split=split,
              task_id=path.stem,
              pair_id=f"{key}_{idx}",
              input_grid=pair["input"],
              output_grid=pair["output"],
          )
      )
  return samples


def _load_from_root(root: Path, source: str, split_name: str) -> List[ArcSample]:
  split_path = root / "data" / split_name
  if not split_path.exists():
    return []
  all_samples: List[ArcSample] = []
  for file_path in sorted(split_path.glob("*.json")):
    all_samples.extend(_read_task(file_path, source=source, split=split_name))
  return all_samples


def load_arc_samples(arc_agi_1_root: Path, arc_agi_2_root: Path) -> Dict[str, List[ArcSample]]:
  train_samples = []
  eval_samples = []

  # ARC-AGI-1 contains only training data in this repository.
  train_samples.extend(_load_from_root(arc_agi_1_root, source="arc_agi_1", split_name="training"))

  # ARC-AGI-2 provides training and evaluation splits.
  train_samples.extend(_load_from_root(arc_agi_2_root, source="arc_agi_2", split_name="training"))
  eval_samples.extend(_load_from_root(arc_agi_2_root, source="arc_agi_2", split_name="evaluation"))

  return {"train": train_samples, "eval": eval_samples}


def split_train_val(samples: Sequence[ArcSample], val_ratio: float, seed: int) -> Tuple[List[ArcSample], List[ArcSample]]:
  if not 0.0 < val_ratio < 1.0:
    raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
  task_ids = sorted({(s.source, s.task_id) for s in samples})
  rng = random.Random(seed)
  rng.shuffle(task_ids)
  n_val = max(1, int(len(task_ids) * val_ratio))
  val_tasks = set(task_ids[:n_val])
  train = [s for s in samples if (s.source, s.task_id) not in val_tasks]
  val = [s for s in samples if (s.source, s.task_id) in val_tasks]
  return train, val


def _pad_grid(grid: List[List[int]]) -> torch.Tensor:
  h = min(len(grid), MAX_GRID_SIZE)
  w = min(len(grid[0]), MAX_GRID_SIZE)
  out = torch.zeros((MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=torch.long)
  out[:h, :w] = torch.tensor([row[:w] for row in grid[:h]], dtype=torch.long)
  return out


class ArcGridDataset(Dataset):
  def __init__(self, samples: Sequence[ArcSample]):
    self.samples = list(samples)

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, idx: int):
    sample = self.samples[idx]
    in_h, in_w = len(sample.input_grid), len(sample.input_grid[0])
    out_h, out_w = len(sample.output_grid), len(sample.output_grid[0])
    return {
        "input_grid": _pad_grid(sample.input_grid),
        "output_grid": _pad_grid(sample.output_grid),
        "input_hw": torch.tensor([in_h - 1, in_w - 1], dtype=torch.long),
        "output_hw": torch.tensor([out_h - 1, out_w - 1], dtype=torch.long),
        "meta": {
            "source": sample.source,
            "split": sample.split,
            "task_id": sample.task_id,
            "pair_id": sample.pair_id,
        },
    }