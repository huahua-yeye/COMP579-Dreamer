from __future__ import annotations

import torch
import torch.nn as nn

from arc_agi.data import MAX_GRID_SIZE, NUM_COLORS


class ArcTransformer(nn.Module):
  def __init__(
      self,
      d_model: int = 256,
      nhead: int = 8,
      num_layers: int = 6,
      dim_feedforward: int = 1024,
      dropout: float = 0.1,
  ):
    super().__init__()
    self.max_grid_size = MAX_GRID_SIZE
    self.num_positions = MAX_GRID_SIZE * MAX_GRID_SIZE

    self.color_emb = nn.Embedding(NUM_COLORS, d_model)
    self.row_emb = nn.Embedding(MAX_GRID_SIZE, d_model)
    self.col_emb = nn.Embedding(MAX_GRID_SIZE, d_model)

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=True,
    )
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    self.out_color_head = nn.Linear(d_model, NUM_COLORS)
    self.out_h_head = nn.Linear(d_model, MAX_GRID_SIZE)
    self.out_w_head = nn.Linear(d_model, MAX_GRID_SIZE)

  def _add_pos_emb(self, x: torch.Tensor) -> torch.Tensor:
    batch_size = x.shape[0]
    rows = torch.arange(self.max_grid_size, device=x.device)
    cols = torch.arange(self.max_grid_size, device=x.device)
    row_grid = rows[:, None].expand(self.max_grid_size, self.max_grid_size).reshape(-1)
    col_grid = cols[None, :].expand(self.max_grid_size, self.max_grid_size).reshape(-1)
    pos = self.row_emb(row_grid) + self.col_emb(col_grid)
    return x + pos.unsqueeze(0).expand(batch_size, -1, -1)

  def forward(self, input_grid: torch.Tensor):
    batch_size = input_grid.shape[0]
    x = input_grid.view(batch_size, -1)
    x = self.color_emb(x)
    x = self._add_pos_emb(x)
    hidden = self.encoder(x)

    color_logits = self.out_color_head(hidden).view(
        batch_size, self.max_grid_size, self.max_grid_size, NUM_COLORS
    )
    pooled = hidden.mean(dim=1)
    h_logits = self.out_h_head(pooled)
    w_logits = self.out_w_head(pooled)
    return color_logits, h_logits, w_logits

