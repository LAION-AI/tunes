"""Architecture for ``laion/music-popularity``."""

from __future__ import annotations

import torch
import torch.nn as nn


class PopularityMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(23040, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
        )
        self.play_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.upvote_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.bottleneck(x)
        return self.play_head(feat).squeeze(-1), self.upvote_head(feat).squeeze(-1)
