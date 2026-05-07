"""Architecture for ``laion/music-aesthetics``.

This mirrors the small architecture published with the Hugging Face model:
segment-pooled ``laion/music-whisper`` encoder features are projected through a
shared bottleneck, then five metric-specific MLP heads produce 1--5 scores.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MusicAestheticsModel(nn.Module):
    def __init__(self, input_dim: int = 23040, bottleneck_dim: int = 256, hidden_dim: int = 64):
        super().__init__()
        self.metrics = ["Coherence", "Musicality", "Memorability", "Clarity", "Naturalness"]

        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, bottleneck_dim),
            nn.ReLU(),
            nn.LayerNorm(bottleneck_dim),
        )

        self.heads = nn.ModuleDict({
            metric: nn.Sequential(
                nn.Linear(bottleneck_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            for metric in self.metrics
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.bottleneck(x)
        return {metric: head(z).squeeze(-1) for metric, head in self.heads.items()}
