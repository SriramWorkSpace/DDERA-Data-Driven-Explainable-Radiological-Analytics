"""B0 black-box baseline (ARCHITECTURE section 5).

``encoder -> Linear(1024 -> 1)``. This is **not** an ante-hoc model and is never presented
as DDERA's method: it exists only as the accuracy ceiling, so the interpretability cost of
forcing the decision through 12 concepts can be measured. Per the Invariant 3/4 note in
ARCHITECTURE section 5, B0 (with M4) is the only model with a direct features -> target path.
"""

from __future__ import annotations

import torch
from torch import nn


class BlackBoxModel(nn.Module):
    """``features -> target logit`` with no concept bottleneck."""

    def __init__(self, *, encoder: nn.Module | None = None, in_features: int = 1024) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else nn.Identity()
        self.classifier = nn.Linear(in_features, 1)

    def features(self, batch: dict) -> torch.Tensor:
        if "features" in batch:
            return batch["features"]
        return self.encoder(batch["image"])

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        return {"target_logit": self.classifier(self.features(batch)).squeeze(-1)}
