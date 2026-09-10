"""Concept prediction layer (ARCHITECTURE stage 5).

``Linear(1024 -> n_concepts)``. Returns concept **logits**; the probability head is a plain
sigmoid. This is the only place the encoder representation is read; everything downstream
sees only the ``n_concepts``-wide vector (Invariant 4).
"""

from __future__ import annotations

import torch
from torch import nn


class ConceptHead(nn.Module):
    """``features -> concept logits``. Trained with masked BCE against the concept labels."""

    def __init__(
        self, in_features: int = 1024, n_concepts: int = 12, *, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.n_concepts = n_concepts
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(in_features, n_concepts)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(features))

    @staticmethod
    def probabilities(concept_logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(concept_logits)
