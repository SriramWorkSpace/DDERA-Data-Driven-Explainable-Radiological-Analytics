"""The interpretable reasoner (ARCHITECTURE stage 7).

``Linear(n_concepts -> 1)``: ``logit(p) = w . c + b``. The weight vector *is* the
explanation. Because the map is linear the logit decomposes exactly:

    logit(p) = b + sum_j  w_j * c_j

Each ``w_j * c_j`` is one concept's signed contribution -- what the dashboard waterfall
shows, and what ``tests/test_cbm_model.py`` asserts sums back to the logit. No attribution
method, no sampling: the explanation is an arithmetic identity.

:meth:`to_numpy_reasoner` bridges to :class:`ddera.xai.intervention.LinearReasoner` so the
entire (domain-agnostic) XAI harness runs on a trained model unchanged.
"""

from __future__ import annotations

import torch
from torch import nn


class LinearReasoner(nn.Module):
    """``concepts in [0, 1]^k  ->  target logit``. Deliberately transparent."""

    def __init__(self, n_concepts: int = 12, *, bias: bool = True) -> None:
        super().__init__()
        self.n_concepts = n_concepts
        self.linear = nn.Linear(n_concepts, 1, bias=bias)

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        return self.linear(concepts).squeeze(-1)

    @property
    def weight(self) -> torch.Tensor:
        """The ``(k,)`` weight vector -- the explanation."""
        return self.linear.weight.detach().reshape(-1)

    @property
    def bias_value(self) -> float:
        return float(self.linear.bias.detach()) if self.linear.bias is not None else 0.0

    def contributions(self, concepts: torch.Tensor) -> torch.Tensor:
        """``(n, k)`` signed contributions ``w_j * c_j``; these + bias sum to the logit."""
        return concepts * self.linear.weight.reshape(-1)

    def to_numpy_reasoner(self, concept_names: list[str] | None = None):
        """A :class:`ddera.xai.intervention.LinearReasoner` with the trained ``w`` and ``b``."""
        from ddera.xai.intervention import LinearReasoner as NumpyLinearReasoner

        return NumpyLinearReasoner(self.weight.cpu().numpy(), self.bias_value, concept_names)
