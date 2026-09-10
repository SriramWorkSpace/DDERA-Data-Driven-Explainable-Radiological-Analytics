"""DenseNet-121 feature encoder (ADR-001).

ImageNet-pretrained DenseNet-121 with the classifier removed: image -> 1024-d visual
representation. This is the **only** component that turns pixels into features; nothing here
maps features to a concept or a target (Invariants 3/4). The concept head and the
interpretable reasoner are Phase 3.

Frozen by default (ADR-008: cached features are the standard substrate for every
frozen-encoder CBM variant). B0 and the joint CBM (M3) fine-tune the encoder and construct
it with ``frozen=False`` -- and must not use the feature cache.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import densenet121

from ddera.config import EncoderConfig


class DenseNet121Encoder(nn.Module):
    """``image -> 1024-d`` DenseNet-121 features. See module docstring for the frozen policy."""

    def __init__(self, config: EncoderConfig | None = None, *, frozen: bool = True) -> None:
        super().__init__()
        self.config = config or EncoderConfig()
        backbone = densenet121(weights=self.config.weights)
        self.features = backbone.features  # conv tower -> (B, 1024, R/32, R/32)
        self.frozen = frozen
        if frozen:
            self.eval()
            for param in self.parameters():
                param.requires_grad_(False)

    @property
    def feature_dim(self) -> int:
        return self.config.feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mirrors torchvision DenseNet.forward up to (but not including) the classifier.
        out = self.features(x)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)  # (B, 1024)

    @torch.no_grad()
    def weights_sha256(self) -> str:
        """Stable hash of the encoder's parameters -- the identity in the cache fingerprint."""
        digest = hashlib.sha256()
        for name, tensor in sorted(self.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def fingerprint(self) -> dict[str, Any]:
        """Everything that must match for a cached feature set to be valid (ADR-008)."""
        return {
            "arch": self.config.arch,
            "weights": self.config.weights,
            "weights_sha256": self.weights_sha256(),
            "resolution": self.config.resolution,
            "channels": self.config.channels,
            "feature_dim": self.config.feature_dim,
            "normalization_mean": list(self.config.normalization_mean),
            "normalization_std": list(self.config.normalization_std),
            "frozen": self.frozen,
        }
