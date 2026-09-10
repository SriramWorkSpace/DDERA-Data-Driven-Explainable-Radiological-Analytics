"""Autocast context (ARCHITECTURE section 7).

bf16 autocast is preferred where the backend supports it (``ddera.device`` selects the
dtype); fp16 is the fallback; CPU runs full precision. Gradient accumulation lives in the
training loop, not here.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from ddera.device import DeviceInfo

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass(frozen=True)
class AmpContext:
    enabled: bool
    dtype: torch.dtype
    device_type: str

    @classmethod
    def disabled(cls) -> AmpContext:
        return cls(False, torch.float32, "cpu")

    @classmethod
    def from_device(cls, info: DeviceInfo) -> AmpContext:
        dtype = _DTYPES.get(info.amp_dtype or "")
        enabled = bool(info.supports_amp and dtype is not None and info.device.type == "cuda")
        return cls(enabled, dtype or torch.float32, info.device.type)

    def autocast(self):
        if not self.enabled:
            return nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=self.dtype)
