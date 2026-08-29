"""Backend resolution.

This is the ONLY module in the codebase that branches on hardware. Everything else takes a
``torch.device`` and stays backend-agnostic.

Backend priority: ROCm/CUDA (both expose ``torch.cuda``) -> DirectML -> CPU.

Per Invariants 7 and 8, the backend may change but the methodology may not. Nothing in this
module may alter model architecture -- only where tensors live and at what precision.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

Backend = Literal["rocm", "cuda", "directml", "cpu"]

#: gfx1031 (RX 6800M / Navi 22) is not an officially supported ROCm target, but gfx1030
#: (Navi 21) IS officially packaged in current ROCm and built into the PyTorch ROCm wheels.
#: This override makes gfx1031 load those shipped gfx1030 code objects. See ADR-009.
GFX_OVERRIDE_ENV = "HSA_OVERRIDE_GFX_VERSION"
GFX1030_OVERRIDE = "10.3.0"


@dataclass(frozen=True)
class DeviceInfo:
    """A description of the resolved compute backend, safe to serialise into a run config."""

    backend: Backend
    device: torch.device
    name: str
    torch_version: str
    hip_version: str | None = None
    cuda_version: str | None = None
    total_memory_gb: float | None = None
    gfx_override: str | None = None
    supports_amp: bool = False
    amp_dtype: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["device"] = str(self.device)
        return d

    def summary(self) -> str:
        lines = [
            f"Backend      : {self.backend}",
            f"Device       : {self.device}  ({self.name})",
            f"torch        : {self.torch_version}",
        ]
        if self.hip_version:
            lines.append(f"HIP (ROCm)   : {self.hip_version}")
        if self.cuda_version:
            lines.append(f"CUDA         : {self.cuda_version}")
        if self.total_memory_gb is not None:
            lines.append(f"VRAM         : {self.total_memory_gb:.1f} GB")
        if self.gfx_override:
            lines.append(f"GFX override : {GFX_OVERRIDE_ENV}={self.gfx_override}")
        lines.append(f"AMP          : {self.amp_dtype if self.supports_amp else 'unavailable'}")
        for note in self.notes:
            lines.append(f"  ! {note}")
        return "\n".join(lines)


def _probe_amp(device: torch.device) -> tuple[bool, str | None]:
    """Return (supported, dtype_name). bfloat16 is preferred: it needs no loss scaling."""
    if device.type != "cuda":
        return False, None
    try:
        if torch.cuda.is_bf16_supported():
            return True, "bfloat16"
    except Exception:  # noqa: BLE001 - probing a capability; any failure means "no"
        pass
    return True, "float16"


def _resolve_directml() -> DeviceInfo | None:
    try:
        import torch_directml  # type: ignore[import-not-found]
    except ImportError:
        return None
    if not torch_directml.is_available():
        return None
    return DeviceInfo(
        backend="directml",
        device=torch_directml.device(),
        name=torch_directml.device_name(0),
        torch_version=torch.__version__,
        supports_amp=False,
        notes=[
            "DEGRADED PATH (ADR-009 rung 4). DirectML is in maintenance mode with "
            "op-coverage gaps and no AMP. Results must be labelled as DirectML-produced.",
        ],
    )


def get_device_info(prefer: Backend | None = None) -> DeviceInfo:
    """Resolve the best available backend.

    Args:
        prefer: force a specific backend instead of auto-detecting. Useful for reproducing
            a CPU-only analysis run on a machine that has a GPU.
    """
    if prefer == "cpu":
        return DeviceInfo(
            backend="cpu",
            device=torch.device("cpu"),
            name=platform.processor() or "cpu",
            torch_version=torch.__version__,
        )

    if prefer in (None, "rocm", "cuda") and torch.cuda.is_available():
        hip = getattr(torch.version, "hip", None)
        backend: Backend = "rocm" if hip else "cuda"
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(0)
        supports_amp, amp_dtype = _probe_amp(device)

        notes: list[str] = []
        override = os.environ.get(GFX_OVERRIDE_ENV)
        if backend == "rocm" and not override:
            notes.append(
                f"{GFX_OVERRIDE_ENV} is not set. On gfx1031 (RX 6800M) this must be "
                f"{GFX1030_OVERRIDE!r} or the GPU will not be usable. See ADR-009."
            )

        return DeviceInfo(
            backend=backend,
            device=device,
            name=props.name,
            torch_version=torch.__version__,
            hip_version=hip,
            cuda_version=torch.version.cuda,
            total_memory_gb=props.total_memory / (1024**3),
            gfx_override=override,
            supports_amp=supports_amp,
            amp_dtype=amp_dtype,
            notes=notes,
        )

    if prefer in (None, "directml"):
        dml = _resolve_directml()
        if dml is not None:
            return dml

    return DeviceInfo(
        backend="cpu",
        device=torch.device("cpu"),
        name=platform.processor() or "cpu",
        torch_version=torch.__version__,
        notes=[
            "No GPU backend found. Frozen-encoder CBM variants and the full XAI protocol run "
            "fine on CPU (ADR-008), but feature extraction, B0 and the joint CBM sweep will be "
            "impractically slow."
        ],
    )


def get_device(prefer: Backend | None = None) -> torch.device:
    """Convenience wrapper for when only the ``torch.device`` is needed."""
    return get_device_info(prefer).device


if __name__ == "__main__":
    print(get_device_info().summary())
