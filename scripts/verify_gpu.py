"""DDERA Phase 0 GPU verification gate.

Proves that the resolved backend can actually run this project's workload -- not merely that
a device is visible. `torch.cuda.is_available()` returning True on an unsupported gfx target
means very little; MIOpen convolutions and a sustained DenseNet training loop are where these
stacks actually fail.

Context (ADR-009): the development GPU is a Radeon RX 6800M (Navi 22, gfx1031), which is not
on AMD's supported list. gfx1030 IS officially packaged in current ROCm and built into the
PyTorch ROCm wheels, so gfx1031 loads those code objects via HSA_OVERRIDE_GFX_VERSION=10.3.0.
That is well-founded but unsupported, so it must be demonstrated rather than assumed.

    NO TRAINING PHASE BEGINS UNTIL THIS PASSES.

Usage:
    python scripts/verify_gpu.py              # checks 1-6 and 8 (a few minutes)
    python scripts/verify_gpu.py --quick      # checks 1-3 only (seconds; per-session sanity)
    python scripts/verify_gpu.py --full       # everything, including the 30-minute soak
    python scripts/verify_gpu.py --soak-minutes 10
    python scripts/verify_gpu.py --json out.json

Exit code is 0 only if every selected check passes. Paste the output into decisions.md
ADR-009 under "Verification result", including exact versions.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from ddera.device import GFX1030_OVERRIDE, GFX_OVERRIDE_ENV, get_device_info  # noqa: E402
from ddera.seed import set_seed  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


@dataclass
class CheckResult:
    number: int
    name: str
    status: str
    detail: str = ""
    seconds: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)


def _fmt(result: CheckResult) -> str:
    icon = {PASS: "[ OK ]", FAIL: "[FAIL]", SKIP: "[SKIP]"}[result.status]
    line = f"{icon} {result.number}. {result.name}  ({result.seconds:.1f}s)"
    if result.detail:
        line += "\n       " + result.detail.replace("\n", "\n       ")
    return line


def _run(number: int, name: str, fn: Callable[[], tuple[str, dict[str, Any]]]) -> CheckResult:
    start = time.time()
    try:
        detail, data = fn()
        return CheckResult(number, name, PASS, detail, time.time() - start, data)
    except Exception as exc:  # noqa: BLE001 - a failing check must never abort the suite
        detail = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, AssertionError):
            detail += "\n" + traceback.format_exc(limit=3)
        return CheckResult(number, name, FAIL, detail, time.time() - start)


# ---------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------


def check_1_device(info: Any) -> tuple[str, dict[str, Any]]:
    """Device visible, and the gfx override present when it is needed."""
    assert info.backend != "cpu", (
        "No GPU backend resolved. On Linux confirm `rocminfo` sees the card and that the "
        "PyTorch ROCm wheel (not the CPU wheel) is installed."
    )
    if info.backend == "rocm" and not info.gfx_override:
        raise AssertionError(
            f"ROCm is active but {GFX_OVERRIDE_ENV} is unset. On gfx1031 set it to "
            f"{GFX1030_OVERRIDE!r} before launching Python."
        )
    return info.summary(), info.to_dict()


def check_2_matmul(device: torch.device) -> tuple[str, dict[str, Any]]:
    """Large matmul must agree with a CPU reference."""
    set_seed(0)
    a = torch.randn(2048, 2048)
    b = torch.randn(2048, 2048)
    expected = a @ b
    actual = (a.to(device) @ b.to(device)).cpu()

    max_err = (actual - expected).abs().max().item()
    assert torch.allclose(actual, expected, atol=1e-2, rtol=1e-3), (
        f"GPU matmul disagrees with CPU (max abs error {max_err:.3e}). "
        "This indicates a miscompiled kernel for this gfx target."
    )
    return f"2048x2048 matmul matches CPU (max abs err {max_err:.2e})", {"max_abs_err": max_err}


def check_3_conv2d(device: torch.device) -> tuple[str, dict[str, Any]]:
    """conv2d forward AND backward -- the MIOpen path, where unsupported targets usually break."""
    set_seed(0)
    conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
    x = torch.randn(4, 3, 64, 64)

    cpu_out = conv(x)
    cpu_out.sum().backward()
    cpu_grad = conv.weight.grad.clone()

    conv.zero_grad(set_to_none=True)
    conv_gpu = conv.to(device)
    gpu_out = conv_gpu(x.to(device))
    gpu_out.sum().backward()
    gpu_grad = conv_gpu.weight.grad.cpu()

    fwd_err = (gpu_out.cpu() - cpu_out).abs().max().item()
    bwd_err = (gpu_grad - cpu_grad).abs().max().item()

    assert torch.allclose(gpu_out.cpu(), cpu_out, atol=1e-3, rtol=1e-3), (
        f"conv2d FORWARD mismatch (max abs err {fwd_err:.3e}). MIOpen is producing wrong results."
    )
    assert torch.allclose(gpu_grad, cpu_grad, atol=1e-2, rtol=1e-2), (
        f"conv2d BACKWARD mismatch (max abs err {bwd_err:.3e}). Training would silently diverge."
    )
    return (
        f"conv2d fwd/bwd match CPU (fwd {fwd_err:.2e}, bwd {bwd_err:.2e})",
        {"fwd_max_err": fwd_err, "bwd_max_err": bwd_err},
    )


def check_4_densenet_amp(device: torch.device, info: Any) -> tuple[str, dict[str, Any]]:
    """The actual workload: DenseNet-121 fwd+bwd at 224^2, batch 32, under autocast."""
    from torchvision.models import densenet121

    set_seed(0)
    model = densenet121(weights=None, num_classes=12).to(device)
    x = torch.randn(32, 3, 224, 224, device=device)
    y = torch.randint(0, 2, (32, 12), device=device).float()
    criterion = nn.BCEWithLogitsLoss()

    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(info.amp_dtype or "")
    use_amp = info.supports_amp and amp_dtype is not None

    if use_amp:
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            loss = criterion(model(x), y)
    else:
        loss = criterion(model(x), y)
    loss.backward()

    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "No gradients were produced."
    assert all(torch.isfinite(g).all() for g in grads), (
        "Non-finite gradients under AMP. Try bfloat16, or disable AMP and re-run."
    )

    return (
        f"DenseNet-121 fwd+bwd OK, batch 32 @ 224px, "
        f"amp={info.amp_dtype if use_amp else 'off'}, loss={loss.item():.4f}",
        {"loss": loss.item(), "amp": info.amp_dtype if use_amp else None},
    )


def check_5_bce(device: torch.device) -> tuple[str, dict[str, Any]]:
    """Masked BCEWithLogitsLoss backward -- DDERA's concept loss (ADR-004 U-Mask)."""
    set_seed(0)
    logits = torch.randn(64, 12, requires_grad=True)
    targets = torch.randint(0, 2, (64, 12)).float()
    mask = torch.randint(0, 2, (64, 12)).float()  # stands in for the -1 uncertain entries

    def masked_bce(lg: torch.Tensor, tg: torch.Tensor, mk: torch.Tensor) -> torch.Tensor:
        per_elem = nn.functional.binary_cross_entropy_with_logits(lg, tg, reduction="none")
        return (per_elem * mk).sum() / mk.sum().clamp(min=1.0)

    cpu_loss = masked_bce(logits, targets, mask)
    cpu_loss.backward()
    cpu_grad = logits.grad.clone()

    g_logits = logits.detach().to(device).requires_grad_(True)
    gpu_loss = masked_bce(g_logits, targets.to(device), mask.to(device))
    gpu_loss.backward()
    gpu_grad = g_logits.grad.cpu()

    err = (gpu_grad - cpu_grad).abs().max().item()
    assert torch.isfinite(gpu_loss), "Masked BCE loss is not finite on GPU."
    assert torch.allclose(gpu_grad, cpu_grad, atol=1e-5, rtol=1e-4), (
        f"Masked BCE gradient mismatch (max abs err {err:.3e})."
    )
    return f"Masked BCE fwd/bwd matches CPU (max abs err {err:.2e})", {"max_grad_err": err}


def check_6_overfit(device: torch.device) -> tuple[str, dict[str, Any]]:
    """200 steps on 8 fixed images must drive the loss to ~0.

    This is the real proof that learning works end to end. A stack can pass every numerical
    check above and still fail to optimise if the optimiser or an in-place op is broken.
    """
    from torchvision.models import densenet121

    set_seed(0)
    model = densenet121(weights=None, num_classes=12).to(device)
    x = torch.randn(8, 3, 224, 224, device=device)
    y = torch.randint(0, 2, (8, 12), device=device).float()

    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    first = last = float("nan")
    for step in range(200):
        optimiser.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimiser.step()
        if step == 0:
            first = loss.item()
        last = loss.item()

    assert torch.isfinite(torch.tensor(last)), "Loss became non-finite during optimisation."
    assert last < 0.05, (
        f"Failed to overfit 8 images: loss {first:.4f} -> {last:.4f} after 200 steps "
        "(expected < 0.05). The backend cannot train reliably."
    )
    return f"Overfit 8 images: loss {first:.4f} -> {last:.4f}", {"first": first, "final": last}


def check_7_soak(device: torch.device, minutes: float) -> tuple[str, dict[str, Any]]:
    """Sustained load. Targets the reported gfx1031 SIGSEGV / memory-growth failure mode."""
    from torchvision.models import densenet121

    set_seed(0)
    model = densenet121(weights=None, num_classes=12).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    deadline = time.time() + minutes * 60
    steps = 0
    start_mem = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0

    model.train()
    while time.time() < deadline:
        x = torch.randn(16, 3, 224, 224, device=device)
        y = torch.randint(0, 2, (16, 12), device=device).float()
        optimiser.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimiser.step()
        steps += 1
        if not torch.isfinite(loss):
            raise AssertionError(f"Loss became non-finite after {steps} steps.")

    end_mem = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    growth_mb = (end_mem - start_mem) / 1024**2
    assert growth_mb < 512, (
        f"Allocated memory grew {growth_mb:.0f} MB over {steps} steps -- likely a leak."
    )
    return (
        f"Survived {minutes:.0f} min / {steps} steps, memory growth {growth_mb:+.0f} MB",
        {"steps": steps, "minutes": minutes, "growth_mb": growth_mb},
    )


def check_8_vram(device: torch.device, info: Any) -> tuple[str, dict[str, Any]]:
    """Find the largest workable batch size at 224 and 320 px. Feeds the training configs."""
    from torchvision.models import densenet121

    if device.type != "cuda":
        raise AssertionError("VRAM probing requires a CUDA/ROCm device.")

    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(info.amp_dtype or "")
    use_amp = info.supports_amp and amp_dtype is not None
    results: dict[str, int] = {}

    for resolution in (224, 320):
        model = densenet121(weights=None, num_classes=12).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        best = 0

        for batch in (8, 16, 32, 48, 64, 96, 128):
            try:
                torch.cuda.empty_cache()
                x = torch.randn(batch, 3, resolution, resolution, device=device)
                y = torch.randint(0, 2, (batch, 12), device=device).float()
                optimiser.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        loss = criterion(model(x), y)
                else:
                    loss = criterion(model(x), y)
                loss.backward()
                optimiser.step()
                best = batch
                del x, y, loss
            except torch.cuda.OutOfMemoryError:
                break
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                break

        results[f"max_batch_{resolution}px"] = best
        del model, optimiser
        torch.cuda.empty_cache()

    assert results["max_batch_224px"] >= 8, (
        "Cannot fit even batch 8 at 224px. Check for another process holding VRAM."
    )
    detail = " | ".join(f"{k.replace('max_batch_', '')}: batch {v}" for k, v in results.items())
    return f"Max batch size -- {detail}", results


# ---------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="DDERA Phase 0 GPU verification gate")
    parser.add_argument("--quick", action="store_true", help="checks 1-3 only")
    parser.add_argument("--full", action="store_true", help="include the soak test")
    parser.add_argument("--soak-minutes", type=float, default=30.0)
    parser.add_argument("--json", type=Path, help="write results to this JSON file")
    args = parser.parse_args()

    print("=" * 78)
    print("DDERA GPU VERIFICATION GATE".center(78))
    print("=" * 78)
    print(f"Host   : {platform.node()}  ({platform.system()} {platform.release()})")
    print(f"Python : {sys.version.split()[0]}")
    print("-" * 78)

    info = get_device_info()
    results: list[CheckResult] = [_run(1, "Device detection", lambda: check_1_device(info))]
    print(_fmt(results[-1]))

    if results[0].status == FAIL:
        print("\n" + "=" * 78)
        print("GATE FAILED at check 1 -- no usable backend. Later checks cannot run.")
        print("Work the ADR-009 ladder; record the failure in decisions.md before stepping down.")
        print("=" * 78)
        return 1

    device = info.device
    planned: list[tuple[int, str, Callable[[], tuple[str, dict[str, Any]]]]] = [
        (2, "Matmul correctness", lambda: check_2_matmul(device)),
        (3, "conv2d fwd/bwd (MIOpen)", lambda: check_3_conv2d(device)),
    ]
    if not args.quick:
        planned += [
            (4, "DenseNet-121 fwd/bwd + AMP", lambda: check_4_densenet_amp(device, info)),
            (5, "Masked BCE backward", lambda: check_5_bce(device)),
            (6, "Overfit 8 images (200 steps)", lambda: check_6_overfit(device)),
        ]
        if args.full:
            planned.append(
                (
                    7,
                    f"Soak test ({args.soak_minutes:.0f} min)",
                    lambda: check_7_soak(device, args.soak_minutes),
                )
            )
        planned.append((8, "VRAM headroom probe", lambda: check_8_vram(device, info)))

    for number, name, fn in planned:
        result = _run(number, name, fn)
        results.append(result)
        print(_fmt(result))

    if not args.full and not args.quick:
        skipped = CheckResult(
            7,
            f"Soak test ({args.soak_minutes:.0f} min)",
            SKIP,
            "Not run. Use --full before committing to this stack.",
        )
        results.append(skipped)
        print(_fmt(skipped))

    failed = [r for r in results if r.status == FAIL]
    skipped = [r for r in results if r.status == SKIP]

    print("-" * 78)
    print(
        f"{len(results) - len(failed) - len(skipped)} passed, "
        f"{len(failed)} failed, {len(skipped)} skipped"
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {"device": info.to_dict(), "checks": [asdict(r) for r in results]},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"Results written to {args.json}")

    print("=" * 78)
    if failed:
        print("GATE FAILED. Do not begin any training phase.")
        print("Record the failure and exact versions in decisions.md (ADR-009), then step to")
        print("the next rung of the ladder. Do not install an old ROCm version without a")
        print("recorded failure justifying it.")
        print("=" * 78)
        return 1

    if skipped:
        print("Checks passed, but the soak test was skipped. Run --full before committing.")
    else:
        print("GATE PASSED. Paste this output into decisions.md ADR-009 with exact versions.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
