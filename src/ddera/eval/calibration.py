"""Calibration.

An interpretable model that says "84%" should be right about 84% of the time. If it is not,
the concept contributions are being read off a scale that does not mean what it appears to
mean, and the explanation is misleading even when the ranking is correct. Calibration is
therefore part of the interpretability claim here, not a separate nicety.

Provides Expected Calibration Error (equal-width and equal-mass binning), the Brier score,
reliability curves, and temperature scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar

BinStrategy = Literal["uniform", "quantile"]
_EPS = 1e-12


@dataclass(frozen=True)
class ReliabilityCurve:
    """Binned reliability data, ready to plot."""

    bin_lower: npt.NDArray[np.float64]
    bin_upper: npt.NDArray[np.float64]
    bin_confidence: npt.NDArray[np.float64]  # mean predicted probability in bin
    bin_accuracy: npt.NDArray[np.float64]  # observed positive rate in bin
    bin_count: npt.NDArray[np.int_]

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "bin_lower": self.bin_lower.tolist(),
            "bin_upper": self.bin_upper.tolist(),
            "bin_confidence": self.bin_confidence.tolist(),
            "bin_accuracy": self.bin_accuracy.tolist(),
            "bin_count": self.bin_count.tolist(),
        }


def _bin_edges(probs: npt.NDArray[np.float64], n_bins: int, strategy: BinStrategy):
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    # Quantile binning puts equal sample counts per bin, which stops a handful of points in
    # a sparse high-confidence bin from dominating the ECE.
    edges = np.quantile(probs, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    return np.unique(edges)


def reliability_curve(
    y_true: npt.ArrayLike,
    y_prob: npt.ArrayLike,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
) -> ReliabilityCurve:
    """Bin predictions and compare mean confidence against observed accuracy."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_prob.shape}")

    edges = _bin_edges(y_prob, n_bins, strategy)
    lowers, uppers, confs, accs, counts = [], [], [], [], []

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        # Right-closed bins, with the first bin also closed on the left so p == 0 is counted.
        in_bin = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= lo) & (y_prob <= hi)
        lowers.append(lo)
        uppers.append(hi)
        counts.append(int(in_bin.sum()))
        confs.append(float(y_prob[in_bin].mean()) if in_bin.any() else float("nan"))
        accs.append(float(y_true[in_bin].mean()) if in_bin.any() else float("nan"))

    return ReliabilityCurve(
        bin_lower=np.array(lowers),
        bin_upper=np.array(uppers),
        bin_confidence=np.array(confs),
        bin_accuracy=np.array(accs),
        bin_count=np.array(counts),
    )


def expected_calibration_error(
    y_true: npt.ArrayLike,
    y_prob: npt.ArrayLike,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
) -> float:
    """ECE: count-weighted mean of ``|accuracy - confidence|`` across bins.

    ``0`` is perfect calibration. Empty bins contribute nothing (they carry zero weight).
    """
    curve = reliability_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    total = curve.bin_count.sum()
    if total == 0:
        return float("nan")
    occupied = curve.bin_count > 0
    gaps = np.abs(curve.bin_accuracy[occupied] - curve.bin_confidence[occupied])
    weights = curve.bin_count[occupied] / total
    return float(np.sum(weights * gaps))


def maximum_calibration_error(
    y_true: npt.ArrayLike,
    y_prob: npt.ArrayLike,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
) -> float:
    """Worst-case bin gap. Surfaces a badly miscalibrated region that ECE would average away."""
    curve = reliability_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    occupied = curve.bin_count > 0
    if not occupied.any():
        return float("nan")
    return float(np.max(np.abs(curve.bin_accuracy[occupied] - curve.bin_confidence[occupied])))


def brier_score(y_true: npt.ArrayLike, y_prob: npt.ArrayLike) -> float:
    """Mean squared error of the probability forecast. Lower is better."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    return float(np.mean((y_prob - y_true) ** 2))


def fit_temperature(
    logits: npt.ArrayLike,
    y_true: npt.ArrayLike,
    bounds: tuple[float, float] = (0.05, 20.0),
) -> float:
    """Fit a single temperature ``T`` minimising NLL of ``sigmoid(logit / T)``.

    Temperature scaling is the right post-processing step for DDERA because it is
    **monotone**: it rescales the logit without touching the relative ordering of concept
    contributions. Sharpening the probabilities therefore does not alter which concepts the
    explanation attributes the decision to.

    Fit this on the validation split only; applying a temperature fitted on test data would
    leak.
    """
    logits = np.asarray(logits, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if logits.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: {logits.shape} vs {y_true.shape}")
    if len(np.unique(y_true)) < 2:
        return 1.0

    def nll(temperature: float) -> float:
        p = np.clip(_sigmoid(logits / temperature), _EPS, 1 - _EPS)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

    result = minimize_scalar(nll, bounds=bounds, method="bounded")
    return float(result.x)


def apply_temperature(logits: npt.ArrayLike, temperature: float) -> npt.NDArray[np.float64]:
    """Apply a fitted temperature and return probabilities."""
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    return _sigmoid(np.asarray(logits, dtype=float) / temperature)


def calibration_report(
    y_true: npt.ArrayLike,
    y_prob: npt.ArrayLike,
    n_bins: int = 10,
) -> dict[str, Any]:
    """The full calibration family, as written into ``metrics.json``."""
    return {
        "ece": expected_calibration_error(y_true, y_prob, n_bins, "uniform"),
        "ece_quantile": expected_calibration_error(y_true, y_prob, n_bins, "quantile"),
        "mce": maximum_calibration_error(y_true, y_prob, n_bins),
        "brier": brier_score(y_true, y_prob),
        "mean_confidence": float(np.mean(np.asarray(y_prob, dtype=float))),
        "observed_rate": float(np.mean(np.asarray(y_true, dtype=float))),
        "n_bins": n_bins,
        "reliability": reliability_curve(y_true, y_prob, n_bins).to_dict(),
    }


def _sigmoid(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
