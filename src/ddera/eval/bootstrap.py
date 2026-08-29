"""Bootstrap confidence intervals.

Mandatory on every headline metric. Pneumonia is sparse in CheXpert (ADR-003), so a point
estimate of AUROC on a few hundred positives is close to meaningless on its own -- and the
research question is a *comparison* between a CBM and a black box. Without intervals we
cannot say whether an observed gap is real.

Two details that matter:

**Stratified resampling.** Positives and negatives are resampled separately so every replicate
preserves the observed prevalence. Unstratified resampling of a rare class occasionally
produces a replicate with zero positives, where AUROC is undefined; those replicates would
be silently dropped and bias the interval.

**Paired resampling.** :func:`paired_bootstrap_diff` resamples the *same* indices for both
models, so the interval is on the difference rather than on two independent estimates. Two
overlapping marginal intervals do not imply a non-significant difference; the paired
interval is the correct test for "is the CBM worse than the black box?".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

MetricFn = Callable[[npt.NDArray, npt.NDArray], float]


@dataclass(frozen=True)
class BootstrapResult:
    """A point estimate with a percentile confidence interval."""

    point: float
    lower: float
    upper: float
    std: float
    n_resamples: int
    n_valid: int
    confidence: float

    @property
    def ci_width(self) -> float:
        """Interval width -- feeds the ADR-003 escalation rule."""
        return self.upper - self.lower

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "ci_width": self.ci_width}

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.lower:.4f}, {self.upper:.4f}]"


def bootstrap_ci(
    y_true: npt.ArrayLike,
    y_score: npt.ArrayLike,
    metric_fn: MetricFn,
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    stratified: bool = True,
    seed: int = 42,
) -> BootstrapResult:
    """Percentile bootstrap CI for a metric.

    Args:
        y_true: binary ground truth.
        y_score: predicted scores.
        metric_fn: callable ``(y_true, y_score) -> float``. May return ``nan`` for a
            degenerate replicate; those are excluded and counted in ``n_valid``.
        n_resamples: number of bootstrap replicates.
        confidence: e.g. ``0.95`` for a 95% interval.
        stratified: resample classes separately, preserving prevalence.
        seed: RNG seed.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.shape != y_score.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_score.shape}")
    if y_true.size == 0:
        raise ValueError("Cannot bootstrap an empty array.")

    rng = np.random.default_rng(seed)
    point = float(metric_fn(y_true, y_score))

    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    n = y_true.size

    values: list[float] = []
    for _ in range(n_resamples):
        if stratified and pos_idx.size and neg_idx.size:
            idx = np.concatenate(
                [
                    rng.choice(pos_idx, size=pos_idx.size, replace=True),
                    rng.choice(neg_idx, size=neg_idx.size, replace=True),
                ]
            )
        else:
            idx = rng.integers(0, n, size=n)
        value = metric_fn(y_true[idx], y_score[idx])
        if value == value:  # exclude nan
            values.append(float(value))

    if not values:
        return BootstrapResult(
            point, float("nan"), float("nan"), float("nan"), n_resamples, 0, confidence
        )

    arr = np.asarray(values)
    alpha = (1.0 - confidence) / 2.0
    return BootstrapResult(
        point=point,
        lower=float(np.quantile(arr, alpha)),
        upper=float(np.quantile(arr, 1 - alpha)),
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        n_resamples=n_resamples,
        n_valid=arr.size,
        confidence=confidence,
    )


def paired_bootstrap_diff(
    y_true: npt.ArrayLike,
    y_score_a: npt.ArrayLike,
    y_score_b: npt.ArrayLike,
    metric_fn: MetricFn,
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    stratified: bool = True,
    seed: int = 42,
) -> BootstrapResult:
    """CI on ``metric(A) - metric(B)`` using shared resample indices.

    This is the correct instrument for DDERA's headline comparison. If the interval on
    ``AUROC(black box) - AUROC(CBM)`` excludes zero, the interpretability cost is real; if it
    straddles zero, we say the difference is not resolvable at this sample size rather than
    reporting a spurious gap.
    """
    y_true = np.asarray(y_true)
    a = np.asarray(y_score_a)
    b = np.asarray(y_score_b)
    if not (y_true.shape == a.shape == b.shape):
        raise ValueError(f"Shape mismatch: {y_true.shape}, {a.shape}, {b.shape}")

    rng = np.random.default_rng(seed)
    point = float(metric_fn(y_true, a) - metric_fn(y_true, b))

    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    n = y_true.size

    diffs: list[float] = []
    for _ in range(n_resamples):
        if stratified and pos_idx.size and neg_idx.size:
            idx = np.concatenate(
                [
                    rng.choice(pos_idx, size=pos_idx.size, replace=True),
                    rng.choice(neg_idx, size=neg_idx.size, replace=True),
                ]
            )
        else:
            idx = rng.integers(0, n, size=n)
        va = metric_fn(y_true[idx], a[idx])
        vb = metric_fn(y_true[idx], b[idx])
        if va == va and vb == vb:
            diffs.append(float(va - vb))

    if not diffs:
        return BootstrapResult(
            point, float("nan"), float("nan"), float("nan"), n_resamples, 0, confidence
        )

    arr = np.asarray(diffs)
    alpha = (1.0 - confidence) / 2.0
    return BootstrapResult(
        point=point,
        lower=float(np.quantile(arr, alpha)),
        upper=float(np.quantile(arr, 1 - alpha)),
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        n_resamples=n_resamples,
        n_valid=arr.size,
        confidence=confidence,
    )


def is_significant(result: BootstrapResult) -> bool:
    """True when a difference interval excludes zero."""
    if result.lower != result.lower or result.upper != result.upper:
        return False
    return not (result.lower <= 0.0 <= result.upper)
