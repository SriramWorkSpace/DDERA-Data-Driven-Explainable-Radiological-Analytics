"""Concept completeness.

How much of the task-relevant information does the bottleneck actually carry?

Most CBM work asserts completeness or gestures at it. DDERA measures it, with the
**hybrid-residual sweep**: widen a residual channel of size ``k`` that bypasses the concept
bottleneck, and watch what performance the bypass buys.

.. code-block:: text

    k = 0   concepts only            pure bottleneck, fully interpretable
    k > 0   concepts + k residuals   partially interpretable
    k -> d  approaches the black box  fully opaque

``AUROC(k) - AUROC(0)`` is then the performance the 12 concepts *fail* to deliver -- an
empirical quantity, not an assumption. The curve's shape is the interesting part: if it
saturates immediately, the concepts were nearly complete and the interpretability cost is
small; if it climbs steadily, the bottleneck is discarding a lot.

The normalised :func:`completeness_ratio` places the bottleneck on a scale from chance (0)
to the black-box reference (1), which is the number that belongs in the abstract.

Note on Invariants 3 and 4: the hybrid model exists **only** to quantify what the bottleneck
costs. It is never DDERA's model, and its residual width must always be reported next to its
metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

CHANCE_AUROC = 0.5


@dataclass(frozen=True)
class CompletenessCurve:
    """AUROC as a function of residual-channel width."""

    k: npt.NDArray[np.int_]
    auroc: npt.NDArray[np.float64]
    reference_auroc: float | None = None

    @property
    def bottleneck_auroc(self) -> float:
        """AUROC at ``k = 0`` -- the pure, fully interpretable bottleneck."""
        idx = int(np.argmin(self.k))
        if self.k[idx] != 0:
            raise ValueError("Curve does not contain k=0; cannot identify the pure bottleneck.")
        return float(self.auroc[idx])

    @property
    def saturated_auroc(self) -> float:
        """AUROC at the widest residual measured."""
        return float(self.auroc[int(np.argmax(self.k))])

    @property
    def interpretability_cost(self) -> float:
        """AUROC given up by using the pure bottleneck instead of the widest residual."""
        return self.saturated_auroc - self.bottleneck_auroc

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k.tolist(),
            "auroc": self.auroc.tolist(),
            "reference_auroc": self.reference_auroc,
            "bottleneck_auroc": self.bottleneck_auroc,
            "saturated_auroc": self.saturated_auroc,
            "interpretability_cost": self.interpretability_cost,
        }


def completeness_ratio(
    auroc_bottleneck: float,
    auroc_reference: float,
    chance: float = CHANCE_AUROC,
) -> float:
    """Fraction of the reference model's *skill* retained by the pure bottleneck.

    .. code-block:: text

        ratio = (AUROC_bottleneck - chance) / (AUROC_reference - chance)

    Skill is measured above chance, not from zero: an AUROC of 0.5 is worth nothing, so
    dividing raw AUROCs would flatter every model. ``1.0`` means the concepts lose nothing;
    ``0.0`` means the bottleneck is no better than guessing.

    Values above 1 are possible and legitimate -- the constrained model occasionally
    generalises better than the black box, because the bottleneck regularises.
    """
    denominator = auroc_reference - chance
    if abs(denominator) < 1e-12:
        return float("nan")
    return float((auroc_bottleneck - chance) / denominator)


def completeness_curve(
    auroc_by_k: dict[int, float],
    reference_auroc: float | None = None,
) -> CompletenessCurve:
    """Build a curve from a ``{residual_width: auroc}`` mapping (as produced by the sweep)."""
    if not auroc_by_k:
        raise ValueError("auroc_by_k is empty.")
    if 0 not in auroc_by_k:
        raise ValueError("The sweep must include k=0 (the pure bottleneck) to be interpretable.")

    ks = np.array(sorted(auroc_by_k), dtype=int)
    return CompletenessCurve(
        k=ks,
        auroc=np.array([auroc_by_k[int(k)] for k in ks], dtype=float),
        reference_auroc=reference_auroc,
    )


def completeness_report(
    auroc_by_k: dict[int, float],
    *,
    reference_auroc: float | None = None,
    chance: float = CHANCE_AUROC,
) -> dict[str, Any]:
    """The full Phase 5 completeness family, as written into ``metrics.json``."""
    curve = completeness_curve(auroc_by_k, reference_auroc)
    bottleneck = curve.bottleneck_auroc
    reference = reference_auroc if reference_auroc is not None else curve.saturated_auroc

    marginal = {int(k): float(auroc_by_k[int(k)] - bottleneck) for k in curve.k if int(k) != 0}
    saturation_k = _saturation_point(curve)

    return {
        "curve": curve.to_dict(),
        "bottleneck_auroc": bottleneck,
        "reference_auroc": reference,
        "completeness_ratio": completeness_ratio(bottleneck, reference, chance),
        "interpretability_cost": float(reference - bottleneck),
        "marginal_gain_by_k": marginal,
        "saturation_k": saturation_k,
        "interpretation": _describe(completeness_ratio(bottleneck, reference, chance)),
    }


def _saturation_point(curve: CompletenessCurve, tolerance: float = 0.005) -> int | None:
    """Smallest residual width reaching within ``tolerance`` of the best AUROC observed.

    A small saturation point is a strong result: it means a handful of extra dimensions
    recovers everything the bottleneck lost, so the concepts were nearly sufficient.
    """
    best = float(np.max(curve.auroc))
    for k, auroc in zip(curve.k, curve.auroc, strict=True):
        if best - auroc <= tolerance:
            return int(k)
    return None


def _describe(ratio: float) -> str:
    if ratio != ratio:
        return "undefined (reference model is at chance)"
    if ratio >= 0.99:
        return "concepts are effectively complete: no measurable interpretability cost"
    if ratio >= 0.95:
        return "concepts are nearly complete: small interpretability cost"
    if ratio >= 0.85:
        return "moderate interpretability cost; report the trade-off explicitly"
    return "substantial interpretability cost; the bottleneck discards real predictive signal"
