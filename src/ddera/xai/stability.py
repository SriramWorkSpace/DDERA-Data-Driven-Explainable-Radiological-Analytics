"""Explanation stability under clinically irrelevant perturbation.

An explanation that changes when the image is rotated two degrees is not an explanation.
If a small, diagnostically meaningless perturbation reorders which concepts the model claims
to be reasoning from, then the concept vector is tracking something other than the anatomy
it is named after -- and the ante-hoc claim fails regardless of how good the AUROC is.

Three complementary views, because they fail independently:

**Magnitude** (:func:`concept_drift`)
    How far did the concept values move? Absolute movement.

**Ordering** (:func:`rank_stability`)
    Did the *ranking* of concepts survive? A uniform shift in all concepts is far less
    damaging to an explanation than a reshuffle of which concept ranks first -- the
    explanation is read as an ordering.

**Decision** (:func:`prediction_flip_rate`)
    Did the final call change? The clinically consequential view.

A model can be stable on one and unstable on another. Reporting only the flattering one
would be exactly the kind of selective reporting CLAUDE.md section 8 forbids.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import spearmanr


def concept_drift(
    concepts_base: npt.ArrayLike,
    concepts_perturbed: npt.ArrayLike,
) -> dict[str, Any]:
    """Magnitude of concept-vector movement under perturbation.

    Returns mean/median/max L1 drift per sample, mean L-infinity drift, and per-concept mean
    absolute drift so an individual unstable concept can be identified rather than hidden in
    an average.
    """
    base = np.asarray(concepts_base, dtype=float)
    pert = np.asarray(concepts_perturbed, dtype=float)
    if base.shape != pert.shape:
        raise ValueError(f"Shape mismatch: {base.shape} vs {pert.shape}")

    abs_diff = np.abs(pert - base)
    l1_per_sample = abs_diff.sum(axis=1)
    linf_per_sample = abs_diff.max(axis=1)

    return {
        "l1_mean": float(l1_per_sample.mean()),
        "l1_median": float(np.median(l1_per_sample)),
        "l1_max": float(l1_per_sample.max()),
        "l1_normalised": float(l1_per_sample.mean() / base.shape[1]),
        "linf_mean": float(linf_per_sample.mean()),
        "linf_max": float(linf_per_sample.max()),
        "per_concept_mean_abs_drift": abs_diff.mean(axis=0).tolist(),
    }


def rank_stability(
    concepts_base: npt.ArrayLike,
    concepts_perturbed: npt.ArrayLike,
) -> dict[str, Any]:
    """Does the ordering of concepts survive the perturbation?

    Spearman rank correlation is computed **per sample** across concepts, then summarised.
    This is the right granularity: an explanation is read one patient at a time, so what
    matters is whether *this* patient's concept ordering is stable.

    Samples whose concept values are constant (no variance to rank) yield an undefined
    correlation and are excluded, with the count reported.
    """
    base = np.asarray(concepts_base, dtype=float)
    pert = np.asarray(concepts_perturbed, dtype=float)
    if base.shape != pert.shape:
        raise ValueError(f"Shape mismatch: {base.shape} vs {pert.shape}")
    if base.shape[1] < 2:
        raise ValueError("Rank stability needs at least 2 concepts.")

    correlations: list[float] = []
    for i in range(base.shape[0]):
        if np.ptp(base[i]) < 1e-12 or np.ptp(pert[i]) < 1e-12:
            continue
        rho = spearmanr(base[i], pert[i]).statistic
        if rho == rho:
            correlations.append(float(rho))

    if not correlations:
        return {
            "spearman_mean": float("nan"),
            "spearman_median": float("nan"),
            "spearman_min": float("nan"),
            "n_valid": 0,
            "n_excluded": int(base.shape[0]),
        }

    arr = np.asarray(correlations)
    return {
        "spearman_mean": float(arr.mean()),
        "spearman_median": float(np.median(arr)),
        "spearman_min": float(arr.min()),
        "spearman_p05": float(np.quantile(arr, 0.05)),
        "fraction_above_0.9": float(np.mean(arr > 0.9)),
        "n_valid": int(arr.size),
        "n_excluded": int(base.shape[0] - arr.size),
    }


def prediction_flip_rate(
    probs_base: npt.ArrayLike,
    probs_perturbed: npt.ArrayLike,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Did the decision change? The clinically consequential view."""
    base = np.asarray(probs_base, dtype=float)
    pert = np.asarray(probs_perturbed, dtype=float)
    if base.shape != pert.shape:
        raise ValueError(f"Shape mismatch: {base.shape} vs {pert.shape}")

    flips = (base >= threshold) != (pert >= threshold)
    delta = np.abs(pert - base)
    return {
        "flip_rate": float(flips.mean()),
        "n_flipped": int(flips.sum()),
        "mean_abs_delta_p": float(delta.mean()),
        "max_abs_delta_p": float(delta.max()),
        "p95_abs_delta_p": float(np.quantile(delta, 0.95)),
        "threshold": threshold,
    }


def stability_report(
    concepts_base: npt.ArrayLike,
    perturbed_runs: list[npt.ArrayLike],
    *,
    predict_fn=None,
    threshold: float = 0.5,
    perturbation_names: list[str] | None = None,
) -> dict[str, Any]:
    """The full Phase 5 stability family across several perturbation types.

    Args:
        concepts_base: ``(n, k)`` concepts from the unperturbed inputs.
        perturbed_runs: one ``(n, k)`` concept matrix per perturbation type (e.g. rotation,
            brightness, noise), row-aligned with ``concepts_base``.
        predict_fn: optional ``concepts -> probabilities``; enables the flip-rate view.
        threshold: decision threshold for flips.
        perturbation_names: labels for the runs.

    The aggregate reports the **worst** case across perturbations, not the mean. A model
    that is stable under three perturbations and falls apart under a fourth is not stable,
    and averaging would conceal exactly the failure worth knowing about.
    """
    base = np.asarray(concepts_base, dtype=float)
    names = perturbation_names or [f"perturbation_{i}" for i in range(len(perturbed_runs))]
    if len(names) != len(perturbed_runs):
        raise ValueError(f"Got {len(perturbed_runs)} runs but {len(names)} names.")

    per_perturbation: dict[str, dict[str, Any]] = {}
    for name, pert in zip(names, perturbed_runs, strict=True):
        pert_arr = np.asarray(pert, dtype=float)
        entry: dict[str, Any] = {
            "drift": concept_drift(base, pert_arr),
            "rank": rank_stability(base, pert_arr),
        }
        if predict_fn is not None:
            entry["prediction"] = prediction_flip_rate(
                np.asarray(predict_fn(base), dtype=float),
                np.asarray(predict_fn(pert_arr), dtype=float),
                threshold=threshold,
            )
        per_perturbation[name] = entry

    worst_l1 = max(e["drift"]["l1_mean"] for e in per_perturbation.values())
    ranks = [
        e["rank"]["spearman_mean"]
        for e in per_perturbation.values()
        if e["rank"]["spearman_mean"] == e["rank"]["spearman_mean"]
    ]
    aggregate: dict[str, Any] = {
        "worst_l1_drift": float(worst_l1),
        "worst_rank_stability": float(min(ranks)) if ranks else float("nan"),
    }
    if predict_fn is not None:
        aggregate["worst_flip_rate"] = float(
            max(e["prediction"]["flip_rate"] for e in per_perturbation.values())
        )

    aggregate["interpretation"] = _describe(
        aggregate["worst_rank_stability"], aggregate.get("worst_flip_rate")
    )
    return {"per_perturbation": per_perturbation, "aggregate": aggregate}


def _describe(rank_stability_value: float, flip_rate: float | None) -> str:
    if rank_stability_value != rank_stability_value:
        return "undefined (insufficient concept variance to rank)"
    if rank_stability_value > 0.95 and (flip_rate is None or flip_rate < 0.02):
        return "stable: concept ordering and decisions survive irrelevant perturbation"
    if rank_stability_value > 0.85:
        return "moderately stable: some concept reordering under perturbation"
    return (
        "unstable: concept ordering changes materially under clinically irrelevant "
        "perturbation; the interpretability claim is weakened and this must be reported"
    )
