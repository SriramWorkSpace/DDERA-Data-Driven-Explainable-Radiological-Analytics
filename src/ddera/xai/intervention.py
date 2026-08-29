"""Concept contributions and intervention analysis.

This module is the mechanical core of DDERA's ante-hoc claim.

**Why the explanation is exact.** With a linear reasoner the logit decomposes as

.. code-block:: text

    logit(p) = b + sum_j w_j * c_j

so ``w_j * c_j`` is concept *j*'s signed contribution -- an arithmetic identity, not an
attribution estimate. There is no sampling, no surrogate model, and no approximation error
to argue about. :func:`verify_decomposition` asserts this holds to floating-point precision,
and the dashboard's waterfall renders exactly these terms.

**Why intervention is the real test.** Displaying an explanation proves nothing; a model can
show plausible numbers it does not actually use. Intervention asks the harder question:
*if a concept changes, does the prediction change appropriately?* For a linear reasoner the
answer is predictable in closed form -- setting ``c_j := v`` must shift the logit by exactly
``w_j * (v - c_j)`` -- and :func:`intervention_effect` checks the model against that
prediction.

**Domain-agnostic by construction.** Every public function takes a ``predict_fn``
(``concepts -> probabilities``) rather than a model object, so the harness works for any
reasoner, linear or not, in any domain. Nothing here knows what a radiograph is. That is
what makes Invariant 9 testable in Phase 8 rather than merely asserted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from ddera.eval.metrics import safe_auroc

PredictFn = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
Ordering = Literal["random", "uncertainty", "weight", "oracle"]

_EPS = 1e-12


def sigmoid(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Numerically stable logistic function."""
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -500, 500)))


def logit(p: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Inverse of :func:`sigmoid`, clipped away from 0 and 1 to stay finite."""
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


# ---------------------------------------------------------------------------------------
# The interpretable reasoner
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearReasoner:
    """``p = sigmoid(w . c + b)`` -- DDERA's primary interpretable reasoner.

    Deliberately transparent: ``weights`` *is* the explanation. A concept's weight says how
    much, and in which direction, that concept moves the decision, globally and for every
    patient.
    """

    weights: npt.NDArray[np.float64]
    bias: float
    concept_names: list[str] | None = None

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=float)
        object.__setattr__(self, "weights", w)
        if w.ndim != 1:
            raise ValueError(f"weights must be 1-D, got shape {w.shape}")
        if self.concept_names is not None and len(self.concept_names) != w.size:
            raise ValueError(f"Got {w.size} weights but {len(self.concept_names)} concept names.")

    @property
    def n_concepts(self) -> int:
        return int(self.weights.size)

    def predict_logit(self, concepts: npt.ArrayLike) -> npt.NDArray[np.float64]:
        c = _as_concept_matrix(concepts, self.n_concepts)
        return c @ self.weights + self.bias

    def predict_proba(self, concepts: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return sigmoid(self.predict_logit(concepts))

    def contributions(self, concepts: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """``(n, k)`` matrix of signed contributions ``w_j * c_j``.

        These terms plus ``bias`` sum exactly to the logit. This is what the dashboard
        waterfall shows.
        """
        c = _as_concept_matrix(concepts, self.n_concepts)
        return c * self.weights[None, :]

    def as_predict_fn(self) -> PredictFn:
        """Adapt to the ``predict_fn`` interface the rest of this module consumes."""
        return self.predict_proba


def verify_decomposition(
    reasoner: LinearReasoner,
    concepts: npt.ArrayLike,
    atol: float = 1e-9,
) -> float:
    """Assert that contributions plus bias reproduce the logit exactly.

    Returns the maximum absolute error. Raises if it exceeds ``atol``.

    This guards the dashboard's central honesty claim: if the displayed contributions did
    not sum to the decision, the explanation would be decorative.
    """
    c = _as_concept_matrix(concepts, reasoner.n_concepts)
    reconstructed = reasoner.contributions(c).sum(axis=1) + reasoner.bias
    error = float(np.max(np.abs(reconstructed - reasoner.predict_logit(c))))
    if error > atol:
        raise AssertionError(
            f"Contribution decomposition is broken: max abs error {error:.3e} > {atol:.1e}. "
            "The explanation would not match the prediction."
        )
    return error


# ---------------------------------------------------------------------------------------
# Single interventions
# ---------------------------------------------------------------------------------------


def intervene(
    concepts: npt.ArrayLike,
    index: int,
    value: float | npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Return a copy of ``concepts`` with column ``index`` set to ``value``.

    Never mutates the input: an in-place intervention would silently corrupt the baseline
    that every downstream comparison is measured against.
    """
    c = np.array(concepts, dtype=float, copy=True)
    if c.ndim == 1:
        c = c[None, :]
    if not 0 <= index < c.shape[1]:
        raise IndexError(f"Concept index {index} out of range for {c.shape[1]} concepts.")
    c[:, index] = value
    return c


def intervention_effect(
    predict_fn: PredictFn,
    concepts: npt.ArrayLike,
    index: int,
    new_value: float,
) -> dict[str, Any]:
    """Measure what one intervention actually does to the prediction.

    Returns the before/after probabilities and the realised logit shift. For a linear
    reasoner the shift must equal ``w_j * (new_value - old_value)``; comparing the realised
    shift against that expectation is exactly what :func:`faithfulness_report` automates.
    """
    c = np.array(concepts, dtype=float, copy=True)
    if c.ndim == 1:
        c = c[None, :]

    old_values = c[:, index].copy()
    p_before = np.asarray(predict_fn(c), dtype=float)
    p_after = np.asarray(predict_fn(intervene(c, index, new_value)), dtype=float)

    return {
        "concept_index": index,
        "old_value": old_values,
        "new_value": float(new_value),
        "p_before": p_before,
        "p_after": p_after,
        "delta_p": p_after - p_before,
        "delta_logit": logit(p_after) - logit(p_before),
        "mean_delta_p": float(np.mean(p_after - p_before)),
    }


def expected_logit_shift(
    weights: npt.ArrayLike,
    index: int,
    old_value: npt.ArrayLike,
    new_value: float,
) -> npt.NDArray[np.float64]:
    """Closed-form logit shift for a linear reasoner: ``w_j * (new - old)``."""
    w = np.asarray(weights, dtype=float)
    return w[index] * (float(new_value) - np.asarray(old_value, dtype=float))


# ---------------------------------------------------------------------------------------
# Intervention orderings
# ---------------------------------------------------------------------------------------


def intervention_order(
    concepts_pred: npt.ArrayLike,
    strategy: Ordering = "random",
    *,
    weights: npt.ArrayLike | None = None,
    concepts_true: npt.ArrayLike | None = None,
    seed: int = 0,
) -> npt.NDArray[np.int_]:
    """Build an ``(n, k)`` matrix giving, per sample, the order to intervene on concepts.

    Strategies:

    ``random``
        A shuffled order per sample. The neutral reference curve.
    ``uncertainty``
        Most uncertain first (``|c - 0.5|`` ascending). This is the *deployable* policy: it
        needs no ground truth, so it answers "which concept should a radiologist verify
        first?". Phase 6 builds on it directly.
    ``weight``
        Largest ``|w_j|`` first. Tests whether the concepts the model says matter most are
        the ones worth correcting.
    ``oracle``
        Largest concept error ``|c_pred - c_true|`` first. Requires ground truth, so it is
        an upper bound rather than a usable policy -- it bounds what any ordering could
        achieve.
    """
    c = np.asarray(concepts_pred, dtype=float)
    n, k = c.shape

    if strategy == "random":
        rng = np.random.default_rng(seed)
        return np.argsort(rng.random((n, k)), axis=1)

    if strategy == "uncertainty":
        return np.argsort(np.abs(c - 0.5), axis=1)

    if strategy == "weight":
        if weights is None:
            raise ValueError("strategy='weight' requires `weights`.")
        global_order = np.argsort(-np.abs(np.asarray(weights, dtype=float)))
        return np.tile(global_order, (n, 1))

    if strategy == "oracle":
        if concepts_true is None:
            raise ValueError("strategy='oracle' requires `concepts_true`.")
        error = np.abs(c - np.asarray(concepts_true, dtype=float))
        return np.argsort(-error, axis=1)

    raise ValueError(f"Unknown ordering strategy {strategy!r}")


def apply_intervention_order(
    concepts_pred: npt.ArrayLike,
    concepts_true: npt.ArrayLike,
    order: npt.NDArray[np.int_],
    n_intervened: int,
) -> npt.NDArray[np.float64]:
    """Replace the first ``n_intervened`` concepts (per sample, per ``order``) with truth."""
    c_pred = np.asarray(concepts_pred, dtype=float)
    c_true = np.asarray(concepts_true, dtype=float)
    if c_pred.shape != c_true.shape:
        raise ValueError(f"Shape mismatch: {c_pred.shape} vs {c_true.shape}")

    n, k = c_pred.shape
    if not 0 <= n_intervened <= k:
        raise ValueError(f"n_intervened must be in [0, {k}], got {n_intervened}")

    out = c_pred.copy()
    if n_intervened == 0:
        return out
    rows = np.repeat(np.arange(n), n_intervened)
    cols = order[:, :n_intervened].ravel()
    out[rows, cols] = c_true[rows, cols]
    return out


# ---------------------------------------------------------------------------------------
# Test-time intervention (TTI) curves
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TTICurve:
    """AUROC as a function of how many concepts were corrected to ground truth."""

    n_intervened: npt.NDArray[np.int_]
    auroc: npt.NDArray[np.float64]
    strategy: str
    baseline_auroc: float
    full_intervention_auroc: float

    @property
    def total_gain(self) -> float:
        """AUROC recovered by correcting every concept.

        Near zero means the reasoner barely uses its concepts -- an intervention-inert model,
        which is a serious negative finding regardless of how good its explanations look.
        """
        return self.full_intervention_auroc - self.baseline_auroc

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "n_intervened": self.n_intervened.tolist(),
            "auroc": self.auroc.tolist(),
            "baseline_auroc": self.baseline_auroc,
            "full_intervention_auroc": self.full_intervention_auroc,
            "total_gain": self.total_gain,
        }


def tti_curve(
    predict_fn: PredictFn,
    concepts_pred: npt.ArrayLike,
    concepts_true: npt.ArrayLike,
    y_true: npt.ArrayLike,
    *,
    strategy: Ordering = "random",
    weights: npt.ArrayLike | None = None,
    n_repeats: int = 1,
    seed: int = 0,
) -> TTICurve:
    """Test-time intervention curve.

    Progressively replaces predicted concepts with ground truth and records target AUROC at
    each step. A model that genuinely reasons through its bottleneck improves monotonically;
    a model that has routed information around the bottleneck barely moves.

    ``n_repeats`` averages over random orderings (only meaningful for ``strategy='random'``,
    where a single draw is noisy).
    """
    c_pred = np.asarray(concepts_pred, dtype=float)
    c_true = np.asarray(concepts_true, dtype=float)
    y = np.asarray(y_true)
    k = c_pred.shape[1]
    repeats = n_repeats if strategy == "random" else 1

    accumulated = np.zeros(k + 1)
    for r in range(repeats):
        order = intervention_order(
            c_pred, strategy, weights=weights, concepts_true=c_true, seed=seed + r
        )
        for m in range(k + 1):
            c_m = apply_intervention_order(c_pred, c_true, order, m)
            accumulated[m] += safe_auroc(y, np.asarray(predict_fn(c_m), dtype=float))

    aurocs = accumulated / repeats
    return TTICurve(
        n_intervened=np.arange(k + 1),
        auroc=aurocs,
        strategy=strategy,
        baseline_auroc=float(aurocs[0]),
        full_intervention_auroc=float(aurocs[-1]),
    )


def tti_all_strategies(
    predict_fn: PredictFn,
    concepts_pred: npt.ArrayLike,
    concepts_true: npt.ArrayLike,
    y_true: npt.ArrayLike,
    weights: npt.ArrayLike,
    *,
    n_repeats: int = 5,
    seed: int = 0,
) -> dict[str, TTICurve]:
    """Run all four orderings -- the standard Phase 5 intervention panel."""
    return {
        s: tti_curve(
            predict_fn,
            concepts_pred,
            concepts_true,
            y_true,
            strategy=s,
            weights=weights,
            n_repeats=n_repeats,
            seed=seed,
        )
        for s in ("random", "uncertainty", "weight", "oracle")
    }


# ---------------------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------------------


def empirical_sensitivity(
    predict_fn: PredictFn,
    concepts: npt.ArrayLike,
    *,
    eps: float = 1e-3,
) -> npt.NDArray[np.float64]:
    """Estimate ``d logit(p) / d c_j`` per concept by central finite differences.

    Evaluation points are clipped into ``[0, 1]`` (concepts are probabilities, so the model
    is only meaningfully defined there) and the denominator uses the *realised* step, which
    keeps the estimate correct at the boundaries instead of silently halving it.
    """
    c = np.asarray(concepts, dtype=float)
    n, k = c.shape
    sensitivity = np.zeros((n, k))

    for j in range(k):
        c_plus, c_minus = c.copy(), c.copy()
        c_plus[:, j] = np.clip(c[:, j] + eps, 0.0, 1.0)
        c_minus[:, j] = np.clip(c[:, j] - eps, 0.0, 1.0)
        step = c_plus[:, j] - c_minus[:, j]

        lo = logit(np.asarray(predict_fn(c_minus), dtype=float))
        hi = logit(np.asarray(predict_fn(c_plus), dtype=float))
        with np.errstate(divide="ignore", invalid="ignore"):
            sensitivity[:, j] = np.where(step > 0, (hi - lo) / np.where(step > 0, step, 1.0), 0.0)

    return sensitivity


def faithfulness_report(
    predict_fn: PredictFn,
    concepts: npt.ArrayLike,
    weights: npt.ArrayLike,
    *,
    eps: float = 1e-3,
    concept_names: list[str] | None = None,
) -> dict[str, Any]:
    """Does the model behave the way its weights claim?

    Compares the measured sensitivity ``d logit / d c_j`` against the declared weight
    ``w_j``. For a linear reasoner these must agree to numerical precision; for any other
    reasoner the gap quantifies how much its stated explanation misdescribes its behaviour.

    Reports the Pearson correlation across concepts, the sign-agreement rate, and the
    maximum absolute discrepancy.
    """
    w = np.asarray(weights, dtype=float)
    sensitivity = empirical_sensitivity(predict_fn, concepts, eps=eps)
    mean_sensitivity = sensitivity.mean(axis=0)

    if np.std(mean_sensitivity) < 1e-12 or np.std(w) < 1e-12:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(mean_sensitivity, w)[0, 1])

    # Sign agreement over concepts whose weight is materially non-zero: the sign of a
    # weight that is numerically zero is arbitrary and would add noise, not signal.
    material = np.abs(w) > 1e-6
    sign_agreement = (
        float(np.mean(np.sign(mean_sensitivity[material]) == np.sign(w[material])))
        if material.any()
        else float("nan")
    )

    names = concept_names or [f"concept_{j}" for j in range(w.size)]
    return {
        "correlation": correlation,
        "sign_agreement": sign_agreement,
        "max_abs_discrepancy": float(np.max(np.abs(mean_sensitivity - w))),
        "mean_abs_discrepancy": float(np.mean(np.abs(mean_sensitivity - w))),
        "per_concept": {
            name: {
                "declared_weight": float(w[j]),
                "measured_sensitivity": float(mean_sensitivity[j]),
                "discrepancy": float(mean_sensitivity[j] - w[j]),
            }
            for j, name in enumerate(names)
        },
    }


def _as_concept_matrix(concepts: npt.ArrayLike, n_concepts: int) -> npt.NDArray[np.float64]:
    c = np.asarray(concepts, dtype=float)
    if c.ndim == 1:
        c = c[None, :]
    if c.ndim != 2:
        raise ValueError(f"Concepts must be 1-D or 2-D, got shape {c.shape}")
    if c.shape[1] != n_concepts:
        raise ValueError(f"Expected {n_concepts} concepts, got {c.shape[1]}")
    return c
