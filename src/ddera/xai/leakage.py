"""Leakage and concept-necessity analysis.

Two failure modes are routinely conflated in the CBM literature. DDERA measures them
separately, because they have different causes and different fixes.

**Leakage** (this module's namesake)
    The *soft* concept values encode information beyond their symbolic meaning. A concept
    labelled "Consolidation" whose probability drifts to 0.63 rather than resolving to 0/1
    can smuggle unrelated image information through the bottleneck, because the reasoner
    learns to read that extra precision. The bottleneck then looks intact while the
    explanation has quietly stopped being true. Symptom: interventions behave erratically,
    because replacing a soft value with a clean 0/1 destroys the smuggled signal.
    Measured by :func:`soft_vs_hard_leakage`.

**Incompleteness**
    The concepts genuinely do not carry all the task-relevant information. The bottleneck is
    honest; it is simply narrow. Symptom: a black box beats the CBM, and a residual channel
    recovers the gap. Measured by :func:`residual_probe_leakage` here and by the
    hybrid-residual sweep in :mod:`ddera.xai.completeness`.

Leakage is a defect. Incompleteness is a finding -- and quantifying it *is* the answer to
DDERA's research question.

A third measure, :func:`concept_permutation_necessity`, asks per concept whether the model
uses it at all. A concept that can be shuffled with no effect on performance is decorative,
and reporting that plainly is required by CLAUDE.md section 8.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

from ddera.eval.metrics import safe_auroc
from ddera.xai.intervention import logit

PredictFn = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


def concept_permutation_necessity(
    predict_fn: PredictFn,
    concepts: npt.ArrayLike,
    y_true: npt.ArrayLike,
    *,
    concept_names: list[str] | None = None,
    n_repeats: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    """Per-concept necessity by permutation.

    Shuffles one concept's column across samples (destroying its association with the
    target while preserving its marginal distribution) and measures the AUROC drop.

    Interpretation:

    - **Large drop** -- the model genuinely depends on this concept.
    - **Near zero** -- the concept is *decorative*: it appears in the explanation but does
      not affect the decision. This must be reported, not hidden.
    - **Negative** -- shuffling helped; the concept is actively harmful, usually a sign of
      a mislearned or noisy concept head.

    Returns per-concept mean/std AUROC drop and the list of decorative concepts.
    """
    c = np.asarray(concepts, dtype=float)
    y = np.asarray(y_true)
    n, k = c.shape
    names = concept_names or [f"concept_{j}" for j in range(k)]
    if len(names) != k:
        raise ValueError(f"Got {k} concepts but {len(names)} names.")

    rng = np.random.default_rng(seed)
    baseline = safe_auroc(y, np.asarray(predict_fn(c), dtype=float))

    per_concept: dict[str, dict[str, float]] = {}
    for j, name in enumerate(names):
        drops = []
        for _ in range(n_repeats):
            permuted = c.copy()
            permuted[:, j] = c[rng.permutation(n), j]
            drops.append(baseline - safe_auroc(y, np.asarray(predict_fn(permuted), dtype=float)))
        arr = np.asarray(drops, dtype=float)
        per_concept[name] = {
            "auroc_drop_mean": float(np.mean(arr)),
            "auroc_drop_std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "auroc_permuted_mean": float(baseline - np.mean(arr)),
        }

    drops_only = {n_: v["auroc_drop_mean"] for n_, v in per_concept.items()}
    return {
        "baseline_auroc": baseline,
        "per_concept": per_concept,
        "n_repeats": n_repeats,
        "ranking": sorted(drops_only, key=lambda n_: -drops_only[n_]),
        "decorative_concepts": [n_ for n_, d in drops_only.items() if abs(d) < 0.005],
    }


def soft_vs_hard_leakage(
    predict_fn: PredictFn,
    concepts: npt.ArrayLike,
    y_true: npt.ArrayLike,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Measure how much the model relies on sub-symbolic precision in concept values.

    Compares AUROC using the soft concept probabilities against AUROC using the same
    concepts thresholded to ``{0, 1}``. If the model is genuinely reasoning over the
    *symbols* it names, hardening them should cost little. A large drop means the reasoner
    depends on precision the concept labels never carried -- the definition of leakage.

    ``leakage`` here is ``AUROC(soft) - AUROC(hard)``:

    - ``~0``       clean bottleneck; the explanation means what it says
    - **positive** leakage; interventions with clean 0/1 values will misbehave
    - **negative** hardening helped, usually a symptom of miscalibrated concept outputs
    """
    c = np.asarray(concepts, dtype=float)
    y = np.asarray(y_true)

    auroc_soft = safe_auroc(y, np.asarray(predict_fn(c), dtype=float))
    auroc_hard = safe_auroc(y, np.asarray(predict_fn((c >= threshold).astype(float)), dtype=float))

    return {
        "auroc_soft": auroc_soft,
        "auroc_hard": auroc_hard,
        "leakage": auroc_soft - auroc_hard,
        "threshold": threshold,
        "interpretation": _describe_leakage(auroc_soft - auroc_hard),
    }


def residual_probe_leakage(
    features: npt.ArrayLike,
    y_true: npt.ArrayLike,
    base_probs: npt.ArrayLike,
    *,
    cv: int = 5,
    seed: int = 0,
    max_iter: int = 2000,
) -> dict[str, Any]:
    """How much task information lives in the encoder features but not in the concepts.

    Two complementary probes, both cross-validated so the numbers are honest out-of-sample
    estimates rather than training-set fits:

    1. **Direct probe** -- logistic regression from features to the target.
       ``delta_auroc = AUROC(features) - AUROC(concept bottleneck)``. Positive means the
       encoder knows things the bottleneck discarded.
    2. **Residual probe** -- ridge regression from features to the logit residual
       ``logit(y) - logit(p_base)``, reported as out-of-sample R-squared. Non-zero R-squared
       means the features can systematically predict the bottleneck's *errors*.

    This measures **incompleteness**, not leakage. In a real CBM the concepts are computed
    *from* these features, so ``delta_auroc >= 0`` almost by construction; the magnitude is
    the interesting quantity, and it is what the Phase 4 hybrid-residual sweep independently
    corroborates.
    """
    X = np.asarray(features, dtype=float)
    y = np.asarray(y_true).astype(int)
    p_base = np.clip(np.asarray(base_probs, dtype=float), 1e-6, 1 - 1e-6)
    if X.shape[0] != y.shape[0] or y.shape[0] != p_base.shape[0]:
        raise ValueError(
            f"Row mismatch: features {X.shape[0]}, y {y.shape[0]}, base_probs {p_base.shape[0]}"
        )

    auroc_base = safe_auroc(y, p_base)
    X_scaled = StandardScaler().fit_transform(X)

    probe = LogisticRegression(max_iter=max_iter, random_state=seed)
    probe_probs = cross_val_predict(probe, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
    auroc_probe = safe_auroc(y, probe_probs)

    # Ridge on the logit residual. y is 0/1, so logit(y) is infinite; use the standard
    # working-response form instead: the residual on the logit scale that a single Newton
    # step would need, (y - p) / (p (1 - p)), which is finite and well defined.
    working_residual = (y - p_base) / np.clip(p_base * (1 - p_base), 1e-6, None)
    ridge_pred = cross_val_predict(Ridge(alpha=1.0), X_scaled, working_residual, cv=cv)
    ss_res = float(np.sum((working_residual - ridge_pred) ** 2))
    ss_tot = float(np.sum((working_residual - working_residual.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "auroc_bottleneck": auroc_base,
        "auroc_feature_probe": auroc_probe,
        "delta_auroc": auroc_probe - auroc_base,
        "residual_r2": r2,
        "cv_folds": cv,
        "n_features": int(X.shape[1]),
        "interpretation": _describe_incompleteness(auroc_probe - auroc_base),
    }


def leakage_report(
    predict_fn: PredictFn,
    concepts: npt.ArrayLike,
    y_true: npt.ArrayLike,
    *,
    features: npt.ArrayLike | None = None,
    base_probs: npt.ArrayLike | None = None,
    concept_names: list[str] | None = None,
    n_repeats: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    """The full Phase 5 leakage family, as written into ``metrics.json``."""
    report: dict[str, Any] = {
        "necessity": concept_permutation_necessity(
            predict_fn,
            concepts,
            y_true,
            concept_names=concept_names,
            n_repeats=n_repeats,
            seed=seed,
        ),
        "soft_vs_hard": soft_vs_hard_leakage(predict_fn, concepts, y_true),
    }
    if features is not None:
        probs = base_probs if base_probs is not None else predict_fn(np.asarray(concepts, float))
        report["residual_probe"] = residual_probe_leakage(features, y_true, probs, seed=seed)
    return report


def _describe_leakage(value: float) -> str:
    if value != value:
        return "undefined (degenerate labels)"
    if abs(value) < 0.01:
        return "negligible: the reasoner uses concepts symbolically, as claimed"
    if value > 0.05:
        return "substantial: the reasoner depends on sub-symbolic concept precision"
    if value > 0.01:
        return "mild: some reliance on sub-symbolic concept precision"
    return "negative: hardening improved AUROC; check concept calibration"


def _describe_incompleteness(delta: float) -> str:
    if delta != delta:
        return "undefined (degenerate labels)"
    if delta < 0.01:
        return "concepts capture essentially all task-relevant information in the features"
    if delta < 0.05:
        return "concepts capture most task-relevant information; small residual gap"
    return "substantial information bypassed by the bottleneck; report as an interpretability cost"


def residual_working_response(
    y_true: npt.ArrayLike, probs: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Working response ``(y - p) / (p (1 - p))`` on the logit scale.

    Exposed separately because it is also the natural per-sample residual for error analysis
    in Phase 5.8: large magnitude marks a sample the bottleneck explains badly.
    """
    p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
    return (np.asarray(y_true, dtype=float) - p) / (p * (1 - p))


__all__ = [
    "concept_permutation_necessity",
    "leakage_report",
    "logit",
    "residual_probe_leakage",
    "residual_working_response",
    "soft_vs_hard_leakage",
]
