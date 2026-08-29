"""Predictive and concept-quality metrics.

Accuracy alone is meaningless on imbalanced medical data: predicting "no pneumonia" for
everyone scores ~98% on CheXpert. Every reported result therefore carries AUROC, AUPRC, and
the sensitivity/specificity pair.

Degenerate cases (a split containing only one class) return ``nan`` rather than raising, so
a per-concept sweep over a rare finding does not abort the whole evaluation. ``nan`` is
propagated honestly and shown as "undefined" rather than silently imputed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

CHANCE_AUROC = 0.5


def safe_auroc(y_true: npt.ArrayLike, y_score: npt.ArrayLike) -> float:
    """AUROC, or ``nan`` when only one class is present."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, np.asarray(y_score)))


def safe_auprc(y_true: npt.ArrayLike, y_score: npt.ArrayLike) -> float:
    """Average precision, or ``nan`` when only one class is present.

    More informative than AUROC under heavy imbalance: its baseline is the positive rate,
    so a rare-finding model cannot look good simply by being conservative.
    """
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, np.asarray(y_score)))


def binary_metrics(
    y_true: npt.ArrayLike,
    y_score: npt.ArrayLike,
    threshold: float = 0.5,
) -> dict[str, float]:
    """The standard binary panel at a fixed operating point.

    Returns AUROC, AUPRC, accuracy, precision, recall/sensitivity, specificity, F1, the
    confusion-matrix cells, prevalence, and the Brier score.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_score {y_score.shape}")
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on an empty array.")

    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and (precision + recall) > 0
        else float("nan")
    )

    return {
        "auroc": safe_auroc(y_true, y_score),
        "auprc": safe_auprc(y_true, y_score),
        "accuracy": float((y_pred == y_true).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "prevalence": float(y_true.mean()),
        "brier": float(brier_score_loss(y_true, np.clip(y_score, 0, 1)))
        if len(np.unique(y_true)) > 1
        else float("nan"),
        "threshold": float(threshold),
        "n": int(y_true.size),
    }


def concept_metrics(
    c_true: npt.ArrayLike,
    c_score: npt.ArrayLike,
    concept_names: list[str],
    mask: npt.ArrayLike | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Per-concept quality plus macro averages.

    Args:
        c_true: ``(n, k)`` binary ground-truth concepts.
        c_score: ``(n, k)`` predicted concept probabilities.
        concept_names: ``k`` names, used as result keys.
        mask: optional ``(n, k)`` validity mask. Masked-out entries are excluded per
            concept, which is what makes the ``u_mask`` policy evaluable: a concept the
            radiologist was uncertain about is not scored as if they had been certain.
        threshold: operating point for the thresholded metrics.

    Returns:
        ``{"per_concept": {name: {...}}, "macro": {...}, "coverage": {name: float}}``.
        Macro averages ignore ``nan`` concepts and report how many contributed.
    """
    c_true = np.asarray(c_true, dtype=float)
    c_score = np.asarray(c_score, dtype=float)
    if c_true.shape != c_score.shape:
        raise ValueError(f"Shape mismatch: c_true {c_true.shape} vs c_score {c_score.shape}")
    if c_true.shape[1] != len(concept_names):
        raise ValueError(f"Got {c_true.shape[1]} concept columns but {len(concept_names)} names.")
    mask_arr = np.ones_like(c_true) if mask is None else np.asarray(mask, dtype=float)

    per_concept: dict[str, dict[str, float]] = {}
    coverage: dict[str, float] = {}

    for j, name in enumerate(concept_names):
        keep = mask_arr[:, j] > 0
        coverage[name] = float(keep.mean())
        if keep.sum() == 0:
            per_concept[name] = {k: float("nan") for k in ("auroc", "auprc", "f1")}
            per_concept[name]["n"] = 0
            continue
        per_concept[name] = binary_metrics(c_true[keep, j], c_score[keep, j], threshold)

    macro: dict[str, float] = {}
    for key in ("auroc", "auprc", "f1", "sensitivity", "specificity", "precision"):
        values = [m[key] for m in per_concept.values() if key in m and m[key] == m[key]]
        macro[f"macro_{key}"] = float(np.mean(values)) if values else float("nan")
    macro["n_concepts_scored"] = int(
        sum(1 for m in per_concept.values() if m.get("auroc", float("nan")) == m.get("auroc"))
    )
    macro["n_concepts_total"] = len(concept_names)

    return {"per_concept": per_concept, "macro": macro, "coverage": coverage}


def prediction_agreement(p_a: npt.ArrayLike, p_b: npt.ArrayLike, threshold: float = 0.5) -> float:
    """Fraction of samples on which two probability vectors give the same hard label."""
    a = (np.asarray(p_a, dtype=float) >= threshold).astype(int)
    b = (np.asarray(p_b, dtype=float) >= threshold).astype(int)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return float((a == b).mean())
