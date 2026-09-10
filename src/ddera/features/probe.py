"""Linear concept probe -- the GATE 2 sanity check (project-plan Phase 2 / ADR-008).

Fits a plain logistic regression from cached encoder features to each concept label and
reports per-concept AUROC. It answers one question: do the frozen features linearly carry
the clinical concepts at all? A near-chance probe means the encoder or the cache is broken,
not that the methodology failed -- this is a wiring check, not a research result.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ddera.eval.metrics import safe_auroc

_MIN_TRAIN = 10


def linear_probe_concept_auroc(
    features_train: npt.ArrayLike,
    labels_train: npt.ArrayLike,
    mask_train: npt.ArrayLike,
    features_eval: npt.ArrayLike,
    labels_eval: npt.ArrayLike,
    mask_eval: npt.ArrayLike,
    concept_names: list[str],
    *,
    C: float = 1.0,
    max_iter: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Per-concept logistic-regression AUROC from features to concept labels.

    ``mask_*`` (1 = usable, 0 = ignore) is the u_mask mask from
    :func:`ddera.data.labels.encode_concept_matrix`; masked rows are dropped per concept, so
    a concept the radiologist was uncertain about is not scored as if they had been certain.

    Returns ``{"per_concept": {name: {auroc, n_train, n_eval}}, "macro_auroc": float,
    "n_concepts_scored": int}``.
    """
    x_train = np.asarray(features_train, dtype=np.float64)
    x_eval = np.asarray(features_eval, dtype=np.float64)
    y_train = np.asarray(labels_train, dtype=np.float64)
    y_eval = np.asarray(labels_eval, dtype=np.float64)
    m_train = np.asarray(mask_train, dtype=np.float64) > 0
    m_eval = np.asarray(mask_eval, dtype=np.float64) > 0

    scaler = StandardScaler().fit(x_train)
    x_train, x_eval = scaler.transform(x_train), scaler.transform(x_eval)

    per_concept: dict[str, dict[str, float]] = {}
    for j, name in enumerate(concept_names):
        tr, ev = m_train[:, j], m_eval[:, j]
        entry = {"auroc": float("nan"), "n_train": int(tr.sum()), "n_eval": int(ev.sum())}
        if tr.sum() >= _MIN_TRAIN and ev.sum() > 0 and len(np.unique(y_train[tr, j])) == 2:
            clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
            clf.fit(x_train[tr], y_train[tr, j])
            scores = clf.predict_proba(x_eval[ev])[:, 1]
            entry["auroc"] = safe_auroc(y_eval[ev, j], scores)
        per_concept[name] = entry

    scored = [e["auroc"] for e in per_concept.values() if e["auroc"] == e["auroc"]]
    return {
        "per_concept": per_concept,
        "macro_auroc": float(np.mean(scored)) if scored else float("nan"),
        "n_concepts_scored": len(scored),
    }
