"""The concept-intervention identity -- the demonstration of the whole thesis.

Because the reasoner is linear, editing concept ``j`` from ``c_j`` to ``v`` moves the logit
by **exactly** ``w_j * (v - c_j)``. :func:`intervene` computes the actual shift and the
closed-form expectation and reports whether they match to numerical precision. If they ever
disagree, the explanation is not the mechanism -- and the app should say so loudly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ddera.xai.intervention import LinearReasoner, sigmoid


def intervene(
    reasoner: LinearReasoner,
    concepts: np.ndarray,
    edits: dict[int, float],
) -> dict[str, Any]:
    """Apply ``edits`` (concept index -> new value) and return the before/after picture."""
    base = np.asarray(concepts, dtype=float).reshape(-1)
    edited = base.copy()
    for j, value in edits.items():
        edited[j] = value

    base_logit = float(reasoner.predict_logit(base[None])[0])
    new_logit = float(reasoner.predict_logit(edited[None])[0])
    logit_delta = new_logit - base_logit
    expected_delta = float(sum(reasoner.weights[j] * (v - base[j]) for j, v in edits.items()))

    return {
        "edited_concepts": edited,
        "base_prob": float(sigmoid(base_logit)),
        "new_prob": float(sigmoid(new_logit)),
        "base_logit": base_logit,
        "new_logit": new_logit,
        "logit_delta": logit_delta,
        "expected_delta": expected_delta,
        "residual": abs(logit_delta - expected_delta),
        "exact": abs(logit_delta - expected_delta) < 1e-9,
    }


def contribution_table(
    reasoner: LinearReasoner, concepts: np.ndarray, concept_names: list[str]
) -> list[dict[str, Any]]:
    """Per-concept ``(name, c_j, w_j, w_j * c_j)`` plus a bias row; the ``contribution``
    column plus the bias sum exactly to the logit.
    """
    c = np.asarray(concepts, dtype=float).reshape(-1)
    rows = [
        {
            "concept": name,
            "value": float(c[j]),
            "weight": float(reasoner.weights[j]),
            "contribution": float(reasoner.weights[j] * c[j]),
        }
        for j, name in enumerate(concept_names)
    ]
    rows.append(
        {
            "concept": "(bias)",
            "value": 1.0,
            "weight": float(reasoner.bias),
            "contribution": float(reasoner.bias),
        }
    )
    return rows
