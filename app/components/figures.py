"""App figures -- built through :mod:`ddera.reporting.theme` so they match the notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ddera.reporting.theme import ACCENT, apply_theme
from ddera.xai.intervention import LinearReasoner

_POS = "#16a34a"
_NEG = "#dc2626"


def concept_probability_bars(
    concepts: np.ndarray, concept_names: list[str], *, edited: np.ndarray | None = None
) -> plt.Figure:
    """Predicted concept probabilities; if ``edited`` is given, overlay the edited values."""
    apply_theme()
    c = np.asarray(concepts, dtype=float).reshape(-1)
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(concept_names) + 1.4))
    y = np.arange(len(concept_names))
    ax.barh(y, c, color=ACCENT, label="predicted")
    if edited is not None:
        ax.barh(
            y,
            np.asarray(edited, dtype=float).reshape(-1),
            height=0.4,
            color="#f59e0b",
            label="edited",
        )
        ax.legend(fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(concept_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("probability")
    ax.set_title("Concept vector")
    fig.tight_layout()
    return fig


def contribution_waterfall(
    reasoner: LinearReasoner, concepts: np.ndarray, concept_names: list[str]
) -> plt.Figure:
    """bias + each signed ``w_j * c_j``, cumulative, landing exactly on the logit."""
    apply_theme()
    c = np.asarray(concepts, dtype=float).reshape(-1)
    contribs = reasoner.weights * c
    labels = ["(bias)", *concept_names, "= logit"]
    steps = np.concatenate([[reasoner.bias], contribs, [0.0]])

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(labels) + 1.4))
    running = 0.0
    for i, (lab, step) in enumerate(zip(labels, steps, strict=True)):
        if lab == "= logit":
            ax.barh(i, running, color="#334155")
            ax.text(running, i, f" {running:+.2f}", va="center", fontsize=8)
            continue
        colour = _POS if step >= 0 else _NEG
        ax.barh(i, step, left=running, color=colour)
        running += step
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#94a3b8", lw=0.8)
    ax.set_xlabel("contribution to the logit")
    ax.set_title("Contribution waterfall  logit = b + sum_j w_j c_j")
    fig.tight_layout()
    return fig


def roc_curve_figure(target: np.ndarray, prob: np.ndarray) -> plt.Figure:
    apply_theme()
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(target, prob)
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot(fpr, tpr, color=ACCENT, lw=2)
    ax.plot([0, 1], [0, 1], ls="--", color="#94a3b8", lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC")
    fig.tight_layout()
    return fig
