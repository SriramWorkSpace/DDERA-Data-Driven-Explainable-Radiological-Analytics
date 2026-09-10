"""Progress-report figures built from real project artifacts.

Every function takes already-loaded data (a manifest / splits DataFrame, a ``ConceptSpec``,
a probe result) and returns a matplotlib ``Figure``. **Nothing here computes a research
result** -- these are distribution and sanity-check views for a progress presentation.
Figures made from synthetic data must be saved with ``synthetic=True``
(:func:`ddera.reporting.theme.save_figure`).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ddera.config import ConceptSpec
from ddera.data.chexpert import concept_matrix, cooccurrence_matrix
from ddera.data.labels import encode_concept_matrix, label_distribution, mask_coverage
from ddera.reporting.theme import ACCENT, LABEL_COLORS, SPLIT_COLORS

_SPLIT_ORDER = ("train", "val", "test")


def _splits_present(splits_df: pd.DataFrame) -> list[str]:
    return [s for s in _SPLIT_ORDER if s in set(splits_df["split"].unique())]


# ---------------------------------------------------------------------------------------
# Phase 1 / data-layer views
# ---------------------------------------------------------------------------------------


def plot_split_summary(splits_df: pd.DataFrame) -> plt.Figure:
    """Patients and images per split -- the patient-level split (ADR-005) at a glance."""
    order = _splits_present(splits_df)
    grouped = splits_df.groupby("split")
    n_patients = [grouped.get_group(s)["patient_id"].nunique() for s in order]
    n_images = [len(grouped.get_group(s)) for s in order]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, values, title in (
        (axes[0], n_patients, "Patients per split"),
        (axes[1], n_images, "Images per split"),
    ):
        ax.bar(order, values, color=[SPLIT_COLORS[s] for s in order])
        ax.set_title(title)
        for x, v in enumerate(values):
            ax.text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Patient-level split (ADR-005)", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_target_prevalence_by_split(splits_df: pd.DataFrame, target: str = "target") -> plt.Figure:
    """Per-split target prevalence against the pooled rate (stratification check, ADR-011)."""
    order = _splits_present(splits_df)
    grouped = splits_df.groupby("split")
    prevalence = [float(grouped.get_group(s)[target].mean()) for s in order]
    pooled = float(splits_df[target].mean())

    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.bar(order, prevalence, color=[SPLIT_COLORS[s] for s in order])
    ax.axhline(pooled, ls="--", color="#334155", label=f"pooled = {pooled:.3f}")
    for x, v in enumerate(prevalence):
        ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("prevalence")
    ax.set_title("Target prevalence by split")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_label_distribution(manifest: pd.DataFrame, observations: list[str]) -> plt.Figure:
    """Stacked positive / negative / uncertain / blank counts per observation (ADR-004)."""
    names: list[str] = []
    stacks = {"positive": [], "negative": [], "uncertain": [], "blank": []}
    for obs in observations:
        col = f"raw_{obs}"
        if col not in manifest.columns:
            continue
        dist = label_distribution(manifest[col].to_numpy())
        names.append(obs)
        for key in stacks:
            stacks[key].append(dist[key])

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(names) + 1.6))
    left = np.zeros(len(names))
    for key in ("positive", "negative", "uncertain", "blank"):
        arr = np.asarray(stacks[key], dtype=float)
        ax.barh(names, arr, left=left, color=LABEL_COLORS[key], label=key)
        left += arr
    ax.invert_yaxis()
    ax.set_xlabel("count")
    ax.set_title("CheXpert label distribution  (-1 = uncertain, blank = not mentioned)")
    ax.legend(ncol=4, loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_concept_cooccurrence(manifest: pd.DataFrame, concepts: list[str]) -> plt.Figure:
    """Row-normalised concept co-occurrence, ``P(col positive | row positive)``.

    Concepts that almost always co-occur cannot be independently intervened upon, which
    matters for how the Phase-5 intervention results should be read.
    """
    counts = cooccurrence_matrix(manifest, concepts).to_numpy(dtype=float)
    diag = np.clip(np.diag(counts), 1.0, None)
    norm = counts / diag[:, None]

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(concepts)))
    ax.set_xticklabels(concepts, rotation=90, fontsize=7)
    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels(concepts, fontsize=7)
    ax.set_title("Concept co-occurrence  P(col | row)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_mask_coverage(manifest: pd.DataFrame, spec: ConceptSpec) -> plt.Figure:
    """Fraction of each concept's labels that survive the uncertainty policy.

    A concept the model barely gets to learn will score poorly on concept AUROC; this shows
    whether that is a data limitation rather than a modelling failure.
    """
    raw = concept_matrix(manifest, spec.concepts)
    _, mask = encode_concept_matrix(
        raw,
        policy=spec.uncertainty.concept_policy,
        blank_policy=spec.uncertainty.blank_policy,
    )
    coverage = mask_coverage(mask)
    per_concept = [coverage[f"concept_{j}"] for j in range(len(spec.concepts))]

    fig, ax = plt.subplots(figsize=(9, 0.4 * len(spec.concepts) + 1.6))
    ax.barh(spec.concepts, per_concept, color=ACCENT)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("fraction of labels usable")
    ax.set_title(
        f"Concept label coverage - policy '{spec.uncertainty.concept_policy}' "
        f"(overall {coverage['overall']:.2f})"
    )
    fig.tight_layout()
    return fig
