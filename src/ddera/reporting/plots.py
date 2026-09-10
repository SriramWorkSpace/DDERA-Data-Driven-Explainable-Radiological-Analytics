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
import numpy.typing as npt
import pandas as pd

from ddera.config import ConceptSpec, EncoderConfig
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


# ---------------------------------------------------------------------------------------
# Phase 2 / preprocessing + feature views
# ---------------------------------------------------------------------------------------


def _denormalise(tensor: npt.ArrayLike, encoder: EncoderConfig) -> np.ndarray:
    """CHW normalised tensor -> HWC float image in [0, 1] for display."""
    arr = np.asarray(tensor, dtype=np.float32)
    mean = np.asarray(encoder.normalization_mean, dtype=np.float32)[:, None, None]
    std = np.asarray(encoder.normalization_std, dtype=np.float32)[:, None, None]
    return np.clip(arr * std + mean, 0.0, 1.0).transpose(1, 2, 0)


def plot_augmentation_grid(
    image_rgb_uint8: np.ndarray, train_transform, *, n: int = 8, seed: int = 0
) -> plt.Figure:
    """``n`` draws of the training augmentation on one image -- shows the jitter is small
    and (ADR-006) never a reflection.
    """
    enc = EncoderConfig()
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.4 * rows))
    for k, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if k >= n:
            continue
        np.random.seed(seed + k)  # noqa: NPY002 - albumentations uses the legacy global RNG
        out = train_transform(image=image_rgb_uint8)["image"]
        ax.imshow(_denormalise(out, enc))
        ax.set_title(f"draw {k + 1}", fontsize=8)
    fig.suptitle("Training augmentation draws (ADR-006: no reflection)", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_preprocessed_samples(
    dataset, *, n: int = 8, encoder: EncoderConfig | None = None
) -> plt.Figure:
    """A grid of post-transform images from a :class:`CheXpertImageDataset`, with their
    concept counts -- a quick "does the pipeline produce sane tensors" view.
    """
    enc = encoder or EncoderConfig()
    n = min(n, len(dataset))
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.7 * rows))
    for k, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if k >= n:
            continue
        item = dataset[k]
        ax.imshow(_denormalise(item["image"], enc))
        n_pos = int((item["concepts"].numpy() * item["concept_mask"].numpy()).sum())
        ax.set_title(f"y={int(item['target'].item())}  concepts+={n_pos}", fontsize=8)
    fig.suptitle("Preprocessed samples", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_feature_space(
    features: npt.ArrayLike,
    colour_values: npt.ArrayLike,
    *,
    title: str = "Encoder feature space (PCA)",
    colour_label: str = "target",
) -> plt.Figure:
    """2-D PCA scatter of cached features, coloured by a label vector. A sanity view of the
    representation, not a result -- clusters here are not evidence of anything on their own.
    """
    import warnings

    from sklearn.decomposition import PCA

    x = np.asarray(features, dtype=np.float64)
    with warnings.catch_warnings():
        # A near-degenerate feature matrix (e.g. an untrained encoder on tiny demo images)
        # makes PCA's explained-variance ratio divide by ~0. Harmless for a layout view.
        warnings.simplefilter("ignore", RuntimeWarning)
        coords = PCA(n_components=2, random_state=0).fit_transform(x)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], c=np.asarray(colour_values), cmap="coolwarm", s=12, alpha=0.8
    )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, label=colour_label, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_concept_probe_auroc(probe_result: dict) -> plt.Figure:
    """Per-concept linear-probe AUROC (GATE 2 sanity check). 0.5 = chance."""
    per = probe_result["per_concept"]
    names = list(per)
    aurocs = [per[n]["auroc"] for n in names]
    order = np.argsort([a if a == a else -1 for a in aurocs])
    names = [names[i] for i in order]
    aurocs = [aurocs[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 0.4 * len(names) + 1.6))
    ax.barh(names, [0.0 if a != a else a for a in aurocs], color=ACCENT)
    ax.axvline(0.5, ls="--", color="#334155", label="chance")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("probe AUROC")
    macro = probe_result.get("macro_auroc", float("nan"))
    ax.set_title(f"Linear concept probe - macro AUROC {macro:.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_feature_cache_sanity(
    cached: npt.ArrayLike, recomputed: npt.ArrayLike, *, max_points: int = 4000
) -> plt.Figure:
    """Cached float16 features vs a fresh encoder forward pass: they must lie on ``y = x``."""
    a = np.asarray(cached, dtype=np.float64).ravel()
    b = np.asarray(recomputed, dtype=np.float64).ravel()
    if a.size > max_points:
        idx = np.random.default_rng(0).choice(a.size, size=max_points, replace=False)
        a, b = a[idx], b[idx]
    max_abs_err = float(np.max(np.abs(a - b))) if a.size else 0.0

    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
    ax.plot([lo, hi], [lo, hi], color="#334155", ls="--", lw=1)
    ax.scatter(a, b, s=6, alpha=0.35, color=ACCENT)
    ax.set_xlabel("cached (float16)")
    ax.set_ylabel("fresh encoder forward")
    ax.set_title(f"Feature-cache round-trip\nmax abs err {max_abs_err:.2e}")
    fig.tight_layout()
    return fig
