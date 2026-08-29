"""Label encoding and uncertainty policies (ADR-004).

CheXpert-style label semantics:

===========  ======================================================================
Raw value    Meaning
===========  ======================================================================
``1.0``      Positive -- the observation was asserted in the radiology report
``0.0``      Negative -- the observation was explicitly negated
``-1.0``     Uncertain -- the report expressed genuine uncertainty
``NaN``      Blank -- the observation was not mentioned at all
===========  ======================================================================

The uncertain class is not noise; it is a radiologist declining to commit. How we treat it
materially changes results, so the policy is explicit, configurable, and reported. The
default (``u_mask``) declines to invent a label, which preserves concept *semantics* at the
cost of some training signal. Phase 6 revisits this by modelling the uncertainty explicitly
rather than masking it.

Everything here returns an explicit ``(labels, mask)`` pair. The mask is the mechanism that
makes ``u_mask`` work: masked elements contribute no gradient, so the model is never taught
a value the radiologist did not assert.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

POSITIVE = 1.0
NEGATIVE = 0.0
UNCERTAIN = -1.0

ConceptPolicy = Literal["u_mask", "u_zeros", "u_ones", "u_ignore"]
BlankPolicy = Literal["negative", "mask"]

VALID_CONCEPT_POLICIES: tuple[str, ...] = ("u_mask", "u_zeros", "u_ones", "u_ignore")
VALID_BLANK_POLICIES: tuple[str, ...] = ("negative", "mask")

#: The 14 CheXpert observations, in the dataset's own column order.
CHEXPERT_OBSERVATIONS: tuple[str, ...] = (
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
)


def apply_uncertainty_policy(
    raw: npt.ArrayLike,
    policy: ConceptPolicy = "u_mask",
    blank_policy: BlankPolicy = "negative",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Map raw CheXpert label values to ``(labels, mask)``.

    Args:
        raw: array of raw values in {1, 0, -1, NaN}. Any shape.
        policy: how to treat uncertain (-1) entries.

            - ``u_mask``   -- mask them out; they contribute no gradient (**default**)
            - ``u_zeros``  -- treat as negative
            - ``u_ones``   -- treat as positive
            - ``u_ignore`` -- identical masking to ``u_mask`` at element level; used for a
              single target column, where :func:`rows_to_keep` then drops the whole sample

        blank_policy: how to treat blanks (NaN). ``negative`` follows the CheXpert
            convention that an unmentioned observation is absent; ``mask`` declines to
            assume that.

    Returns:
        ``(labels, mask)``, both float arrays shaped like ``raw``. ``labels`` contains only
        0.0/1.0. ``mask`` is 1.0 where the label is usable and 0.0 where it must be ignored.

    Note:
        Masked positions have their label set to 0.0 rather than left as -1 or NaN, so that
        an accidental unmasked loss computation degrades gracefully to "negative" instead of
        producing NaN gradients that silently poison training.
    """
    if policy not in VALID_CONCEPT_POLICIES:
        raise ValueError(f"Unknown policy {policy!r}; expected one of {VALID_CONCEPT_POLICIES}")
    if blank_policy not in VALID_BLANK_POLICIES:
        raise ValueError(
            f"Unknown blank_policy {blank_policy!r}; expected one of {VALID_BLANK_POLICIES}"
        )

    values = np.asarray(raw, dtype=np.float64)
    labels = values.copy()
    mask = np.ones_like(labels, dtype=np.float64)

    is_blank = np.isnan(labels)
    if blank_policy == "negative":
        labels[is_blank] = NEGATIVE
    else:
        labels[is_blank] = NEGATIVE
        mask[is_blank] = 0.0

    is_uncertain = labels == UNCERTAIN
    if policy in ("u_mask", "u_ignore"):
        labels[is_uncertain] = NEGATIVE
        mask[is_uncertain] = 0.0
    elif policy == "u_zeros":
        labels[is_uncertain] = NEGATIVE
    elif policy == "u_ones":
        labels[is_uncertain] = POSITIVE

    _validate_binary(labels)
    return labels, mask


def rows_to_keep(
    raw_target: npt.ArrayLike,
    policy: ConceptPolicy = "u_ignore",
    blank_policy: BlankPolicy = "negative",
) -> npt.NDArray[np.bool_]:
    """Boolean selector over samples for a single target column.

    ``u_ignore`` drops samples whose target is uncertain. Every other policy keeps all rows
    (blanks included, when ``blank_policy='negative'``), since it assigns them a label.
    """
    values = np.asarray(raw_target, dtype=np.float64)
    keep = np.ones(values.shape, dtype=bool)

    if blank_policy == "mask":
        keep &= ~np.isnan(values)
    if policy == "u_ignore":
        keep &= values != UNCERTAIN
    return keep


def label_distribution(raw: npt.ArrayLike) -> dict[str, int]:
    """Count positive / negative / uncertain / blank. The first thing EDA should report."""
    values = np.asarray(raw, dtype=np.float64)
    return {
        "positive": int(np.sum(values == POSITIVE)),
        "negative": int(np.sum(values == NEGATIVE)),
        "uncertain": int(np.sum(values == UNCERTAIN)),
        "blank": int(np.sum(np.isnan(values))),
        "total": int(values.size),
    }


def encode_concept_matrix(
    raw_matrix: npt.ArrayLike,
    policy: ConceptPolicy = "u_mask",
    blank_policy: BlankPolicy = "negative",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Apply the policy to an ``(n_samples, n_concepts)`` matrix.

    Returns ``(labels, mask)`` of the same shape. This pair is exactly what the masked BCE
    concept loss consumes.
    """
    matrix = np.asarray(raw_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D (n_samples, n_concepts) matrix, got shape {matrix.shape}")
    return apply_uncertainty_policy(matrix, policy=policy, blank_policy=blank_policy)


def mask_coverage(mask: npt.ArrayLike) -> dict[str, float]:
    """Fraction of usable labels overall and per concept.

    Low coverage on a concept is a warning: a concept the model barely gets to learn will
    have poor concept AUROC, which then looks like a modelling failure rather than a data
    limitation. Report this alongside concept quality.
    """
    m = np.asarray(mask, dtype=np.float64)
    out: dict[str, float] = {"overall": float(m.mean())}
    if m.ndim == 2:
        for j in range(m.shape[1]):
            out[f"concept_{j}"] = float(m[:, j].mean())
    return out


def _validate_binary(labels: npt.NDArray[np.float64]) -> None:
    """Guard against a policy bug silently leaving -1 or NaN in the label array."""
    if np.isnan(labels).any():
        raise ValueError("NaN survived the uncertainty policy; this is a bug in label encoding.")
    bad = np.setdiff1d(np.unique(labels), np.array([NEGATIVE, POSITIVE]))
    if bad.size:
        raise ValueError(
            f"Labels contain values outside {{0, 1}} after encoding: {bad.tolist()}. "
            "This is a bug in label encoding."
        )
