"""Synthetic concept-bottleneck data with known ground truth.

Used for testing the XAI harness *before* any real model exists. The point is that the true
reasoner weights, the true concepts, and the true generative process are all known, so tests
can assert that the harness **recovers** them rather than merely that it runs without
crashing.

The generative process mirrors the real setting:

.. code-block:: text

    c_true ~ Bernoulli(prevalence)                 latent clinical concepts
    y      ~ Bernoulli(sigmoid(w . c_true + b))    diagnosis from concepts
    X      = c_true @ A + noise                    encoder features carry concept info
    c_pred = sigmoid(alpha (2 c_true - 1) + eps)   an imperfect concept predictor

Two knobs matter for testing specific properties:

``leak_strength``
    Injects target information into the features that does **not** pass through the
    concepts. A leakage detector that cannot separate ``leak_strength=0`` from
    ``leak_strength=1`` is not measuring leakage.

``concept_noise``
    Controls how wrong ``c_pred`` is. Intervention curves must improve as predicted
    concepts are replaced with ground truth; with zero noise there is nothing to fix and the
    curve must stay flat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from ddera.data.labels import UNCERTAIN


@dataclass
class SyntheticCBM:
    """A synthetic dataset plus the ground truth that generated it."""

    concepts_true: npt.NDArray[np.float64]  # (n, k) binary
    concepts_pred: npt.NDArray[np.float64]  # (n, k) probabilities in (0, 1)
    features: npt.NDArray[np.float64]  # (n, d)
    y: npt.NDArray[np.int_]  # (n,) binary target
    weights: npt.NDArray[np.float64]  # (k,) TRUE reasoner weights
    bias: float  # TRUE reasoner bias
    patient_ids: npt.NDArray[np.str_]  # (n,)
    concept_names: list[str]
    raw_labels: npt.NDArray[np.float64] = field(default=None)  # (n, k) with -1 / NaN
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return self.concepts_true.shape[0]

    @property
    def n_concepts(self) -> int:
        return self.concepts_true.shape[1]

    def true_logits(self) -> npt.NDArray[np.float64]:
        """Logits under the TRUE weights and TRUE concepts -- the achievable ceiling."""
        return self.concepts_true @ self.weights + self.bias

    def to_manifest(self) -> pd.DataFrame:
        """An image-level manifest shaped like the real CheXpert one, for split testing."""
        return pd.DataFrame(
            {
                "patient_id": self.patient_ids,
                "study": [f"study{i % 3 + 1}" for i in range(self.n_samples)],
                "view": ["AP" if i % 2 else "PA" for i in range(self.n_samples)],
                "path": [f"synthetic/{p}/img{i}.jpg" for i, p in enumerate(self.patient_ids)],
                "target": self.y,
            }
        )


def make_synthetic_cbm(
    n_patients: int = 400,
    studies_per_patient: int = 2,
    n_concepts: int = 8,
    n_features: int = 64,
    *,
    concept_noise: float = 0.6,
    concept_separability: float = 2.0,
    leak_strength: float = 0.0,
    soft_leak_strength: float = 0.0,
    prevalence: float | npt.ArrayLike = 0.35,
    uncertain_rate: float = 0.0,
    blank_rate: float = 0.0,
    seed: int = 0,
) -> SyntheticCBM:
    """Generate a synthetic concept-bottleneck dataset.

    Args:
        n_patients: number of distinct patients (splitting is patient-level).
        studies_per_patient: rows per patient. Makes patient leakage detectable.
        n_concepts: bottleneck width.
        n_features: encoder feature dimension.
        concept_noise: std-dev of noise on the concept predictor's logits. Higher means
            ``c_pred`` is further from ``c_true``, so interventions have more to repair.
        concept_separability: how strongly ``c_pred`` separates the two concept classes
            before noise. Higher means a better concept predictor.
        leak_strength: magnitude of target information injected into features that bypasses
            the concepts. ``0`` means the concepts are complete; larger means information
            leaks around the bottleneck.
        soft_leak_strength: injects target information into the concept *probabilities*
            without materially changing their hard 0/1 thresholds. This reproduces genuine
            CBM soft-leakage: the concept vector looks symbolically correct, but its
            sub-symbolic precision smuggles extra signal that a reasoner trained on soft
            values will learn to exploit. Hardening the concepts then destroys that channel,
            which is what :func:`ddera.xai.leakage.soft_vs_hard_leakage` detects.
        prevalence: scalar or per-concept base rates.
        uncertain_rate: fraction of raw concept labels marked uncertain (-1).
        blank_rate: fraction of raw concept labels left blank (NaN).
        seed: RNG seed.

    Returns:
        A :class:`SyntheticCBM` carrying the data and the ground truth behind it.
    """
    rng = np.random.default_rng(seed)
    n = n_patients * studies_per_patient

    prev = (
        np.full(n_concepts, prevalence, dtype=float)
        if np.isscalar(prevalence)
        else np.asarray(prevalence, dtype=float)
    )
    if prev.shape != (n_concepts,):
        raise ValueError(f"prevalence must be scalar or shape ({n_concepts},), got {prev.shape}")

    # --- ground-truth concepts -------------------------------------------------------
    concepts_true = (rng.random((n, n_concepts)) < prev).astype(float)

    # --- ground-truth reasoner -------------------------------------------------------
    # Signed weights of varied magnitude: some concepts matter a lot, some barely at all.
    # The near-zero one is deliberate -- the permutation necessity test must flag it as
    # decorative, and that is a capability we need to verify.
    weights = rng.normal(0.0, 1.5, size=n_concepts)
    weights[0] = 2.5  # strongly positive
    weights[1] = -2.0  # strongly negative
    weights[-1] = 0.01  # decorative: present in the bottleneck, irrelevant to the target
    bias = -0.5

    logits = concepts_true @ weights + bias
    y = (rng.random(n) < _sigmoid(logits)).astype(int)

    # --- encoder features ------------------------------------------------------------
    projection = rng.normal(0.0, 1.0, size=(n_concepts, n_features))
    features = concepts_true @ projection + rng.normal(0.0, 0.5, size=(n, n_features))

    if leak_strength > 0:
        # A direction in feature space carrying target info that no concept explains.
        leak_dir = rng.normal(0.0, 1.0, size=n_features)
        leak_dir /= np.linalg.norm(leak_dir)
        features = features + leak_strength * np.outer(y - y.mean(), leak_dir)

    # --- imperfect concept predictor -------------------------------------------------
    concept_logits = concept_separability * (2 * concepts_true - 1) + rng.normal(
        0.0, concept_noise, size=(n, n_concepts)
    )
    if soft_leak_strength > 0:
        # Nudge the concept logits by the target. The shift is small relative to
        # `concept_separability`, so the hard (thresholded) concepts stay close to c_true
        # while the soft probabilities now carry target information they should not have.
        concept_logits = concept_logits + soft_leak_strength * (y - y.mean())[:, None]
    concepts_pred = _sigmoid(concept_logits)

    # --- raw CheXpert-style labels ---------------------------------------------------
    raw = concepts_true.copy()
    if uncertain_rate > 0:
        raw[rng.random((n, n_concepts)) < uncertain_rate] = UNCERTAIN
    if blank_rate > 0:
        raw[rng.random((n, n_concepts)) < blank_rate] = np.nan

    patient_ids = np.array([f"patient{i // studies_per_patient:05d}" for i in range(n)], dtype=str)

    return SyntheticCBM(
        concepts_true=concepts_true,
        concepts_pred=concepts_pred,
        features=features,
        y=y,
        weights=weights,
        bias=bias,
        patient_ids=patient_ids,
        concept_names=[f"concept_{i}" for i in range(n_concepts)],
        raw_labels=raw,
        meta={
            "n_patients": n_patients,
            "studies_per_patient": studies_per_patient,
            "concept_noise": concept_noise,
            "concept_separability": concept_separability,
            "leak_strength": leak_strength,
            "soft_leak_strength": soft_leak_strength,
            "hard_concept_error_rate": float(
                np.mean((concepts_pred >= 0.5).astype(float) != concepts_true)
            ),
            "prevalence": prev.tolist(),
            "target_prevalence": float(y.mean()),
            "seed": seed,
            "decorative_concept_index": n_concepts - 1,
        },
    )


def _sigmoid(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
