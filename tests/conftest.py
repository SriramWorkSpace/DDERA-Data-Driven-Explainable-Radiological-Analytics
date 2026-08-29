"""Shared pytest fixtures.

All fixtures are built on :mod:`ddera.data.synthetic`, where the true reasoner weights and
true concepts are known. That is what lets the tests assert the harness **recovers** the
right answer rather than merely that it runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ddera.data.synthetic import make_synthetic_cbm  # noqa: E402
from ddera.xai.intervention import LinearReasoner  # noqa: E402


@pytest.fixture(scope="session")
def synth():
    """Default dataset: noisy concept predictor, no leakage past the bottleneck."""
    return make_synthetic_cbm(
        n_patients=300,
        studies_per_patient=2,
        n_concepts=8,
        concept_noise=0.8,
        leak_strength=0.0,
        seed=0,
    )


@pytest.fixture(scope="session")
def synth_clean():
    """A near-perfect concept predictor: interventions should have almost nothing to fix."""
    return make_synthetic_cbm(
        n_patients=300,
        studies_per_patient=2,
        n_concepts=8,
        concept_noise=0.01,
        concept_separability=6.0,
        leak_strength=0.0,
        seed=1,
    )


@pytest.fixture(scope="session")
def synth_leaky():
    """Target information injected into features that bypasses the concepts entirely."""
    return make_synthetic_cbm(
        n_patients=300,
        studies_per_patient=2,
        n_concepts=8,
        concept_noise=0.8,
        leak_strength=4.0,
        seed=0,
    )


@pytest.fixture(scope="session")
def synth_uncertain():
    """Raw labels containing uncertain (-1) and blank (NaN) entries."""
    return make_synthetic_cbm(
        n_patients=120,
        studies_per_patient=2,
        n_concepts=6,
        uncertain_rate=0.15,
        blank_rate=0.10,
        seed=3,
    )


@pytest.fixture
def reasoner(synth):
    """The TRUE linear reasoner that generated the synthetic targets."""
    return LinearReasoner(synth.weights, synth.bias, synth.concept_names)


@pytest.fixture
def predict_fn(reasoner):
    """The ``concepts -> probabilities`` callable the XAI harness consumes."""
    return reasoner.as_predict_fn()
