"""The eight-family XAI protocol, assembled from artifacts (Invariant 6, ARCHITECTURE 6/8).

Runs off a ``predictions``-shaped DataFrame + a ``concept_weights`` dict -- no live model, no
GPU -- so the protocol is reproducible from a run directory alone. Built on
``make_synthetic_cbm`` where the true reasoner is known: a clean predictor recovers its
concepts, a leaky one is flagged, and a run with perturbed columns + a baseline reaches
``complete``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ddera.data.synthetic import make_synthetic_cbm
from ddera.xai.intervention import LinearReasoner
from ddera.xai.protocol import FAMILIES, run_protocol


def _frame(s, *, reasoner: LinearReasoner, perturbations: dict[str, np.ndarray] | None = None):
    """A predictions frame from a SyntheticCBM and a reasoner."""
    names = s.concept_names
    probs = reasoner.predict_proba(s.concepts_pred)
    data = {
        "row": np.arange(s.n_samples),
        "target": s.y.astype(int),
        "target_prob": probs,
    }
    for j, name in enumerate(names):
        data[f"concept_prob__{name}"] = s.concepts_pred[:, j]
        data[f"concept_label__{name}"] = s.concepts_true[:, j].astype(int)
        data[f"concept_mask__{name}"] = np.ones(s.n_samples, dtype=int)
    for pert_name, matrix in (perturbations or {}).items():
        for j, name in enumerate(names):
            data[f"concept_prob__{name}__{pert_name}"] = matrix[:, j]
    return pd.DataFrame(data)


def _weights(s) -> dict:
    return {"weights": s.weights.tolist(), "bias": float(s.bias), "concept_names": s.concept_names}


@pytest.fixture(scope="module")
def clean():
    return make_synthetic_cbm(
        n_patients=400, studies_per_patient=1, n_concepts=8, concept_noise=0.4, seed=0
    )


@pytest.fixture(scope="module")
def leaky():
    return make_synthetic_cbm(
        n_patients=400,
        studies_per_patient=1,
        n_concepts=8,
        concept_noise=0.4,
        soft_leak_strength=3.0,
        seed=1,
    )


class TestAssembly:
    def test_all_eight_families_present_with_perturbations_and_baseline(self, clean):
        reasoner = LinearReasoner(clean.weights, clean.bias, clean.concept_names)
        rng = np.random.default_rng(0)
        perts = {
            k: np.clip(clean.concepts_pred + rng.normal(0, 0.05, clean.concepts_pred.shape), 0, 1)
            for k in ("rotate", "brightness", "noise")
        }
        baseline = _frame(
            make_synthetic_cbm(n_patients=400, studies_per_patient=1, n_concepts=8, seed=9),
            reasoner=reasoner,
        )
        metrics = run_protocol(
            _frame(clean, reasoner=reasoner, perturbations=perts),
            _weights(clean),
            features=clean.features,
            baseline_predictions=baseline,
        )
        for fam in FAMILIES:
            assert fam in metrics
            assert metrics[fam].get("status") != "pending", fam
        assert not metrics["completeness"].get("partial_curve")
        assert set(metrics["intervention"]) == {"random", "uncertainty", "weight", "oracle"}
        assert "aggregate" in metrics["stability"]

    def test_stability_pending_without_perturbed_columns(self, clean):
        reasoner = LinearReasoner(clean.weights, clean.bias, clean.concept_names)
        metrics = run_protocol(_frame(clean, reasoner=reasoner), _weights(clean))
        assert metrics["stability"]["status"] == "pending"
        assert metrics["completeness"]["partial_curve"] is True

    def test_faithfulness_is_exact_for_the_linear_reasoner(self, clean):
        reasoner = LinearReasoner(clean.weights, clean.bias, clean.concept_names)
        metrics = run_protocol(_frame(clean, reasoner=reasoner), _weights(clean))
        assert metrics["faithfulness"]["correlation"] > 0.999
        assert metrics["faithfulness"]["max_abs_discrepancy"] < 1e-2


class TestFindings:
    def test_soft_vs_hard_leakage_flags_the_soft_leak(self, leaky, clean):
        rc = LinearReasoner(clean.weights, clean.bias, clean.concept_names)
        rl = LinearReasoner(leaky.weights, leaky.bias, leaky.concept_names)
        honest = run_protocol(_frame(clean, reasoner=rc), _weights(clean), features=clean.features)
        leaked = run_protocol(_frame(leaky, reasoner=rl), _weights(leaky), features=leaky.features)
        # Hardening the concepts costs more AUROC when target info was smuggled into the
        # soft values (the definition of leakage).
        assert (
            leaked["leakage"]["soft_vs_hard"]["leakage"]
            > honest["leakage"]["soft_vs_hard"]["leakage"] + 0.02
        )
        assert "residual_r2" in leaked["leakage"]["residual_probe"]

    def test_decorative_concept_is_named(self, clean):
        # make_synthetic_cbm sets the last concept's weight to ~0 by construction.
        reasoner = LinearReasoner(clean.weights, clean.bias, clean.concept_names)
        metrics = run_protocol(_frame(clean, reasoner=reasoner), _weights(clean))
        assert clean.concept_names[-1] in metrics["leakage"]["necessity"]["decorative_concepts"]
