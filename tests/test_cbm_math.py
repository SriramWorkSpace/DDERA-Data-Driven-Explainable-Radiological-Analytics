"""The CBM mathematics.

These are the correctness-critical tests in the project. If the contribution decomposition
or the intervention arithmetic is wrong, the dashboard displays an explanation that does not
describe the prediction -- which is precisely the failure mode DDERA exists to avoid.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddera.xai.intervention import (
    LinearReasoner,
    apply_intervention_order,
    empirical_sensitivity,
    expected_logit_shift,
    faithfulness_report,
    intervene,
    intervention_effect,
    intervention_order,
    logit,
    sigmoid,
    tti_all_strategies,
    tti_curve,
    verify_decomposition,
)


class TestSigmoidLogit:
    def test_roundtrip(self):
        p = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
        assert np.allclose(sigmoid(logit(p)), p, atol=1e-12)

    def test_logit_of_half_is_zero(self):
        assert abs(float(logit(0.5))) < 1e-12

    def test_extreme_values_stay_finite(self):
        """Saturation must not produce inf/nan: an intervention can push a logit far out."""
        assert np.all(np.isfinite(sigmoid([-1e6, 1e6])))
        assert np.all(np.isfinite(logit([0.0, 1.0])))


class TestDecomposition:
    """logit(p) == bias + sum_j w_j c_j, exactly."""

    def test_contributions_sum_to_logit(self, reasoner, synth):
        error = verify_decomposition(reasoner, synth.concepts_pred, atol=1e-9)
        assert error < 1e-9, f"Decomposition error {error:.3e}"

    def test_decomposition_holds_for_extreme_concepts(self, reasoner):
        """All-zeros and all-ones are the boundary cases the dashboard sliders can reach."""
        k = reasoner.n_concepts
        for concepts in (np.zeros((5, k)), np.ones((5, k)), np.full((5, k), 0.5)):
            assert verify_decomposition(reasoner, concepts, atol=1e-9) < 1e-9

    def test_all_zero_concepts_give_bias_only(self, reasoner):
        """With every concept at zero the logit must be exactly the bias."""
        got = reasoner.predict_logit(np.zeros((1, reasoner.n_concepts)))
        assert np.allclose(got, reasoner.bias, atol=1e-12)

    def test_contribution_of_zero_concept_is_zero(self, reasoner):
        contributions = reasoner.contributions(np.zeros((3, reasoner.n_concepts)))
        assert np.allclose(contributions, 0.0)

    def test_broken_decomposition_is_detected(self, synth):
        """The guard must actually fire -- a test that can never fail is worthless."""

        class Sabotaged(LinearReasoner):
            def contributions(self, concepts):
                return super().contributions(concepts) + 0.1

        bad = Sabotaged(synth.weights, synth.bias)
        with pytest.raises(AssertionError, match="decomposition is broken"):
            verify_decomposition(bad, synth.concepts_pred)


class TestIntervention:
    """Setting c_j := v must shift the logit by exactly w_j * (v - c_j)."""

    def test_intervention_does_not_mutate_input(self, synth):
        original = synth.concepts_pred.copy()
        intervene(synth.concepts_pred, 0, 1.0)
        assert np.array_equal(synth.concepts_pred, original)

    def test_logit_shift_matches_closed_form(self, reasoner, predict_fn, synth):
        concepts = synth.concepts_pred[:200]
        for j in (0, 1, 3, reasoner.n_concepts - 1):
            for new_value in (0.0, 0.2, 0.8, 1.0):
                effect = intervention_effect(predict_fn, concepts, j, new_value)
                expected = expected_logit_shift(reasoner.weights, j, concepts[:, j], new_value)
                assert np.allclose(effect["delta_logit"], expected, atol=1e-8), (
                    f"concept {j} -> {new_value}: realised shift does not match w_j*(v-c_j)"
                )

    def test_zero_weight_concept_has_no_effect(self, reasoner, predict_fn, synth):
        """The decorative concept must not move the prediction, whatever it is set to."""
        j = int(np.argmin(np.abs(reasoner.weights)))
        effect = intervention_effect(predict_fn, synth.concepts_pred[:100], j, 1.0)
        assert abs(effect["mean_delta_p"]) < 0.01

    def test_intervention_direction_follows_weight_sign(self, reasoner, predict_fn, synth):
        """Raising a positively-weighted concept must raise the prediction, and vice versa."""
        concepts = np.full((50, reasoner.n_concepts), 0.5)
        for j in range(reasoner.n_concepts):
            if abs(reasoner.weights[j]) < 1e-3:
                continue
            up = intervention_effect(predict_fn, concepts, j, 1.0)["mean_delta_p"]
            assert np.sign(up) == np.sign(reasoner.weights[j]), (
                f"concept {j} (w={reasoner.weights[j]:.3f}) moved the wrong way"
            )

    def test_intervening_to_current_value_is_a_no_op(self, predict_fn, synth):
        concepts = synth.concepts_pred[:50]
        j = 2
        after = intervene(concepts, j, concepts[:, j])
        assert np.allclose(predict_fn(after), predict_fn(concepts), atol=1e-12)

    def test_out_of_range_index_raises(self, synth):
        with pytest.raises(IndexError):
            intervene(synth.concepts_pred, 999, 1.0)


class TestInterventionOrdering:
    def test_orders_are_permutations(self, synth):
        for strategy in ("random", "uncertainty", "weight", "oracle"):
            order = intervention_order(
                synth.concepts_pred,
                strategy,
                weights=synth.weights,
                concepts_true=synth.concepts_true,
                seed=0,
            )
            assert order.shape == synth.concepts_pred.shape
            for row in order[:20]:
                assert sorted(row.tolist()) == list(range(synth.n_concepts))

    def test_uncertainty_order_puts_most_uncertain_first(self, synth):
        order = intervention_order(synth.concepts_pred, "uncertainty")
        distances = np.abs(synth.concepts_pred - 0.5)
        for i in range(50):
            ordered = distances[i, order[i]]
            assert np.all(np.diff(ordered) >= -1e-12), "not ascending in |c - 0.5|"

    def test_weight_order_is_by_absolute_weight(self, synth):
        order = intervention_order(synth.concepts_pred, "weight", weights=synth.weights)
        magnitudes = np.abs(synth.weights)[order[0]]
        assert np.all(np.diff(magnitudes) <= 1e-12), "not descending in |w|"

    def test_oracle_order_targets_worst_errors(self, synth):
        order = intervention_order(synth.concepts_pred, "oracle", concepts_true=synth.concepts_true)
        errors = np.abs(synth.concepts_pred - synth.concepts_true)
        for i in range(50):
            ordered = errors[i, order[i]]
            assert np.all(np.diff(ordered) <= 1e-12), "not descending in concept error"

    def test_missing_required_argument_raises(self, synth):
        with pytest.raises(ValueError, match="requires"):
            intervention_order(synth.concepts_pred, "weight")
        with pytest.raises(ValueError, match="requires"):
            intervention_order(synth.concepts_pred, "oracle")

    def test_apply_order_replaces_exactly_m_concepts(self, synth):
        order = intervention_order(synth.concepts_pred, "random", seed=0)
        for m in range(synth.n_concepts + 1):
            mixed = apply_intervention_order(synth.concepts_pred, synth.concepts_true, order, m)
            replaced = (mixed == synth.concepts_true).sum(axis=1)
            assert np.all(replaced >= m), f"fewer than {m} concepts replaced"

    def test_full_intervention_recovers_ground_truth(self, synth):
        order = intervention_order(synth.concepts_pred, "random", seed=0)
        mixed = apply_intervention_order(
            synth.concepts_pred, synth.concepts_true, order, synth.n_concepts
        )
        assert np.array_equal(mixed, synth.concepts_true)


class TestTTICurve:
    def test_curve_shape_and_endpoints(self, predict_fn, synth):
        curve = tti_curve(
            predict_fn,
            synth.concepts_pred,
            synth.concepts_true,
            synth.y,
            strategy="random",
            n_repeats=3,
        )
        assert len(curve.auroc) == synth.n_concepts + 1
        assert curve.baseline_auroc == pytest.approx(curve.auroc[0])
        assert curve.full_intervention_auroc == pytest.approx(curve.auroc[-1])

    def test_correcting_concepts_improves_auroc(self, predict_fn, synth):
        """The central intervention claim: fixing concepts must help a concept-driven model."""
        curve = tti_curve(
            predict_fn,
            synth.concepts_pred,
            synth.concepts_true,
            synth.y,
            strategy="random",
            n_repeats=5,
        )
        assert curve.total_gain > 0.01, (
            f"Intervention gained only {curve.total_gain:.4f} AUROC; the model does not "
            "appear to depend on its concepts."
        )

    def test_curve_is_flat_when_concepts_are_already_correct(self, synth):
        """Nothing to repair means no gain. Guards against a curve that always rises."""
        r = LinearReasoner(synth.weights, synth.bias)
        curve = tti_curve(
            r.as_predict_fn(),
            synth.concepts_true,
            synth.concepts_true,
            synth.y,
            strategy="random",
            n_repeats=2,
        )
        assert abs(curve.total_gain) < 1e-9

    def test_oracle_ordering_beats_random_midway(self, predict_fn, synth):
        """Fixing the worst-estimated concepts first must pay off faster than random."""
        curves = tti_all_strategies(
            predict_fn,
            synth.concepts_pred,
            synth.concepts_true,
            synth.y,
            synth.weights,
            n_repeats=5,
        )
        mid = synth.n_concepts // 2
        assert curves["oracle"].auroc[mid] >= curves["random"].auroc[mid] - 1e-9

    def test_all_strategies_share_endpoints(self, predict_fn, synth):
        """m=0 and m=k are ordering-independent by construction."""
        curves = tti_all_strategies(
            predict_fn,
            synth.concepts_pred,
            synth.concepts_true,
            synth.y,
            synth.weights,
            n_repeats=3,
        )
        baselines = [c.baseline_auroc for c in curves.values()]
        finals = [c.full_intervention_auroc for c in curves.values()]
        assert max(baselines) - min(baselines) < 1e-9
        assert max(finals) - min(finals) < 1e-9


class TestFaithfulness:
    """Does the model behave the way its weights claim?"""

    def test_empirical_sensitivity_recovers_weights(self, predict_fn, synth):
        """For a linear reasoner, d logit / d c_j must equal w_j."""
        sensitivity = empirical_sensitivity(predict_fn, synth.concepts_pred[:200], eps=1e-3)
        assert np.allclose(sensitivity.mean(axis=0), synth.weights, atol=1e-4)

    def test_sensitivity_is_constant_across_samples(self, predict_fn, synth):
        """A linear reasoner has the same derivative everywhere; that is why it is readable."""
        sensitivity = empirical_sensitivity(predict_fn, synth.concepts_pred[:100], eps=1e-3)
        assert np.all(sensitivity.std(axis=0) < 1e-4)

    def test_report_shows_perfect_faithfulness_for_linear_model(self, predict_fn, synth):
        report = faithfulness_report(
            predict_fn,
            synth.concepts_pred[:200],
            synth.weights,
            concept_names=synth.concept_names,
        )
        assert report["correlation"] > 0.999
        assert report["sign_agreement"] == 1.0
        assert report["max_abs_discrepancy"] < 1e-3

    def test_report_detects_an_unfaithful_model(self, synth):
        """A model whose declared weights are not the weights it uses must be caught."""
        honest = LinearReasoner(synth.weights, synth.bias)
        declared = -synth.weights  # claims the opposite of what it does
        report = faithfulness_report(honest.as_predict_fn(), synth.concepts_pred[:200], declared)
        assert report["correlation"] < -0.99
        assert report["sign_agreement"] == 0.0

    def test_boundary_concepts_do_not_break_the_derivative(self, predict_fn, synth):
        """Concepts pinned at 0 or 1 must still yield the correct one-sided estimate."""
        concepts = np.zeros((20, synth.n_concepts))
        concepts[:, ::2] = 1.0
        sensitivity = empirical_sensitivity(predict_fn, concepts, eps=1e-3)
        assert np.allclose(sensitivity.mean(axis=0), synth.weights, atol=1e-4)
        assert np.all(np.isfinite(sensitivity))
