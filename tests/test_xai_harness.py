"""Leakage, completeness, stability and the post-hoc guard.

The recurring pattern here is **discrimination**, not smoke-testing: each measure is given a
case where it should fire and a case where it should not, and asserted to tell them apart.
A leakage detector that returns a plausible number on every input is not measuring leakage.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddera.data.synthetic import make_synthetic_cbm
from ddera.xai.completeness import (
    completeness_curve,
    completeness_ratio,
    completeness_report,
)
from ddera.xai.intervention import LinearReasoner
from ddera.xai.leakage import (
    concept_permutation_necessity,
    leakage_report,
    residual_probe_leakage,
    residual_working_response,
    soft_vs_hard_leakage,
)
from ddera.xai.posthoc import (
    BASELINE_ONLY_MESSAGE,
    assert_baseline_only,
    comparison_table,
)
from ddera.xai.stability import (
    concept_drift,
    prediction_flip_rate,
    rank_stability,
    stability_report,
)


class TestConceptNecessity:
    def test_decorative_concept_is_identified(self, predict_fn, synth):
        """The synthetic reasoner has one concept with weight ~0.01. It must be flagged."""
        result = concept_permutation_necessity(
            predict_fn,
            synth.concepts_pred,
            synth.y,
            concept_names=synth.concept_names,
            n_repeats=5,
            seed=0,
        )
        decorative = synth.concept_names[synth.meta["decorative_concept_index"]]
        assert decorative in result["decorative_concepts"]

    def test_important_concept_ranks_above_decorative(self, predict_fn, synth):
        result = concept_permutation_necessity(
            predict_fn,
            synth.concepts_pred,
            synth.y,
            concept_names=synth.concept_names,
            n_repeats=5,
            seed=0,
        )
        drops = {n: v["auroc_drop_mean"] for n, v in result["per_concept"].items()}
        strongest = synth.concept_names[int(np.argmax(np.abs(synth.weights)))]
        decorative = synth.concept_names[synth.meta["decorative_concept_index"]]
        assert drops[strongest] > drops[decorative]

    def test_necessity_correlates_with_absolute_weight(self, predict_fn, synth):
        """Concepts the model weights heavily should be the ones it cannot do without."""
        result = concept_permutation_necessity(
            predict_fn,
            synth.concepts_pred,
            synth.y,
            concept_names=synth.concept_names,
            n_repeats=8,
            seed=0,
        )
        drops = np.array([result["per_concept"][n]["auroc_drop_mean"] for n in synth.concept_names])
        assert np.corrcoef(drops, np.abs(synth.weights))[0, 1] > 0.5

    def test_ranking_is_ordered_by_drop(self, predict_fn, synth):
        result = concept_permutation_necessity(
            predict_fn,
            synth.concepts_pred,
            synth.y,
            concept_names=synth.concept_names,
            n_repeats=3,
            seed=0,
        )
        drops = [result["per_concept"][n]["auroc_drop_mean"] for n in result["ranking"]]
        assert drops == sorted(drops, reverse=True)

    def test_name_count_mismatch_raises(self, predict_fn, synth):
        with pytest.raises(ValueError, match="names"):
            concept_permutation_necessity(
                predict_fn, synth.concepts_pred, synth.y, concept_names=["a", "b"]
            )


class TestSoftVsHardLeakage:
    def test_binary_concepts_show_no_leakage(self, synth):
        """If concepts are already 0/1, hardening is a no-op and leakage must be exactly 0."""
        r = LinearReasoner(synth.weights, synth.bias)
        result = soft_vs_hard_leakage(r.as_predict_fn(), synth.concepts_true, synth.y)
        assert result["leakage"] == pytest.approx(0.0, abs=1e-12)

    def test_detects_genuine_soft_leakage(self):
        """The positive direction.

        Real CBM leakage needs a reasoner *trained on soft concepts* that learns to exploit
        their sub-symbolic precision. Here target information is injected into the concept
        probabilities while leaving their hard thresholds essentially intact, and a logistic
        reasoner is fitted on those soft values. Hardening destroys the smuggled channel, so
        leakage must come out positive.
        """
        from sklearn.linear_model import LogisticRegression

        leaky = make_synthetic_cbm(
            n_patients=600,
            studies_per_patient=2,
            n_concepts=8,
            concept_noise=0.3,
            concept_separability=3.0,
            soft_leak_strength=3.0,
            seed=21,
        )
        # The bottleneck still looks symbolically sound...
        assert leaky.meta["hard_concept_error_rate"] < 0.10

        fitted = LogisticRegression(max_iter=1000).fit(leaky.concepts_pred, leaky.y)
        result = soft_vs_hard_leakage(
            lambda c: fitted.predict_proba(c)[:, 1], leaky.concepts_pred, leaky.y
        )

        assert result["leakage"] > 0.02, (
            "failed to detect a reasoner exploiting sub-symbolic concept precision"
        )
        assert "sub-symbolic" in result["interpretation"]

    def test_hardening_noisy_concepts_denoises_them(self, predict_fn, synth):
        """The negative direction, which is a different finding and must not be conflated.

        When the reasoner was built for binary concepts and the concept predictor is merely
        noisy, thresholding moves the values back toward the truth. AUROC then *improves*
        and leakage is negative -- a concept-calibration signal, not smuggled information.
        """
        result = soft_vs_hard_leakage(predict_fn, synth.concepts_pred, synth.y)
        assert result["leakage"] < 0.0
        assert "calibration" in result["interpretation"]

    def test_confident_predictor_is_near_neutral(self, synth_clean):
        """Concepts already near 0/1 barely move when hardened, either way."""
        reasoner = LinearReasoner(synth_clean.weights, synth_clean.bias)
        result = soft_vs_hard_leakage(
            reasoner.as_predict_fn(), synth_clean.concepts_pred, synth_clean.y
        )
        assert abs(result["leakage"]) < 0.01
        assert "negligible" in result["interpretation"]

    def test_interpretation_string_is_present(self, predict_fn, synth):
        assert isinstance(
            soft_vs_hard_leakage(predict_fn, synth.concepts_pred, synth.y)["interpretation"],
            str,
        )


class TestResidualProbe:
    def test_detects_information_bypassing_the_bottleneck(self, synth, synth_leaky):
        """The discriminative test: leaky features must probe higher than clean ones."""
        clean_r = LinearReasoner(synth.weights, synth.bias)
        clean = residual_probe_leakage(
            synth.features, synth.y, clean_r.predict_proba(synth.concepts_pred), cv=3, seed=0
        )
        leaky_r = LinearReasoner(synth_leaky.weights, synth_leaky.bias)
        leaky = residual_probe_leakage(
            synth_leaky.features,
            synth_leaky.y,
            leaky_r.predict_proba(synth_leaky.concepts_pred),
            cv=3,
            seed=0,
        )
        assert leaky["delta_auroc"] > clean["delta_auroc"], (
            "the probe cannot distinguish leaked information from none"
        )

    def test_reports_expected_fields(self, synth):
        r = LinearReasoner(synth.weights, synth.bias)
        result = residual_probe_leakage(
            synth.features, synth.y, r.predict_proba(synth.concepts_pred), cv=3, seed=0
        )
        for key in ("auroc_bottleneck", "auroc_feature_probe", "delta_auroc", "residual_r2"):
            assert key in result
        assert result["n_features"] == synth.features.shape[1]

    def test_row_mismatch_raises(self, synth):
        with pytest.raises(ValueError, match="Row mismatch"):
            residual_probe_leakage(synth.features, synth.y[:10], synth.y)

    def test_working_response_is_finite_and_signed(self):
        """(y - p) / (p(1-p)): positive when under-predicting, negative when over."""
        response = residual_working_response(np.array([1, 0]), np.array([0.2, 0.8]))
        assert np.all(np.isfinite(response))
        assert response[0] > 0 and response[1] < 0

    def test_working_response_handles_saturated_probabilities(self):
        assert np.all(np.isfinite(residual_working_response([1, 0], [0.0, 1.0])))


class TestLeakageReport:
    def test_report_assembles_the_family(self, predict_fn, synth):
        report = leakage_report(
            predict_fn,
            synth.concepts_pred,
            synth.y,
            features=synth.features,
            concept_names=synth.concept_names,
            n_repeats=3,
            seed=0,
        )
        assert set(report) == {"necessity", "soft_vs_hard", "residual_probe"}

    def test_features_are_optional(self, predict_fn, synth):
        report = leakage_report(
            predict_fn,
            synth.concepts_pred,
            synth.y,
            concept_names=synth.concept_names,
            n_repeats=2,
        )
        assert "residual_probe" not in report


class TestCompleteness:
    def test_ratio_is_skill_above_chance(self):
        """(0.8 - 0.5) / (0.9 - 0.5) = 0.75, not 0.8/0.9."""
        assert completeness_ratio(0.8, 0.9) == pytest.approx(0.75)

    def test_ratio_is_one_when_bottleneck_matches_reference(self):
        assert completeness_ratio(0.85, 0.85) == pytest.approx(1.0)

    def test_ratio_is_zero_at_chance(self):
        assert completeness_ratio(0.5, 0.9) == pytest.approx(0.0)

    def test_ratio_can_exceed_one(self):
        """Legitimate: the bottleneck sometimes regularises better than the black box."""
        assert completeness_ratio(0.92, 0.90) > 1.0

    def test_ratio_is_nan_for_a_chance_reference(self):
        assert np.isnan(completeness_ratio(0.8, 0.5))

    def test_curve_requires_k_zero(self):
        with pytest.raises(ValueError, match="k=0"):
            completeness_curve({4: 0.8, 8: 0.85})

    def test_curve_identifies_bottleneck_and_saturation(self):
        curve = completeness_curve({0: 0.80, 4: 0.84, 16: 0.87, 64: 0.88})
        assert curve.bottleneck_auroc == 0.80
        assert curve.saturated_auroc == 0.88
        assert curve.interpretability_cost == pytest.approx(0.08)

    def test_report_computes_marginal_gains(self):
        report = completeness_report({0: 0.80, 4: 0.84, 16: 0.87}, reference_auroc=0.90)
        assert report["marginal_gain_by_k"][4] == pytest.approx(0.04)
        assert report["marginal_gain_by_k"][16] == pytest.approx(0.07)
        assert report["completeness_ratio"] == pytest.approx((0.80 - 0.5) / (0.90 - 0.5))

    def test_saturation_point_found(self):
        """A curve that plateaus early means the concepts were nearly sufficient."""
        report = completeness_report({0: 0.80, 4: 0.899, 16: 0.90, 64: 0.90})
        assert report["saturation_k"] == 4

    def test_complete_bottleneck_saturates_at_zero(self):
        report = completeness_report({0: 0.90, 4: 0.90, 16: 0.90})
        assert report["saturation_k"] == 0
        assert report["interpretability_cost"] == pytest.approx(0.0)

    def test_empty_sweep_raises(self):
        with pytest.raises(ValueError, match="empty"):
            completeness_curve({})


class TestStability:
    def test_identical_inputs_show_zero_drift(self, synth):
        drift = concept_drift(synth.concepts_pred, synth.concepts_pred)
        assert drift["l1_mean"] == 0.0
        assert drift["linf_max"] == 0.0

    def test_drift_scales_with_perturbation_size(self, synth):
        rng = np.random.default_rng(0)
        base = synth.concepts_pred
        small = np.clip(base + rng.normal(0, 0.01, base.shape), 0, 1)
        large = np.clip(base + rng.normal(0, 0.20, base.shape), 0, 1)
        assert concept_drift(base, small)["l1_mean"] < concept_drift(base, large)["l1_mean"]

    def test_rank_stability_is_one_for_identical_inputs(self, synth):
        result = rank_stability(synth.concepts_pred, synth.concepts_pred)
        assert result["spearman_mean"] == pytest.approx(1.0)

    def test_rank_stability_degrades_under_heavy_noise(self, synth):
        rng = np.random.default_rng(1)
        shuffled = rng.permuted(synth.concepts_pred, axis=1)
        assert rank_stability(synth.concepts_pred, shuffled)["spearman_mean"] < 0.5

    def test_rank_stability_excludes_constant_rows(self):
        """Rows with no concept variance cannot be ranked and must be reported, not faked."""
        base = np.tile(np.array([0.5, 0.5, 0.5, 0.5]), (10, 1))
        result = rank_stability(base, base)
        assert result["n_valid"] == 0
        assert result["n_excluded"] == 10
        assert np.isnan(result["spearman_mean"])

    def test_flip_rate_is_zero_without_change(self, predict_fn, synth):
        probs = predict_fn(synth.concepts_pred)
        assert prediction_flip_rate(probs, probs)["flip_rate"] == 0.0

    def test_flip_rate_detects_crossing_the_threshold(self):
        result = prediction_flip_rate(np.array([0.4, 0.6]), np.array([0.6, 0.4]))
        assert result["flip_rate"] == 1.0
        assert result["n_flipped"] == 2

    def test_report_aggregates_worst_case_not_mean(self, predict_fn, synth):
        """One unstable perturbation must not be averaged away by three stable ones."""
        rng = np.random.default_rng(2)
        base = synth.concepts_pred
        stable = np.clip(base + rng.normal(0, 0.001, base.shape), 0, 1)
        unstable = rng.permuted(base, axis=1)

        report = stability_report(
            base,
            [stable, stable, unstable],
            predict_fn=predict_fn,
            perturbation_names=["a", "b", "shuffled"],
        )
        worst = report["aggregate"]["worst_rank_stability"]
        per = report["per_perturbation"]
        assert worst == pytest.approx(per["shuffled"]["rank"]["spearman_mean"])
        assert worst < per["a"]["rank"]["spearman_mean"]

    def test_report_flags_instability(self, predict_fn, synth):
        rng = np.random.default_rng(3)
        report = stability_report(
            synth.concepts_pred,
            [rng.permuted(synth.concepts_pred, axis=1)],
            predict_fn=predict_fn,
        )
        assert "unstable" in report["aggregate"]["interpretation"]

    def test_report_confirms_stability(self, predict_fn, synth):
        rng = np.random.default_rng(4)
        tiny = np.clip(synth.concepts_pred + rng.normal(0, 0.0005, synth.concepts_pred.shape), 0, 1)
        report = stability_report(synth.concepts_pred, [tiny], predict_fn=predict_fn)
        assert "stable" in report["aggregate"]["interpretation"]

    def test_name_count_mismatch_raises(self, synth):
        with pytest.raises(ValueError, match="names"):
            stability_report(
                synth.concepts_pred, [synth.concepts_pred], perturbation_names=["a", "b"]
            )


class TestPostHocGuard:
    """Invariant 5 must be enforced in code, not merely documented."""

    def test_ante_hoc_model_is_refused(self):
        class FakeCBM:
            is_ante_hoc = True

        with pytest.raises(RuntimeError, match="Invariant 5 violation"):
            assert_baseline_only(FakeCBM())

    def test_black_box_model_is_allowed(self):
        class FakeBlackBox:
            is_ante_hoc = False

        assert_baseline_only(FakeBlackBox())
        assert_baseline_only(object())

    def test_error_message_points_to_the_right_alternative(self):
        assert "contributions()" in BASELINE_ONLY_MESSAGE
        assert "black-box" in BASELINE_ONLY_MESSAGE

    def test_comparison_table_contrasts_the_two_families(self):
        table = comparison_table()
        assert table["post_hoc_saliency"]["interveneable"] is False
        assert table["ddera_concept_bottleneck"]["interveneable"] is True
        assert table["post_hoc_saliency"]["exact"] is False
        assert table["ddera_concept_bottleneck"]["exact"] is True


class TestDomainAgnosticism:
    """Invariant 1/9: the harness must not assume chest X-ray, or any particular domain."""

    def test_harness_runs_unchanged_on_a_different_domain_shape(self):
        """A 22-concept, 6-target domain (VinDr's shape) must need no code change."""
        other = make_synthetic_cbm(
            n_patients=150,
            studies_per_patient=1,
            n_concepts=22,
            n_features=32,
            concept_noise=0.7,
            seed=11,
        )
        reasoner = LinearReasoner(other.weights, other.bias, other.concept_names)
        fn = reasoner.as_predict_fn()

        necessity = concept_permutation_necessity(
            fn,
            other.concepts_pred,
            other.y,
            concept_names=other.concept_names,
            n_repeats=2,
            seed=0,
        )
        leakage = soft_vs_hard_leakage(fn, other.concepts_pred, other.y)
        stability = stability_report(other.concepts_pred, [other.concepts_pred], predict_fn=fn)

        assert len(necessity["per_concept"]) == 22
        assert np.isfinite(leakage["leakage"])
        assert stability["aggregate"]["worst_l1_drift"] == 0.0
