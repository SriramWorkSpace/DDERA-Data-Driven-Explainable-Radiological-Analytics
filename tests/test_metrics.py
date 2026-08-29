"""Metrics, calibration and bootstrap.

Where possible these check against **hand-computable** values rather than against another
implementation, so a shared misunderstanding cannot pass silently.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ddera.eval.bootstrap import bootstrap_ci, is_significant, paired_bootstrap_diff
from ddera.eval.calibration import (
    apply_temperature,
    brier_score,
    calibration_report,
    expected_calibration_error,
    fit_temperature,
    maximum_calibration_error,
    reliability_curve,
)
from ddera.eval.metrics import (
    binary_metrics,
    concept_metrics,
    prediction_agreement,
    safe_auprc,
    safe_auroc,
)


class TestBinaryMetrics:
    def test_perfect_separation(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.8, 0.9])
        m = binary_metrics(y, s)
        assert m["auroc"] == 1.0
        assert m["accuracy"] == 1.0
        assert m["sensitivity"] == 1.0
        assert m["specificity"] == 1.0

    def test_confusion_matrix_cells_by_hand(self):
        """y = [1,1,0,0], predictions at 0.5 -> [1,0,1,0]: tp=1 fn=1 fp=1 tn=1."""
        m = binary_metrics(np.array([1, 1, 0, 0]), np.array([0.9, 0.1, 0.9, 0.1]))
        assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
        assert m["precision"] == 0.5
        assert m["recall"] == 0.5
        assert m["specificity"] == 0.5
        assert m["f1"] == pytest.approx(0.5)

    def test_f1_is_harmonic_mean_of_precision_and_recall(self):
        m = binary_metrics(np.array([1, 1, 1, 0, 0, 0]), np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1]))
        expected = 2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"])
        assert m["f1"] == pytest.approx(expected)

    def test_sensitivity_equals_recall(self, synth):
        m = binary_metrics(synth.y, np.random.default_rng(0).random(synth.n_samples))
        assert m["sensitivity"] == m["recall"]

    def test_threshold_is_respected(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.3, 0.4, 0.6, 0.7])
        assert binary_metrics(y, s, threshold=0.5)["accuracy"] == 1.0
        assert binary_metrics(y, s, threshold=0.95)["accuracy"] == 0.5

    def test_agrees_with_sklearn_auroc(self, synth):
        scores = np.random.default_rng(1).random(synth.n_samples)
        assert safe_auroc(synth.y, scores) == pytest.approx(roc_auc_score(synth.y, scores))

    def test_auprc_baseline_is_the_positive_rate(self):
        """For a random scorer, average precision tends to prevalence -- unlike AUROC's 0.5."""
        rng = np.random.default_rng(0)
        y = (rng.random(20000) < 0.1).astype(int)
        assert safe_auprc(y, rng.random(20000)) == pytest.approx(0.1, abs=0.02)

    def test_single_class_returns_nan_not_an_exception(self):
        """A rare concept with no positives must not abort a whole sweep."""
        rng = np.random.default_rng(0)
        assert np.isnan(safe_auroc(np.zeros(10), rng.random(10)))
        assert np.isnan(safe_auprc(np.ones(10), rng.random(10)))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            binary_metrics(np.array([0, 1]), np.array([0.5]))

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            binary_metrics(np.array([]), np.array([]))


class TestConceptMetrics:
    def test_reports_every_concept(self, synth):
        result = concept_metrics(synth.concepts_true, synth.concepts_pred, synth.concept_names)
        assert set(result["per_concept"]) == set(synth.concept_names)
        assert result["macro"]["n_concepts_total"] == synth.n_concepts

    def test_good_predictor_scores_well(self, synth_clean):
        result = concept_metrics(
            synth_clean.concepts_true, synth_clean.concepts_pred, synth_clean.concept_names
        )
        assert result["macro"]["macro_auroc"] > 0.95

    def test_mask_excludes_entries_from_scoring(self, synth):
        """This is what makes the u_mask policy evaluable rather than merely trainable."""
        mask = np.ones_like(synth.concepts_true)
        mask[: synth.n_samples // 2, 0] = 0.0
        result = concept_metrics(
            synth.concepts_true, synth.concepts_pred, synth.concept_names, mask=mask
        )
        assert result["coverage"][synth.concept_names[0]] == pytest.approx(0.5, abs=0.01)
        assert result["per_concept"][synth.concept_names[0]]["n"] == synth.n_samples // 2

    def test_fully_masked_concept_yields_nan_not_a_crash(self, synth):
        mask = np.ones_like(synth.concepts_true)
        mask[:, 0] = 0.0
        result = concept_metrics(
            synth.concepts_true, synth.concepts_pred, synth.concept_names, mask=mask
        )
        assert np.isnan(result["per_concept"][synth.concept_names[0]]["auroc"])
        assert result["macro"]["n_concepts_scored"] == synth.n_concepts - 1

    def test_macro_ignores_nan_concepts(self, synth):
        mask = np.ones_like(synth.concepts_true)
        mask[:, 0] = 0.0
        result = concept_metrics(
            synth.concepts_true, synth.concepts_pred, synth.concept_names, mask=mask
        )
        assert not np.isnan(result["macro"]["macro_auroc"])

    def test_name_count_mismatch_raises(self, synth):
        with pytest.raises(ValueError, match="names"):
            concept_metrics(synth.concepts_true, synth.concepts_pred, ["only_one"])


class TestCalibration:
    def test_ece_hand_computed(self):
        """y=[0,0,1,1], p=[.1,.1,.9,.9], 10 uniform bins.

        Bin [0.0,0.1]: conf 0.1, acc 0.0, gap 0.1, n=2
        Bin (0.8,0.9]: conf 0.9, acc 1.0, gap 0.1, n=2
        ECE = 0.5*0.1 + 0.5*0.1 = 0.1
        """
        ece = expected_calibration_error(
            np.array([0, 0, 1, 1]), np.array([0.1, 0.1, 0.9, 0.9]), n_bins=10
        )
        assert ece == pytest.approx(0.1, abs=1e-9)

    def test_perfectly_calibrated_scores_near_zero(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.05, 0.95, 40000)
        y = (rng.random(40000) < p).astype(int)
        assert expected_calibration_error(y, p, n_bins=10) < 0.02

    def test_overconfident_model_ece_hand_computed(self):
        """Confidently wrong on many negatives.

        y alternates 0/1 over 1000 samples; p = 0.98 for indices 0-399, then alternating
        0.02/0.98. Working it through:

        - bin (0.9, 1.0]: 400 + 300 = 700 samples, of which 200 + 300 = 500 are positive.
          conf 0.98, acc 5/7, gap 0.2657, weight 0.70  ->  0.1860
        - bin [0.0, 0.1]:  300 samples, all negative.
          conf 0.02, acc 0.00, gap 0.0200, weight 0.30  ->  0.0060

        ECE = 0.192
        """
        y = np.array([0, 1] * 500)
        p = np.array([0.02, 0.98] * 500)
        p[:400] = 0.98
        assert expected_calibration_error(y, p) == pytest.approx(0.192, abs=1e-9)

    def test_mce_is_at_least_ece(self):
        rng = np.random.default_rng(2)
        p = rng.uniform(0, 1, 2000)
        y = (rng.random(2000) < p).astype(int)
        assert maximum_calibration_error(y, p) >= expected_calibration_error(y, p) - 1e-12

    def test_brier_hand_computed(self):
        """mean((p - y)^2) for p=[0.0,1.0], y=[0,1] is 0; for p=[0.5,0.5] it is 0.25."""
        assert brier_score(np.array([0, 1]), np.array([0.0, 1.0])) == pytest.approx(0.0)
        assert brier_score(np.array([0, 1]), np.array([0.5, 0.5])) == pytest.approx(0.25)

    def test_reliability_bins_cover_every_sample(self):
        rng = np.random.default_rng(3)
        p = rng.random(500)
        y = (rng.random(500) < p).astype(int)
        curve = reliability_curve(y, p, n_bins=10)
        assert curve.bin_count.sum() == 500

    def test_quantile_binning_balances_counts(self):
        rng = np.random.default_rng(4)
        p = rng.beta(2, 5, 1000)
        y = (rng.random(1000) < p).astype(int)
        curve = reliability_curve(y, p, n_bins=10, strategy="quantile")
        occupied = curve.bin_count[curve.bin_count > 0]
        assert occupied.max() / occupied.min() < 3.0

    def test_temperature_recovers_a_known_scaling(self):
        """Generate y from sigmoid(z) but present 2z: the fitted temperature must be ~2."""
        rng = np.random.default_rng(5)
        z = rng.normal(0, 2, 40000)
        y = (rng.random(40000) < 1 / (1 + np.exp(-z))).astype(int)
        assert fit_temperature(2 * z, y) == pytest.approx(2.0, abs=0.15)

    def test_temperature_of_calibrated_logits_is_one(self):
        rng = np.random.default_rng(6)
        z = rng.normal(0, 2, 40000)
        y = (rng.random(40000) < 1 / (1 + np.exp(-z))).astype(int)
        assert fit_temperature(z, y) == pytest.approx(1.0, abs=0.12)

    def test_temperature_scaling_preserves_ranking(self):
        """Crucial for DDERA: sharpening probabilities must not reorder contributions."""
        z = np.array([-2.0, -0.5, 0.3, 1.7, 3.0])
        scaled = apply_temperature(z, 2.5)
        assert np.array_equal(np.argsort(z), np.argsort(scaled))

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError, match="positive"):
            apply_temperature(np.array([0.0]), 0.0)

    def test_report_contains_the_full_family(self):
        rng = np.random.default_rng(7)
        p = rng.random(500)
        y = (rng.random(500) < p).astype(int)
        report = calibration_report(y, p)
        for key in ("ece", "ece_quantile", "mce", "brier", "reliability"):
            assert key in report


class TestBootstrap:
    def test_interval_contains_the_point_estimate(self, synth):
        rng = np.random.default_rng(0)
        scores = 0.5 * synth.y + rng.random(synth.n_samples)
        result = bootstrap_ci(synth.y, scores, safe_auroc, n_resamples=200, seed=0)
        assert result.lower <= result.point <= result.upper
        assert result.ci_width > 0

    def test_more_data_gives_a_narrower_interval(self):
        rng = np.random.default_rng(1)

        def make(n):
            y = (rng.random(n) < 0.3).astype(int)
            return y, 0.6 * y + rng.random(n)

        small = bootstrap_ci(*make(200), safe_auroc, n_resamples=200, seed=0)
        large = bootstrap_ci(*make(4000), safe_auroc, n_resamples=200, seed=0)
        assert large.ci_width < small.ci_width

    def test_stratified_resampling_preserves_prevalence(self):
        """Unstratified resampling of a rare class can yield replicates with no positives."""
        rng = np.random.default_rng(2)
        y = np.zeros(500, dtype=int)
        y[:10] = 1
        scores = rng.random(500)
        result = bootstrap_ci(y, scores, safe_auroc, n_resamples=300, stratified=True, seed=0)
        assert result.n_valid == 300, "stratification should keep every replicate valid"

    def test_is_deterministic_given_a_seed(self, synth):
        rng = np.random.default_rng(3)
        scores = 0.5 * synth.y + rng.random(synth.n_samples)
        a = bootstrap_ci(synth.y, scores, safe_auroc, n_resamples=100, seed=99)
        b = bootstrap_ci(synth.y, scores, safe_auroc, n_resamples=100, seed=99)
        assert (a.lower, a.upper) == (b.lower, b.upper)

    def test_paired_difference_of_identical_scores_is_exactly_zero(self, synth):
        rng = np.random.default_rng(4)
        scores = rng.random(synth.n_samples)
        result = paired_bootstrap_diff(synth.y, scores, scores, safe_auroc, n_resamples=100, seed=0)
        assert result.point == 0.0
        assert result.lower == 0.0 and result.upper == 0.0
        assert not is_significant(result)

    def test_paired_difference_detects_a_real_gap(self, synth):
        """The headline comparison: is the CBM genuinely worse than the black box?"""
        rng = np.random.default_rng(5)
        good = 2.0 * synth.y + rng.random(synth.n_samples)
        poor = rng.random(synth.n_samples)
        result = paired_bootstrap_diff(synth.y, good, poor, safe_auroc, n_resamples=300, seed=0)
        assert result.point > 0
        assert is_significant(result), "a large real gap must be detected as significant"

    def test_shape_mismatch_raises(self, synth):
        with pytest.raises(ValueError, match="Shape mismatch"):
            bootstrap_ci(synth.y, np.zeros(5), safe_auroc)


def test_prediction_agreement():
    assert prediction_agreement([0.9, 0.1], [0.8, 0.2]) == 1.0
    assert prediction_agreement([0.9, 0.1], [0.1, 0.9]) == 0.0
    assert prediction_agreement([0.9, 0.1], [0.9, 0.9]) == 0.5
