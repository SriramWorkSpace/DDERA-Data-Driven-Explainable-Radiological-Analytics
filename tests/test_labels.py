"""Uncertainty policies (ADR-004).

A bug here silently changes what the model is taught about every concept, and would
invalidate every concept-quality number in the project. The policies are therefore checked
against an explicit truth table rather than by round-tripping.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddera.data.labels import (
    CHEXPERT_OBSERVATIONS,
    apply_uncertainty_policy,
    encode_concept_matrix,
    label_distribution,
    mask_coverage,
    rows_to_keep,
)

# raw:            positive, negative, uncertain, blank
RAW = np.array([1.0, 0.0, -1.0, np.nan])


class TestUncertaintyPolicies:
    """Truth table: (policy, blank_policy) -> (labels, mask)."""

    def test_u_mask_masks_uncertain_keeps_blank_as_negative(self):
        labels, mask = apply_uncertainty_policy(RAW, "u_mask", "negative")
        assert labels.tolist() == [1.0, 0.0, 0.0, 0.0]
        assert mask.tolist() == [1.0, 1.0, 0.0, 1.0]

    def test_u_zeros_treats_uncertain_as_negative(self):
        labels, mask = apply_uncertainty_policy(RAW, "u_zeros", "negative")
        assert labels.tolist() == [1.0, 0.0, 0.0, 0.0]
        assert mask.tolist() == [1.0, 1.0, 1.0, 1.0], "u_zeros must not mask anything"

    def test_u_ones_treats_uncertain_as_positive(self):
        labels, mask = apply_uncertainty_policy(RAW, "u_ones", "negative")
        assert labels.tolist() == [1.0, 0.0, 1.0, 0.0]
        assert mask.tolist() == [1.0, 1.0, 1.0, 1.0]

    def test_blank_policy_mask_masks_blanks(self):
        labels, mask = apply_uncertainty_policy(RAW, "u_mask", "mask")
        assert labels.tolist() == [1.0, 0.0, 0.0, 0.0]
        assert mask.tolist() == [1.0, 1.0, 0.0, 0.0]

    def test_u_zeros_and_u_ones_differ_only_on_uncertain(self):
        zeros, _ = apply_uncertainty_policy(RAW, "u_zeros", "negative")
        ones, _ = apply_uncertainty_policy(RAW, "u_ones", "negative")
        differences = np.flatnonzero(zeros != ones)
        assert differences.tolist() == [2], "policies must differ only at the uncertain entry"

    def test_output_is_strictly_binary(self):
        """No -1 or NaN may survive; either would poison the loss silently."""
        for policy in ("u_mask", "u_zeros", "u_ones", "u_ignore"):
            labels, _ = apply_uncertainty_policy(RAW, policy, "negative")
            assert set(np.unique(labels)).issubset({0.0, 1.0})
            assert not np.isnan(labels).any()

    def test_masked_positions_are_zero_not_negative_one(self):
        """Fail-safe: an accidental unmasked loss degrades to 'negative', never to NaN."""
        labels, mask = apply_uncertainty_policy(RAW, "u_mask", "negative")
        assert np.all(labels[mask == 0] == 0.0)

    def test_input_is_not_mutated(self):
        original = RAW.copy()
        apply_uncertainty_policy(RAW, "u_ones", "negative")
        assert np.array_equal(RAW, original, equal_nan=True)

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError, match="Unknown policy"):
            apply_uncertainty_policy(RAW, "u_invent", "negative")
        with pytest.raises(ValueError, match="Unknown blank_policy"):
            apply_uncertainty_policy(RAW, "u_mask", "guess")

    def test_shape_is_preserved(self):
        raw2d = np.array([[1.0, -1.0], [0.0, np.nan]])
        labels, mask = apply_uncertainty_policy(raw2d, "u_mask", "negative")
        assert labels.shape == mask.shape == raw2d.shape


class TestRowsToKeep:
    def test_u_ignore_drops_uncertain_rows(self):
        assert rows_to_keep(RAW, "u_ignore", "negative").tolist() == [True, True, False, True]

    def test_other_policies_keep_all_rows(self):
        for policy in ("u_mask", "u_zeros", "u_ones"):
            assert rows_to_keep(RAW, policy, "negative").all()

    def test_blank_mask_also_drops_blanks(self):
        assert rows_to_keep(RAW, "u_ignore", "mask").tolist() == [True, True, False, False]


class TestConceptMatrix:
    def test_encode_matches_elementwise_policy(self, synth_uncertain):
        labels, mask = encode_concept_matrix(synth_uncertain.raw_labels, "u_mask", "negative")
        assert labels.shape == mask.shape == synth_uncertain.raw_labels.shape
        is_uncertain = synth_uncertain.raw_labels == -1.0
        assert np.all(mask[is_uncertain] == 0.0)
        assert np.all(mask[~is_uncertain & ~np.isnan(synth_uncertain.raw_labels)] == 1.0)

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            encode_concept_matrix(np.array([1.0, 0.0]), "u_mask")

    def test_mask_coverage_reports_per_concept(self, synth_uncertain):
        _, mask = encode_concept_matrix(synth_uncertain.raw_labels, "u_mask", "negative")
        coverage = mask_coverage(mask)
        assert 0.0 < coverage["overall"] < 1.0, "fixture should contain some uncertain labels"
        assert len(coverage) == synth_uncertain.n_concepts + 1

    def test_coverage_is_one_when_nothing_is_masked(self):
        assert mask_coverage(np.ones((10, 4)))["overall"] == 1.0


class TestLabelDistribution:
    def test_counts_every_category(self):
        raw = np.array([1.0, 1.0, 0.0, -1.0, np.nan, np.nan])
        assert label_distribution(raw) == {
            "positive": 2,
            "negative": 1,
            "uncertain": 1,
            "blank": 2,
            "total": 6,
        }

    def test_categories_sum_to_total(self, synth_uncertain):
        d = label_distribution(synth_uncertain.raw_labels)
        assert d["positive"] + d["negative"] + d["uncertain"] + d["blank"] == d["total"]


def test_chexpert_observation_list_is_correct():
    """Guards against a typo silently dropping a concept from the bottleneck."""
    assert len(CHEXPERT_OBSERVATIONS) == 14
    assert CHEXPERT_OBSERVATIONS[0] == "No Finding"
    assert "Pneumonia" in CHEXPERT_OBSERVATIONS
    assert len(set(CHEXPERT_OBSERVATIONS)) == 14
