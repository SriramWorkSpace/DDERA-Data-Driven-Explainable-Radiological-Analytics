"""Configuration and the concept specification.

``ConceptSpec`` is the object that encodes Invariant 4 as data: it refuses a configuration
in which the target also appears among the concepts, because that makes the bottleneck
circular. Phase 8's generality claim rests on this file being the *only* thing that changes
between domains.
"""

from __future__ import annotations

import pytest

from ddera.config import CONFIG_ROOT, ConceptSpec, EscalationSpec, load_yaml

CHEXPERT_CONFIG = CONFIG_ROOT / "concepts" / "chexpert_v1.yaml"


class TestConceptSpec:
    def test_loads_the_project_config(self):
        spec = ConceptSpec.from_yaml(CHEXPERT_CONFIG)
        assert spec.name == "chexpert_v1"
        assert spec.domain == "chest_xray"
        assert spec.target == "Pneumonia"
        assert spec.n_concepts == 12

    def test_target_is_not_among_the_concepts(self):
        """Invariant 4: a target inside the bottleneck would be circular."""
        spec = ConceptSpec.from_yaml(CHEXPERT_CONFIG)
        assert spec.target not in spec.concepts

    def test_no_finding_is_excluded(self):
        """It is derivable from the other observations, so it would leak by construction."""
        spec = ConceptSpec.from_yaml(CHEXPERT_CONFIG)
        assert "No Finding" not in spec.concepts
        assert any(e["label"] == "No Finding" for e in spec.excluded)

    def test_circular_bottleneck_is_rejected(self):
        with pytest.raises(ValueError, match="circular"):
            ConceptSpec(
                name="bad",
                domain="d",
                dataset="ds",
                target="Pneumonia",
                concepts=["Edema", "Pneumonia"],
            )

    def test_duplicate_concepts_are_rejected(self):
        with pytest.raises(ValueError, match="Duplicate"):
            ConceptSpec(
                name="bad",
                domain="d",
                dataset="ds",
                target="T",
                concepts=["Edema", "Edema"],
            )

    def test_empty_bottleneck_is_rejected(self):
        with pytest.raises(ValueError, match="not a bottleneck"):
            ConceptSpec(name="bad", domain="d", dataset="ds", target="T", concepts=[])

    def test_leakage_watchlist_must_name_real_concepts(self):
        with pytest.raises(ValueError, match="non-existent"):
            ConceptSpec(
                name="bad",
                domain="d",
                dataset="ds",
                target="T",
                concepts=["Edema"],
                expected_leakage_watchlist=["Ghost"],
            )

    def test_watchlist_records_the_expected_contamination(self):
        """Consolidation and Lung Opacity are radiographic evidence *of* pneumonia."""
        spec = ConceptSpec.from_yaml(CHEXPERT_CONFIG)
        assert "Consolidation" in spec.expected_leakage_watchlist
        assert "Lung Opacity" in spec.expected_leakage_watchlist

    def test_uncertainty_policy_matches_adr_004(self):
        spec = ConceptSpec.from_yaml(CHEXPERT_CONFIG)
        assert spec.uncertainty.concept_policy == "u_mask"
        assert spec.uncertainty.target_policy == "u_ignore"
        assert spec.uncertainty.blank_policy == "negative"
        assert set(spec.uncertainty.sensitivity_analysis) == {"u_zeros", "u_ones"}

    def test_cohort_matches_adr_005(self):
        spec = ConceptSpec.from_yaml(CHEXPERT_CONFIG)
        assert spec.cohort.views == ["AP", "PA"]
        assert spec.cohort.split_level == "patient"
        assert sum(spec.cohort.split_ratios.values()) == pytest.approx(1.0)

    def test_to_dict_is_serialisable(self):
        spec = ConceptSpec.from_yaml(CHEXPERT_CONFIG)
        payload = spec.to_dict()
        assert payload["n_concepts"] == 12
        assert payload["target"] == "Pneumonia"


class TestEscalationRule:
    """ADR-003's rule is committed in advance so the choice stays data-driven."""

    def test_loads_five_fallback_targets(self):
        spec = ConceptSpec.from_yaml(CHEXPERT_CONFIG)
        assert len(spec.escalation.fallback_targets) == 5
        assert "Pneumonia" not in spec.escalation.fallback_targets

    def test_no_escalation_when_criteria_are_met(self):
        rule = EscalationSpec(min_test_positives=250, max_auroc_ci_width=0.10)
        escalate, reason = rule.should_escalate(n_test_positives=500, auroc_ci_width=0.05)
        assert escalate is False
        assert "no escalation" in reason

    def test_escalates_on_too_few_positives(self):
        rule = EscalationSpec(min_test_positives=250, max_auroc_ci_width=0.10)
        escalate, reason = rule.should_escalate(n_test_positives=100, auroc_ci_width=0.05)
        assert escalate is True
        assert "100 test positives" in reason

    def test_escalates_on_a_wide_interval(self):
        rule = EscalationSpec(min_test_positives=250, max_auroc_ci_width=0.10)
        escalate, reason = rule.should_escalate(n_test_positives=500, auroc_ci_width=0.20)
        assert escalate is True
        assert "CI width" in reason

    def test_reports_both_reasons_together(self):
        rule = EscalationSpec(min_test_positives=250, max_auroc_ci_width=0.10)
        escalate, reason = rule.should_escalate(n_test_positives=10, auroc_ci_width=0.30)
        assert escalate is True
        assert "positives" in reason and "CI width" in reason

    def test_boundary_values_do_not_escalate(self):
        rule = EscalationSpec(min_test_positives=250, max_auroc_ci_width=0.10)
        escalate, _ = rule.should_escalate(n_test_positives=250, auroc_ci_width=0.10)
        assert escalate is False


class TestYamlLoading:
    def test_resolves_bare_names_against_configs(self):
        assert load_yaml("concepts/chexpert_v1.yaml")["name"] == "chexpert_v1"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_yaml("concepts/does_not_exist.yaml")
