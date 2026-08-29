"""Patient-level split integrity (ADR-005).

``test_no_patient_appears_in_two_splits`` is the single most important test in the
repository. Patient leakage inflates every metric in the project, and it is invisible in the
results -- the numbers simply come out better than they should. This test must never be
skipped, weakened, or marked xfail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ddera.data.splits import (
    assert_no_patient_leakage,
    check_split_integrity,
    patient_level_split,
)


@pytest.fixture
def manifest(synth):
    return synth.to_manifest()


class TestPatientLevelSplit:
    def test_no_patient_appears_in_two_splits(self, manifest):
        """THE critical guarantee. Everything else in the project rests on it."""
        split_df, _ = patient_level_split(manifest, seed=42)
        by_split = {s: set(g["patient_id"]) for s, g in split_df.groupby("split")}
        assert by_split["train"] & by_split["val"] == set()
        assert by_split["train"] & by_split["test"] == set()
        assert by_split["val"] & by_split["test"] == set()

    def test_every_row_is_assigned(self, manifest):
        split_df, _ = patient_level_split(manifest, seed=42)
        assert split_df["split"].notna().all()
        assert len(split_df) == len(manifest)
        assert set(split_df["split"].unique()) == {"train", "val", "test"}

    def test_all_rows_of_a_patient_land_together(self, manifest):
        """Multiple studies per patient must not be separated."""
        split_df, _ = patient_level_split(manifest, seed=42)
        per_patient = split_df.groupby("patient_id")["split"].nunique()
        assert (per_patient == 1).all(), "a patient was split across sets"

    def test_ratios_are_approximately_respected(self, manifest):
        split_df, report = patient_level_split(
            manifest, ratios={"train": 0.7, "val": 0.1, "test": 0.2}, seed=42
        )
        total = len(split_df)
        assert report.n_images["train"] / total == pytest.approx(0.7, abs=0.05)
        assert report.n_images["val"] / total == pytest.approx(0.1, abs=0.05)
        assert report.n_images["test"] / total == pytest.approx(0.2, abs=0.05)

    def test_stratification_preserves_prevalence(self, manifest):
        _, report = patient_level_split(manifest, seed=42)
        prevalences = list(report.prevalence.values())
        assert max(prevalences) - min(prevalences) < 0.10

    def test_split_is_deterministic_given_a_seed(self, manifest):
        a, _ = patient_level_split(manifest, seed=7)
        b, _ = patient_level_split(manifest, seed=7)
        assert a["split"].equals(b["split"])

    def test_different_seeds_give_different_splits(self, manifest):
        a, _ = patient_level_split(manifest, seed=1)
        b, _ = patient_level_split(manifest, seed=2)
        assert not a["split"].equals(b["split"])

    def test_input_frame_is_not_mutated(self, manifest):
        before = manifest.copy()
        patient_level_split(manifest, seed=42)
        pd.testing.assert_frame_equal(manifest, before)


class TestLeakageDetection:
    def test_detector_catches_deliberate_leakage(self):
        """The guard must actually fire. A leakage test that cannot fail is worthless."""
        leaky = pd.DataFrame(
            {
                "patient_id": ["p1", "p1", "p2", "p3"],
                "split": ["train", "test", "train", "test"],
                "target": [1, 1, 0, 0],
            }
        )
        with pytest.raises(AssertionError, match="PATIENT LEAKAGE DETECTED"):
            assert_no_patient_leakage(leaky)

    def test_detector_passes_a_clean_split(self):
        clean = pd.DataFrame(
            {
                "patient_id": ["p1", "p1", "p2", "p3"],
                "split": ["train", "train", "val", "test"],
                "target": [1, 1, 0, 0],
            }
        )
        assert_no_patient_leakage(clean)

    def test_error_message_names_the_offending_splits(self):
        leaky = pd.DataFrame(
            {"patient_id": ["p1", "p1"], "split": ["train", "test"], "target": [1, 0]}
        )
        with pytest.raises(AssertionError) as excinfo:
            assert_no_patient_leakage(leaky)
        assert "train" in str(excinfo.value) and "test" in str(excinfo.value)


class TestSplitValidation:
    def test_ratios_must_sum_to_one(self, manifest):
        with pytest.raises(ValueError, match="sum to 1"):
            patient_level_split(manifest, ratios={"train": 0.5, "val": 0.1, "test": 0.2})

    def test_ratios_must_have_the_expected_keys(self, manifest):
        with pytest.raises(ValueError, match="exactly the keys"):
            patient_level_split(manifest, ratios={"train": 0.8, "test": 0.2})

    def test_missing_column_raises(self, manifest):
        with pytest.raises(ValueError, match="not found"):
            patient_level_split(manifest, patient_col="nonexistent")

    def test_single_class_target_raises(self):
        df = pd.DataFrame({"patient_id": [f"p{i}" for i in range(20)], "target": [1] * 20})
        with pytest.raises(ValueError, match="All patients share target"):
            patient_level_split(df)

    def test_too_few_patients_raises(self):
        df = pd.DataFrame({"patient_id": ["p1", "p2"], "target": [0, 1]})
        with pytest.raises(ValueError, match="at least 3 patients"):
            patient_level_split(df)

    def test_rare_class_raises_with_an_actionable_message(self):
        df = pd.DataFrame(
            {"patient_id": [f"p{i}" for i in range(50)], "target": [1] * 2 + [0] * 48}
        )
        with pytest.raises(ValueError, match="cannot stratify"):
            patient_level_split(df)


class TestIntegrityReport:
    def test_clean_split_reports_ok(self, manifest):
        split_df, _ = patient_level_split(manifest, seed=42)
        report = check_split_integrity(split_df)
        assert report["ok"] is True
        assert report["leakage_free"] is True
        assert report["issues"] == []

    def test_leaky_split_is_flagged(self, manifest):
        split_df, _ = patient_level_split(manifest, seed=42)
        # Force one patient into two splits.
        victim = split_df.loc[split_df["split"] == "train", "patient_id"].iloc[0]
        idx = split_df.index[split_df["patient_id"] == victim][0]
        split_df.loc[idx, "split"] = "test"

        report = check_split_integrity(split_df)
        assert report["ok"] is False
        assert report["leakage_free"] is False

    def test_report_counts_patients_and_images(self, manifest):
        split_df, _ = patient_level_split(manifest, seed=42)
        report = check_split_integrity(split_df)
        assert sum(report["n_images"].values()) == len(split_df)
        assert sum(report["n_patients"].values()) == split_df["patient_id"].nunique()

    def test_ordinary_sampling_noise_is_not_flagged(self, manifest):
        """A correct split must not trip the prevalence check.

        The val split holds ~10% of patients, where binomial noise moves prevalence by
        10%+ relative. A flat percentage tolerance would false-alarm here; the z-score
        check should not.
        """
        split_df, _ = patient_level_split(manifest, seed=42)
        report = check_split_integrity(split_df)
        assert all(abs(z) < 3.0 for z in report["prevalence_z_scores"].values())
        assert not any("prevalence" in issue for issue in report["issues"])

    def test_deliberate_prevalence_skew_is_flagged(self, manifest):
        """The check must still catch a genuinely broken stratification."""
        split_df, _ = patient_level_split(manifest, seed=42)
        skewed = split_df.copy()

        # Assign whole patients by target, so prevalence skews without leaking patients.
        patient_target = skewed.groupby("patient_id")["target"].max().sort_values()
        ordered = patient_target.index.tolist()
        cut = int(0.8 * len(ordered))
        assignment = {p: ("train" if i < cut else "test") for i, p in enumerate(ordered)}
        skewed["split"] = skewed["patient_id"].map(assignment)

        report = check_split_integrity(skewed)
        assert report["leakage_free"] is True, "this fixture must skew prevalence, not leak"
        assert report["ok"] is False
        assert max(abs(z) for z in report["prevalence_z_scores"].values()) > 3.0


def test_split_report_summary_renders(manifest):
    _, report = patient_level_split(manifest, seed=42)
    text = report.summary()
    assert "train" in text and "test" in text
    assert np.isclose(sum(report.ratios_achieved.values()), 1.0)
