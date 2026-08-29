"""CheXpert manifest construction.

Path parsing is strict on purpose: a mis-parsed patient ID would put one patient in two
splits and inflate every metric in the project. These tests confirm it refuses ambiguity
rather than guessing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ddera.data.chexpert import (
    concept_matrix,
    cooccurrence_matrix,
    manifest_from_frame,
    parse_chexpert_path,
    target_vector,
)
from ddera.data.labels import CHEXPERT_OBSERVATIONS


@pytest.fixture
def raw_csv() -> pd.DataFrame:
    """A miniature CheXpert CSV covering positive/negative/uncertain/blank and both views."""
    rows = [
        (
            "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg",
            "Female",
            61,
            "Frontal",
            "AP",
        ),
        (
            "CheXpert-v1.0-small/train/patient00001/study2/view1_frontal.jpg",
            "Female",
            61,
            "Frontal",
            "PA",
        ),
        (
            "CheXpert-v1.0-small/train/patient00002/study1/view1_frontal.jpg",
            "Male",
            45,
            "Frontal",
            "AP",
        ),
        (
            "CheXpert-v1.0-small/train/patient00002/study1/view2_lateral.jpg",
            "Male",
            45,
            "Lateral",
            np.nan,
        ),
        (
            "CheXpert-v1.0-small/train/patient00003/study1/view1_frontal.jpg",
            "Male",
            72,
            "Frontal",
            "PA",
        ),
    ]
    df = pd.DataFrame(rows, columns=["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"])
    rng = np.random.default_rng(0)
    for observation in CHEXPERT_OBSERVATIONS:
        df[observation] = rng.choice([1.0, 0.0, -1.0, np.nan], size=len(df))
    df["Pneumonia"] = [1.0, 0.0, -1.0, np.nan, 1.0]
    df["Consolidation"] = [1.0, 1.0, 0.0, 0.0, 1.0]
    df["Edema"] = [1.0, 0.0, 0.0, 0.0, 1.0]
    return df


class TestPathParsing:
    def test_extracts_patient_study_and_view(self):
        parsed = parse_chexpert_path(
            "CheXpert-v1.0-small/train/patient00123/study4/view1_frontal.jpg"
        )
        assert parsed["patient_id"] == "patient00123"
        assert parsed["study"] == "study4"
        assert parsed["study_id"] == "patient00123/study4"
        assert parsed["view_file"] == "view1_frontal.jpg"

    def test_handles_windows_separators(self):
        parsed = parse_chexpert_path(
            r"CheXpert-v1.0-small\train\patient00007\study2\view1_frontal.jpg"
        )
        assert parsed["patient_id"] == "patient00007"
        assert parsed["study"] == "study2"

    def test_study_ids_are_unique_across_patients(self):
        """'study1' alone is ambiguous; the study_id must be patient-qualified."""
        a = parse_chexpert_path("x/patient00001/study1/view1_frontal.jpg")
        b = parse_chexpert_path("x/patient00002/study1/view1_frontal.jpg")
        assert a["study"] == b["study"]
        assert a["study_id"] != b["study_id"]

    @pytest.mark.parametrize(
        "bad",
        [
            "not/a/chexpert/path.jpg",
            "CheXpert/train/subject01/study1/view1.jpg",  # wrong patient prefix
            "CheXpert/train/patient00001/view1_frontal.jpg",  # study level missing
            "",
        ],
    )
    def test_malformed_paths_raise_rather_than_guess(self, bad):
        with pytest.raises(ValueError, match="does not match the CheXpert layout"):
            parse_chexpert_path(bad)


class TestManifest:
    def test_parses_all_rows_and_keeps_frontal_only(self, raw_csv):
        manifest, summary = manifest_from_frame(raw_csv, target="Pneumonia")
        assert len(manifest) == 4, "the lateral row should be dropped"
        assert summary.dropped_by_view == 1
        assert set(manifest["view"]) <= {"AP", "PA"}

    def test_counts_patients_and_studies(self, raw_csv):
        manifest, summary = manifest_from_frame(raw_csv, target="Pneumonia")
        assert summary.n_patients == 3
        assert summary.n_studies == manifest["study_id"].nunique()

    def test_raw_label_values_are_preserved_untouched(self, raw_csv):
        """Applying an uncertainty policy is a separate, explicit step -- not done here."""
        manifest, _ = manifest_from_frame(raw_csv, target="Pneumonia", views=[])
        assert manifest["raw_Pneumonia"].tolist()[:3] == [1.0, 0.0, -1.0]
        assert np.isnan(manifest["raw_Pneumonia"].tolist()[3])

    def test_default_concepts_exclude_target_and_no_finding(self, raw_csv):
        manifest, _ = manifest_from_frame(raw_csv, target="Pneumonia")
        concepts = manifest.attrs["concepts"]
        assert "Pneumonia" not in concepts
        assert "No Finding" not in concepts
        assert len(concepts) == 12, "ADR-003 specifies a 12-concept bottleneck"

    def test_explicit_concepts_are_respected(self, raw_csv):
        manifest, _ = manifest_from_frame(
            raw_csv, target="Pneumonia", concepts=["Consolidation", "Edema"]
        )
        assert manifest.attrs["concepts"] == ["Consolidation", "Edema"]
        assert "raw_Consolidation" in manifest.columns

    def test_keeping_all_views(self, raw_csv):
        manifest, summary = manifest_from_frame(raw_csv, target="Pneumonia", views=[])
        assert len(manifest) == 5
        assert summary.dropped_by_view == 0

    def test_drop_missing_target_is_opt_in(self, raw_csv):
        kept, _ = manifest_from_frame(raw_csv, target="Pneumonia", views=[])
        dropped, summary = manifest_from_frame(
            raw_csv, target="Pneumonia", views=[], drop_missing_target=True
        )
        assert len(kept) == 5
        assert len(dropped) == 4
        assert summary.dropped_by_missing_target == 1

    def test_missing_target_column_raises(self, raw_csv):
        with pytest.raises(ValueError, match="Target column"):
            manifest_from_frame(raw_csv, target="Nonexistent")

    def test_missing_concept_column_raises(self, raw_csv):
        with pytest.raises(ValueError, match="Concept columns not found"):
            manifest_from_frame(raw_csv, target="Pneumonia", concepts=["Imaginary Finding"])

    def test_missing_path_column_raises(self):
        with pytest.raises(ValueError, match="'Path' column"):
            manifest_from_frame(pd.DataFrame({"Pneumonia": [1.0]}))

    def test_summary_renders(self, raw_csv):
        _, summary = manifest_from_frame(raw_csv, target="Pneumonia")
        assert "patients" in summary.summary()


class TestExtraction:
    def test_concept_matrix_shape_and_values(self, raw_csv):
        manifest, _ = manifest_from_frame(
            raw_csv, target="Pneumonia", concepts=["Consolidation", "Edema"], views=[]
        )
        matrix = concept_matrix(manifest, ["Consolidation", "Edema"])
        assert matrix.shape == (5, 2)
        assert matrix[:, 0].tolist() == [1.0, 1.0, 0.0, 0.0, 1.0]

    def test_target_vector_matches_source(self, raw_csv):
        manifest, _ = manifest_from_frame(raw_csv, target="Pneumonia", views=[])
        target = target_vector(manifest, "Pneumonia")
        assert target[:3].tolist() == [1.0, 0.0, -1.0]

    def test_missing_column_raises(self, raw_csv):
        manifest, _ = manifest_from_frame(raw_csv, target="Pneumonia")
        with pytest.raises(ValueError, match="missing"):
            concept_matrix(manifest, ["Not A Concept"])
        with pytest.raises(ValueError, match="missing"):
            target_vector(manifest, "Not A Target")

    def test_cooccurrence_is_symmetric_with_counts_on_the_diagonal(self, raw_csv):
        manifest, _ = manifest_from_frame(
            raw_csv, target="Pneumonia", concepts=["Consolidation", "Edema"], views=[]
        )
        matrix = cooccurrence_matrix(manifest, ["Consolidation", "Edema"])
        assert matrix.shape == (2, 2)
        assert np.array_equal(matrix.to_numpy(), matrix.to_numpy().T)
        assert matrix.loc["Consolidation", "Consolidation"] == 3  # 1,1,0,0,1
        assert matrix.loc["Edema", "Edema"] == 2  # 1,0,0,0,1
        assert matrix.loc["Consolidation", "Edema"] == 2  # rows 0 and 4
