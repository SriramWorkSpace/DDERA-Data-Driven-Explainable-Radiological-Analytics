"""Phase-1 acquisition pipeline: local-download inspection, manifest/splits persistence,
image probing, and the CLI.

No real CheXpert data -- a synthetic CheXpert-shaped CSV (and, where needed, tiny generated
JPEGs) stand in. The ADRs exercised: 003 (target + escalation), 004 (target u_ignore),
005 (frontal-only, patient-level split, held-out valid.csv), 011 (integrity report).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # so `scripts.get_data` imports (namespace package)

from ddera.config import ConceptSpec  # noqa: E402
from ddera.data.acquire import build_processed_dataset, inspect_download  # noqa: E402
from ddera.data.chexpert import load_manifest  # noqa: E402
from ddera.data.labels import CHEXPERT_OBSERVATIONS  # noqa: E402
from ddera.data.splits import assert_no_patient_leakage, load_splits  # noqa: E402

CONCEPT_COLS = [c for c in CHEXPERT_OBSERVATIONS if c not in ("No Finding", "Pneumonia")]


def _spec() -> ConceptSpec:
    return ConceptSpec.from_yaml("configs/concepts/chexpert_v1.yaml")


def _write_synthetic_chexpert(
    root: Path,
    *,
    n_patients: int = 60,
    studies_per_patient: int = 2,
    with_images: bool = False,
    n_missing: int = 0,
    n_corrupt: int = 0,
    seed: int = 0,
) -> Path:
    """Write a miniature CheXpert-v1.0-small tree (train.csv, valid.csv, optional images)."""
    root = Path(root)
    ds = root / "CheXpert-v1.0-small"
    ds.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    for p in range(n_patients):
        pid = f"patient{p:05d}"
        patient_pos = rng.random() < 0.45  # ~45% of patients get a positive study
        for s in range(studies_per_patient):
            row = {
                "Path": f"CheXpert-v1.0-small/train/{pid}/study{s + 1}/view1_frontal.jpg",
                "Sex": str(rng.choice(["Male", "Female"])),
                "Age": int(rng.integers(20, 90)),
                "Frontal/Lateral": "Frontal",
                "AP/PA": str(rng.choice(["AP", "PA"])),
            }
            for obs in CHEXPERT_OBSERVATIONS:
                row[obs] = float(rng.choice([1.0, 0.0, -1.0, np.nan]))
            row["Pneumonia"] = (
                1.0
                if (patient_pos and s == 0)
                else float(rng.choice([0.0, 0.0, 0.0, -1.0, np.nan]))
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    if with_images:
        # Guarantee the first rows survive the target policy so probe counts are stable.
        df.loc[df.index[:8], "Pneumonia"] = 1.0
    df.to_csv(ds / "train.csv", index=False)

    valid = df.head(6).copy()
    valid["Pneumonia"] = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    valid.to_csv(ds / "valid.csv", index=False)

    if with_images:
        paths = list(df["Path"])
        missing = set(paths[:n_missing])
        corrupt = set(paths[n_missing : n_missing + n_corrupt])
        for rel in paths:
            if rel in missing:
                continue
            fpath = root / rel
            fpath.parent.mkdir(parents=True, exist_ok=True)
            if rel in corrupt:
                fpath.write_bytes(b"not a real jpeg")
            else:
                Image.new("L", (16, 16), color=127).save(fpath, format="JPEG")
    return root


# ---------------------------------------------------------------------------------------


class TestInspectDownload:
    def test_finds_nested_chexpert_dir_and_image_root(self, tmp_path):
        _write_synthetic_chexpert(tmp_path, with_images=True)
        layout = inspect_download(tmp_path)
        assert layout.ok
        assert layout.train_csv.name == "train.csv"
        assert layout.valid_csv is not None
        assert layout.image_root == tmp_path  # CSV 'Path' values resolve against here
        assert layout.has_images

    def test_accepts_dir_pointed_straight_at_chexpert_subdir(self, tmp_path):
        _write_synthetic_chexpert(tmp_path)
        layout = inspect_download(tmp_path / "CheXpert-v1.0-small")
        assert layout.ok
        assert layout.train_csv.name == "train.csv"

    def test_missing_train_csv_is_not_ok(self, tmp_path):
        layout = inspect_download(tmp_path)
        assert not layout.ok
        assert layout.train_csv is None
        assert not layout.has_images


class TestBuildProcessedDataset:
    def test_writes_manifest_and_splits_with_sidecars(self, tmp_path):
        _write_synthetic_chexpert(tmp_path, n_patients=80)
        out = tmp_path / "processed"
        art = build_processed_dataset(tmp_path, _spec(), out, seed=42, check_images=False)

        for name in ("manifest.parquet", "manifest.json", "splits.parquet", "splits.json"):
            assert (out / name).exists(), name
        assert art.external_path == out / "external.parquet"
        assert (out / "external.parquet").exists()

        split_df, sidecar = load_splits(out / "splits.parquet")
        assert set(split_df["split"].unique()) == {"train", "val", "test"}
        assert "split_report" in sidecar
        assert "integrity" in sidecar

    def test_no_patient_appears_in_two_splits(self, tmp_path):
        _write_synthetic_chexpert(tmp_path, n_patients=80)
        out = tmp_path / "processed"
        build_processed_dataset(tmp_path, _spec(), out, seed=42, check_images=False)
        split_df, _ = load_splits(out / "splits.parquet")
        assert_no_patient_leakage(split_df)  # raises on leakage
        per_patient = split_df.groupby("patient_id")["split"].nunique()
        assert (per_patient == 1).all()

    def test_target_is_binary_and_uncertain_rows_dropped(self, tmp_path):
        _write_synthetic_chexpert(tmp_path, n_patients=80)
        out = tmp_path / "processed"
        build_processed_dataset(tmp_path, _spec(), out, seed=1, check_images=False)
        manifest, meta = load_manifest(out / "manifest.parquet")
        assert set(np.unique(manifest["target"])) <= {0, 1}
        assert (manifest["raw_Pneumonia"] != -1.0).all()  # ADR-004 u_ignore
        assert meta["target"] == "Pneumonia"
        assert meta["concepts"] == CONCEPT_COLS

    def test_frontal_only_and_concepts_persisted_raw(self, tmp_path):
        _write_synthetic_chexpert(tmp_path, n_patients=60)
        out = tmp_path / "processed"
        build_processed_dataset(tmp_path, _spec(), out, seed=0, check_images=False)
        manifest, _ = load_manifest(out / "manifest.parquet")
        assert set(manifest["view"]) <= {"AP", "PA"}  # ADR-005
        for concept in CONCEPT_COLS:
            assert f"raw_{concept}" in manifest.columns  # raw, policy NOT baked in
        assert not any(c == concept for c in manifest.columns for concept in CONCEPT_COLS)

    def test_split_is_reproducible_for_a_fixed_seed(self, tmp_path):
        _write_synthetic_chexpert(tmp_path, n_patients=70)
        a, b = tmp_path / "a", tmp_path / "b"
        build_processed_dataset(tmp_path, _spec(), a, seed=7, check_images=False)
        build_processed_dataset(tmp_path, _spec(), b, seed=7, check_images=False)
        da, _ = load_splits(a / "splits.parquet")
        db, _ = load_splits(b / "splits.parquet")
        merged = da[["path", "split"]].merge(
            db[["path", "split"]], on="path", suffixes=("_a", "_b")
        )
        assert len(merged) == len(da)
        assert (merged["split_a"] == merged["split_b"]).all()

    def test_ratios_are_approximately_respected(self, tmp_path):
        _write_synthetic_chexpert(tmp_path, n_patients=120)
        out = tmp_path / "processed"
        art = build_processed_dataset(tmp_path, _spec(), out, seed=42, check_images=False)
        achieved = art.split_report.ratios_achieved
        assert achieved["train"] == pytest.approx(0.70, abs=0.12)
        assert achieved["test"] == pytest.approx(0.20, abs=0.10)

    def test_missing_dataset_raises_with_acquisition_help(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Stanford AIMI"):
            build_processed_dataset(tmp_path, _spec(), tmp_path / "out")

    def test_escalation_trigger_is_reported_not_raised(self, tmp_path):
        _write_synthetic_chexpert(tmp_path, n_patients=60, seed=3)
        out = tmp_path / "processed"
        art = build_processed_dataset(tmp_path, _spec(), out, seed=3, check_images=False)
        # Pneumonia is sparse at this scale -> the ADR-003 trigger fires, but as advisory.
        assert art.escalation["would_escalate"] is True
        assert art.escalation["n_test_positives"] < 250
        assert (out / "splits.parquet").exists()  # pipeline still completed


class TestImageProbing:
    def test_reports_missing_and_corrupt_and_can_drop(self, tmp_path):
        _write_synthetic_chexpert(
            tmp_path, n_patients=40, with_images=True, n_missing=3, n_corrupt=2
        )
        out = tmp_path / "processed"
        art = build_processed_dataset(
            tmp_path, _spec(), out, seed=0, check_images=True, drop_unreadable=True
        )
        assert art.image_report.dimensions_available
        assert art.image_report.n_missing >= 1
        assert art.image_report.n_unreadable >= 1

        manifest, _ = load_manifest(out / "manifest.parquet")
        assert manifest["width"].notna().all()  # unreadable rows were dropped
        assert (manifest["width"] == 16).all()
        assert (manifest["height"] == 16).all()

    def test_probe_is_skipped_without_images(self, tmp_path):
        _write_synthetic_chexpert(tmp_path, n_patients=40, with_images=False)
        out = tmp_path / "processed"
        art = build_processed_dataset(tmp_path, _spec(), out, seed=0)  # auto -> no image tree
        assert not art.image_report.dimensions_available
        manifest, _ = load_manifest(out / "manifest.parquet")
        assert "width" not in manifest.columns


class TestManifestPersistence:
    def test_write_then_load_round_trips_the_contract(self, tmp_path):
        _write_synthetic_chexpert(tmp_path, n_patients=30)
        out = tmp_path / "processed"
        build_processed_dataset(tmp_path, _spec(), out, seed=0, check_images=False)
        manifest, meta = load_manifest(out / "manifest.parquet")
        assert manifest.attrs["target"] == "Pneumonia"
        assert manifest.attrs["concepts"] == CONCEPT_COLS
        assert meta["concept_spec"] == "chexpert_v1"
        assert meta["views"] == ["AP", "PA"]

    def test_load_manifest_requires_its_sidecar(self, tmp_path):
        df = pd.DataFrame({"patient_id": ["p1"], "path": ["x"], "target": [1]})
        p = tmp_path / "m.parquet"
        df.to_parquet(p, index=False)  # deliberately no sidecar
        with pytest.raises(FileNotFoundError, match="sidecar"):
            load_manifest(p)


class TestCLI:
    def test_exit_2_when_dataset_absent(self, tmp_path, capsys):
        from scripts.get_data import main

        rc = main(["--dest", str(tmp_path), "--out", str(tmp_path / "o")])
        assert rc == 2
        assert "Stanford AIMI" in capsys.readouterr().err

    def test_happy_path_exits_0_and_writes_artifacts(self, tmp_path):
        from scripts.get_data import main

        _write_synthetic_chexpert(tmp_path, n_patients=80)
        out = tmp_path / "processed"
        rc = main(["--dest", str(tmp_path), "--out", str(out), "--no-check-images", "--seed", "42"])
        assert rc == 0
        assert (out / "manifest.parquet").exists()
        assert (out / "splits.parquet").exists()
        assert (out / "external.parquet").exists()

    def test_json_summary_is_written_when_requested(self, tmp_path):
        from scripts.get_data import main

        _write_synthetic_chexpert(tmp_path, n_patients=60)
        out = tmp_path / "processed"
        report = tmp_path / "run.json"
        rc = main(
            [
                "--dest",
                str(tmp_path),
                "--out",
                str(out),
                "--no-check-images",
                "--json",
                str(report),
            ]
        )
        assert rc == 0
        assert report.exists()
        import json

        payload = json.loads(report.read_text())
        assert payload["manifest_path"].endswith("manifest.parquet")
        assert "split_report" in payload
