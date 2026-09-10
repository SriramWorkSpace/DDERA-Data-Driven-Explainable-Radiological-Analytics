"""Run artifacts + the Invariant 6 enforcement (ADR-010).

The load-bearing test: ``log_run`` refuses to mark a run ``complete`` unless all eight
metric families are present, but writes a labelled ``partial`` run when told to.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ddera.reporting.runs import (
    REQUIRED_METRIC_FAMILIES,
    IncompleteRunError,
    compare_runs,
    list_runs,
    load_run,
    log_run,
    missing_families,
    new_run_id,
)


def _family_stub() -> dict:
    return {"value": 1.0}


def _all_eight() -> dict:
    return {"split": "test", **{fam: _family_stub() for fam in REQUIRED_METRIC_FAMILIES}}


def _predictions() -> pd.DataFrame:
    return pd.DataFrame({"row": [0, 1, 2], "target": [0, 1, 0], "target_prob": [0.2, 0.8, 0.4]})


class TestRunIdAndCompleteness:
    def test_run_id_shape(self):
        rid = new_run_id("m2_sequential")
        assert rid.startswith("cbm_sequential_")
        assert len(rid.split("_")[-1]) == 15  # YYYYMMDD-HHMMSS

    def test_missing_families_flags_absent_and_pending(self):
        metrics = _all_eight()
        metrics.pop("stability")
        metrics["leakage"] = {"status": "pending"}
        assert set(missing_families(metrics)) == {"stability", "leakage"}


class TestLogRun:
    def test_partial_run_writes_all_five_artifacts_and_indexes(self, tmp_path):
        metrics = {"split": "val", "predictive": {"auroc": 0.71}, "calibration": {"ece": 0.05}}
        paths = log_run(
            tmp_path / "run",
            config={"model": {"variant": "m2_sequential"}},
            metrics=metrics,
            predictions=_predictions(),
            concept_weights={"weights": [0.1] * 12, "bias": 0.0},
            checkpoint={"w": 1},
            partial=True,
            index_csv=tmp_path / "runs_index.csv",
        )
        for p in (
            paths.config_yaml,
            paths.metrics_json,
            paths.predictions_parquet,
            paths.concept_weights_json,
            paths.checkpoint_pt,
        ):
            assert p.exists()
        index = pd.read_csv(tmp_path / "runs_index.csv")
        assert index.loc[0, "status"] == "partial"
        assert index.loc[0, "variant"] == "m2_sequential"

    def test_incomplete_run_marked_complete_raises(self, tmp_path):
        metrics = _all_eight()
        metrics.pop("stability")
        with pytest.raises(IncompleteRunError, match="stability"):
            log_run(
                tmp_path / "run",
                config={"variant": "m2_sequential"},
                metrics=metrics,
                predictions=_predictions(),
                concept_weights={},
                partial=False,
                index_csv=tmp_path / "idx.csv",
            )

    def test_complete_run_with_all_eight_families_is_accepted(self, tmp_path):
        paths = log_run(
            tmp_path / "run",
            config={"model": {"variant": "m2_sequential"}},
            metrics=_all_eight(),
            predictions=_predictions(),
            concept_weights={},
            partial=False,
            index_csv=tmp_path / "idx.csv",
        )
        run = load_run(paths.root)
        assert run["metrics"]["status"] == "complete"

    def test_load_and_compare_round_trip(self, tmp_path):
        a = log_run(
            tmp_path / "a",
            config={"model": {"variant": "m2_sequential"}},
            metrics={"split": "val", "predictive": {"auroc": 0.7}, "calibration": {"brier": 0.2}},
            predictions=_predictions(),
            concept_weights={"weights": [0.0] * 12, "bias": 0.1},
            partial=True,
            index_csv=tmp_path / "runs_index.csv",
        )
        b = log_run(
            tmp_path / "b",
            config={"model": {"variant": "b0"}},
            metrics={"split": "val", "predictive": {"auroc": 0.8}, "calibration": {"brier": 0.15}},
            predictions=_predictions(),
            concept_weights={},
            partial=True,
            index_csv=tmp_path / "runs_index.csv",
        )
        loaded = load_run(a.root)
        assert loaded["concept_weights"]["bias"] == 0.1
        assert "predictions" in loaded

        table = compare_runs([a.root, b.root])
        assert set(table["variant"]) == {"m2_sequential", "b0"}
        assert list_runs(tmp_path).shape[0] == 2
