"""Run artifacts and the Invariant 6 enforcement point (ADR-010).

Every run writes ``experiments/runs/<run_id>/`` with ``config.yaml``, ``metrics.json``,
``predictions.parquet``, ``concept_weights.json`` and a gitignored ``checkpoint.pt``, plus a
row in the committed ``experiments/runs_index.csv``. The Streamlit app reads these files, so
results can never drift out of sync with the app.

:func:`log_run` refuses to mark a run ``complete`` unless ``metrics.json`` carries all eight
metric families (predictive, calibration, concept quality, intervention, leakage,
completeness, faithfulness, stability). A run that is missing any -- e.g. a Phase-3 partial
run before the Phase-5 protocol exists -- is written and labelled ``partial``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ddera.config import RUNS_ROOT

REQUIRED_METRIC_FAMILIES: tuple[str, ...] = (
    "predictive",
    "calibration",
    "concept_quality",
    "intervention",
    "leakage",
    "completeness",
    "faithfulness",
    "stability",
)

_VARIANT_RUN_PREFIX = {
    "b0": "blackbox_b0",
    "m1_independent": "cbm_independent",
    "m2_sequential": "cbm_sequential",
    "m3_joint": "cbm_joint",
    "m4_hybrid": "cbm_hybrid",
}


class IncompleteRunError(RuntimeError):
    """Raised when a run is marked complete without all eight metric families."""


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def config_yaml(self) -> Path:
        return self.root / "config.yaml"

    @property
    def metrics_json(self) -> Path:
        return self.root / "metrics.json"

    @property
    def predictions_parquet(self) -> Path:
        return self.root / "predictions.parquet"

    @property
    def concept_weights_json(self) -> Path:
        return self.root / "concept_weights.json"

    @property
    def checkpoint_pt(self) -> Path:
        return self.root / "checkpoint.pt"


def new_run_id(variant: str, *, now: datetime | None = None) -> str:
    """``{family}_{variant}_{YYYYMMDD-HHMMSS}`` (CLAUDE.md section 5)."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    return f"{_VARIANT_RUN_PREFIX.get(variant, variant)}_{stamp}"


def missing_families(metrics: dict[str, Any]) -> list[str]:
    """Families absent or still marked ``pending``."""
    out = []
    for fam in REQUIRED_METRIC_FAMILIES:
        value = metrics.get(fam)
        if not isinstance(value, dict) or value.get("status") == "pending":
            out.append(fam)
    return out


def log_run(
    run_dir: str | Path,
    *,
    config: dict[str, Any],
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    concept_weights: dict[str, Any],
    checkpoint: Any | None = None,
    partial: bool = False,
    index_csv: str | Path | None = None,
) -> RunPaths:
    """Write the five artifacts and append the index row.

    Raises :class:`IncompleteRunError` when ``partial=False`` and any of the eight families
    is missing.
    """
    absent = missing_families(metrics)
    if absent and not partial:
        raise IncompleteRunError(
            f"Run cannot be marked complete: missing metric families {absent}. "
            "Pass partial=True to write it as a partial run (Invariant 6)."
        )

    paths = RunPaths(Path(run_dir))
    paths.root.mkdir(parents=True, exist_ok=True)

    metrics = {**metrics, "status": "partial" if (absent or partial) else "complete"}
    paths.config_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    paths.metrics_json.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    predictions.to_parquet(paths.predictions_parquet, index=False)
    paths.concept_weights_json.write_text(
        json.dumps(concept_weights, indent=2, default=str), encoding="utf-8"
    )
    if checkpoint is not None:
        import torch

        torch.save(checkpoint, paths.checkpoint_pt)

    _append_index(paths.root, config, metrics, index_csv)
    return paths


def _append_index(
    run_root: Path, config: dict[str, Any], metrics: dict[str, Any], index_csv: str | Path | None
) -> None:
    index_path = Path(index_csv) if index_csv is not None else RUNS_ROOT / "runs_index.csv"
    predictive = (
        metrics.get("predictive", {}) if isinstance(metrics.get("predictive"), dict) else {}
    )
    row = {
        "run_id": run_root.name,
        "variant": config.get("model", {}).get("variant", config.get("variant", "")),
        "status": metrics.get("status", "partial"),
        "split": metrics.get("split", ""),
        "auroc": predictive.get("auroc", ""),
        "auroc_ci_width": predictive.get("auroc_ci_width", ""),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "path": str(run_root),
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if index_path.exists():
        frame = pd.concat([pd.read_csv(index_path), frame], ignore_index=True)
    frame.to_csv(index_path, index=False)


def load_run(run_dir: str | Path) -> dict[str, Any]:
    paths = RunPaths(Path(run_dir))
    if not paths.metrics_json.exists():
        raise FileNotFoundError(f"No metrics.json under {paths.root}")
    out: dict[str, Any] = {
        "run_id": paths.root.name,
        "config": yaml.safe_load(paths.config_yaml.read_text(encoding="utf-8")),
        "metrics": json.loads(paths.metrics_json.read_text(encoding="utf-8")),
    }
    if paths.concept_weights_json.exists():
        out["concept_weights"] = json.loads(paths.concept_weights_json.read_text(encoding="utf-8"))
    if paths.predictions_parquet.exists():
        out["predictions"] = pd.read_parquet(paths.predictions_parquet)
    return out


def list_runs(runs_root: str | Path | None = None) -> pd.DataFrame:
    """The committed index, or an empty frame with the right columns."""
    index_path = Path(runs_root or RUNS_ROOT) / "runs_index.csv"
    if index_path.exists():
        return pd.read_csv(index_path)
    return pd.DataFrame(
        columns=[
            "run_id",
            "variant",
            "status",
            "split",
            "auroc",
            "auroc_ci_width",
            "timestamp",
            "path",
        ]
    )


def compare_runs(run_dirs: list[str | Path]) -> pd.DataFrame:
    rows = []
    for d in run_dirs:
        run = load_run(d)
        predictive = run["metrics"].get("predictive", {})
        rows.append(
            {
                "run_id": run["run_id"],
                "variant": run["config"].get("model", {}).get("variant", ""),
                "status": run["metrics"].get("status"),
                "auroc": predictive.get("auroc"),
                "auroc_ci": predictive.get("auroc_ci"),
                "brier": run["metrics"].get("calibration", {}).get("brier"),
                "ece": run["metrics"].get("calibration", {}).get("ece"),
            }
        )
    return pd.DataFrame(rows)
