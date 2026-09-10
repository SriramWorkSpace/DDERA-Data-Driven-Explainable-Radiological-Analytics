"""Load run artifacts for the app. Thin wrapper over :mod:`ddera.reporting.runs`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ddera.config import RUNS_ROOT
from ddera.reporting.runs import list_runs, load_run
from ddera.xai.intervention import LinearReasoner


def available_runs(runs_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Every run in the index that still has a metrics.json on disk, newest first."""
    root = Path(runs_root or RUNS_ROOT)
    index = list_runs(root)
    runs: list[dict[str, Any]] = []
    for _, entry in index.iterrows():
        run_dir = Path(entry.get("path", root / str(entry["run_id"])))
        if not run_dir.is_absolute():
            run_dir = root / run_dir
        if (run_dir / "metrics.json").exists():
            try:
                runs.append(load_run(run_dir))
            except (FileNotFoundError, ValueError):
                continue
    runs.sort(key=lambda r: r["run_id"].split("_")[-1], reverse=True)
    return runs


def latest_run(runs_root: str | Path | None = None) -> dict[str, Any] | None:
    runs = available_runs(runs_root)
    return runs[0] if runs else None


def run_is_synthetic(run: dict[str, Any]) -> bool:
    return bool(run.get("metrics", {}).get("synthetic")) or bool(
        run.get("config", {}).get("_synthetic")
    )


def load_reasoner(run: dict[str, Any]) -> tuple[LinearReasoner, list[str]] | None:
    """Rebuild the interpretable reasoner from ``concept_weights.json``. ``None`` for B0."""
    weights = run.get("concept_weights") or {}
    if not weights.get("weights"):
        return None
    names = weights.get("concept_names") or [f"concept_{i}" for i in range(len(weights["weights"]))]
    return LinearReasoner(
        np.asarray(weights["weights"], dtype=float), float(weights["bias"]), names
    ), names


def sample_concepts(run: dict[str, Any], row: int) -> np.ndarray | None:
    """The predicted concept vector for one row of ``predictions.parquet``."""
    preds = run.get("predictions")
    if preds is None:
        return None
    cols = [c for c in preds.columns if c.startswith("concept_prob__")]
    if not cols or row >= len(preds):
        return None
    return preds.loc[row, cols].to_numpy(dtype=float)
