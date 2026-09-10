"""CheXpert manifest construction.

Turns the raw ``train.csv`` / ``valid.csv`` into a typed manifest the rest of the pipeline
consumes, with patient and study identifiers parsed out of the path so splitting can happen
at patient level (ADR-005).

CheXpert path layout::

    CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg
                              ^^^^^^^^^^^^ ^^^^^^ ^^^^^^^^^^^^^^^^
                              patient      study  view file

The parsing is deliberately strict. A silently mis-parsed patient ID would put the same
patient in two splits and inflate every metric in the project, so a malformed path raises
rather than falling back to something plausible.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from ddera.data.labels import CHEXPERT_OBSERVATIONS

PATH_PATTERN = re.compile(
    r"(?P<patient_id>patient\d+)[/\\](?P<study>study\d+)[/\\](?P<view_file>[^/\\]+)$"
)

META_COLUMNS = ("Path", "Sex", "Age", "Frontal/Lateral", "AP/PA")
FRONTAL_VIEWS = ("AP", "PA")


@dataclass(frozen=True)
class ManifestSummary:
    """Counts produced while building a manifest, for the EDA notebook and the run log."""

    n_rows_raw: int
    n_rows_kept: int
    n_patients: int
    n_studies: int
    view_counts: dict[str, int]
    dropped_by_view: int
    dropped_by_missing_target: int

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def summary(self) -> str:
        return (
            f"rows {self.n_rows_raw:,} -> {self.n_rows_kept:,}  "
            f"({self.n_patients:,} patients, {self.n_studies:,} studies)\n"
            f"  dropped: {self.dropped_by_view:,} by view, "
            f"{self.dropped_by_missing_target:,} by missing target\n"
            f"  views: {self.view_counts}"
        )


def parse_chexpert_path(path: str) -> dict[str, str]:
    """Extract patient, study and view-file identifiers from a CheXpert image path.

    Raises:
        ValueError: if the path does not match the expected layout. Guessing here would
            risk patient leakage, so we refuse instead.
    """
    match = PATH_PATTERN.search(str(path))
    if not match:
        raise ValueError(
            f"Path does not match the CheXpert layout "
            f"(.../patientNNNNN/studyN/viewN_*.jpg): {path!r}"
        )
    parts = match.groupdict()
    return {
        "patient_id": parts["patient_id"],
        "study": parts["study"],
        "study_id": f"{parts['patient_id']}/{parts['study']}",
        "view_file": parts["view_file"],
    }


def build_manifest(
    csv_path: str | Path,
    *,
    target: str = "Pneumonia",
    concepts: list[str] | None = None,
    views: tuple[str, ...] | list[str] = FRONTAL_VIEWS,
    drop_missing_target: bool = False,
) -> tuple[pd.DataFrame, ManifestSummary]:
    """Read a CheXpert CSV into a manifest.

    Args:
        csv_path: path to ``train.csv`` or ``valid.csv``.
        target: the target observation column.
        concepts: concept columns to retain. Defaults to all observations except the target
            and ``No Finding`` (both excluded by ADR-003).
        views: which ``AP/PA`` values to keep. Defaults to frontal only.
        drop_missing_target: drop rows whose target is blank. Left ``False`` by default
            because the blank-handling policy belongs to
            :mod:`ddera.data.labels`, not here -- this function does not silently apply a
            label policy.

    Returns:
        ``(manifest, summary)``. Raw label values (1/0/-1/NaN) are preserved untouched;
        applying an uncertainty policy is a separate, explicit step.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CheXpert CSV not found: {csv_path}")

    raw = pd.read_csv(csv_path)
    return manifest_from_frame(
        raw,
        target=target,
        concepts=concepts,
        views=views,
        drop_missing_target=drop_missing_target,
    )


def manifest_from_frame(
    raw: pd.DataFrame,
    *,
    target: str = "Pneumonia",
    concepts: list[str] | None = None,
    views: tuple[str, ...] | list[str] = FRONTAL_VIEWS,
    drop_missing_target: bool = False,
) -> tuple[pd.DataFrame, ManifestSummary]:
    """Manifest construction from an already-loaded frame.

    Split out from :func:`build_manifest` so the parsing and filtering logic is testable
    without the 11 GB dataset present.
    """
    if "Path" not in raw.columns:
        raise ValueError(f"Expected a 'Path' column. Got: {list(raw.columns)}")
    if target not in raw.columns:
        raise ValueError(f"Target column {target!r} not found. Got: {list(raw.columns)}")

    if concepts is None:
        concepts = [c for c in CHEXPERT_OBSERVATIONS if c not in (target, "No Finding")]
    missing = [c for c in concepts if c not in raw.columns]
    if missing:
        raise ValueError(f"Concept columns not found in the CSV: {missing}")

    n_raw = len(raw)
    parsed = pd.DataFrame([parse_chexpert_path(p) for p in raw["Path"]], index=raw.index)

    manifest = pd.concat(
        [
            parsed,
            raw[["Path"]].rename(columns={"Path": "path"}),
            raw[[c for c in ("Sex", "Age", "Frontal/Lateral", "AP/PA") if c in raw.columns]],
        ],
        axis=1,
    ).rename(
        columns={"Frontal/Lateral": "orientation", "AP/PA": "view", "Sex": "sex", "Age": "age"}
    )

    manifest[f"raw_{target}"] = raw[target].to_numpy()
    for concept in concepts:
        manifest[f"raw_{concept}"] = raw[concept].to_numpy()

    view_counts_before = (
        manifest["view"].fillna("unknown").value_counts().to_dict()
        if "view" in manifest.columns
        else {}
    )

    dropped_by_view = 0
    if views and "view" in manifest.columns:
        before = len(manifest)
        manifest = manifest[manifest["view"].isin(list(views))].copy()
        dropped_by_view = before - len(manifest)

    dropped_by_missing_target = 0
    if drop_missing_target:
        before = len(manifest)
        manifest = manifest[manifest[f"raw_{target}"].notna()].copy()
        dropped_by_missing_target = before - len(manifest)

    manifest = manifest.reset_index(drop=True)
    manifest.attrs["target"] = target
    manifest.attrs["concepts"] = list(concepts)

    summary = ManifestSummary(
        n_rows_raw=n_raw,
        n_rows_kept=len(manifest),
        n_patients=int(manifest["patient_id"].nunique()),
        n_studies=int(manifest["study_id"].nunique()),
        view_counts=view_counts_before,
        dropped_by_view=dropped_by_view,
        dropped_by_missing_target=dropped_by_missing_target,
    )
    return manifest, summary


def concept_matrix(manifest: pd.DataFrame, concepts: list[str]) -> np.ndarray:
    """Extract the ``(n, k)`` raw concept matrix (values still 1/0/-1/NaN)."""
    missing = [c for c in concepts if f"raw_{c}" not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest is missing raw columns for: {missing}")
    return manifest[[f"raw_{c}" for c in concepts]].to_numpy(dtype=float)


def target_vector(manifest: pd.DataFrame, target: str) -> np.ndarray:
    """Extract the raw target vector (values still 1/0/-1/NaN)."""
    column = f"raw_{target}"
    if column not in manifest.columns:
        raise ValueError(f"Manifest is missing {column!r}.")
    return manifest[column].to_numpy(dtype=float)


def cooccurrence_matrix(manifest: pd.DataFrame, observations: list[str]) -> pd.DataFrame:
    """Pairwise positive co-occurrence counts.

    A core EDA output: concepts that almost always co-occur cannot be independently
    intervened upon in any meaningful way, which is directly relevant to how the Phase 5
    intervention results should be read.
    """
    columns = [f"raw_{o}" for o in observations]
    missing = [c for c in columns if c not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest is missing raw columns for: {missing}")

    positives = (manifest[columns] == 1.0).astype(int).to_numpy()
    counts = positives.T @ positives
    return pd.DataFrame(counts, index=observations, columns=observations)


# ---------------------------------------------------------------------------------------
# Image probing
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageProbeReport:
    """Result of checking that each manifest row points at a readable image."""

    n_checked: int
    n_missing: int
    n_unreadable: int
    dimensions_available: bool
    missing_paths: list[str]
    unreadable_paths: list[str]

    def to_dict(self) -> dict[str, Any]:
        # Cap the path lists so a wholly-broken download does not write a 200k-line sidecar.
        return {
            "n_checked": self.n_checked,
            "n_missing": self.n_missing,
            "n_unreadable": self.n_unreadable,
            "dimensions_available": self.dimensions_available,
            "missing_paths": self.missing_paths[:50],
            "unreadable_paths": self.unreadable_paths[:50],
        }

    def summary(self) -> str:
        return (
            f"image probe: {self.n_checked:,} checked, {self.n_missing:,} missing, "
            f"{self.n_unreadable:,} unreadable"
        )


def probe_image_dimensions(
    manifest: pd.DataFrame,
    image_root: str | Path | None,
    *,
    path_column: str = "path",
) -> tuple[pd.DataFrame, ImageProbeReport]:
    """Resolve each row's image against ``image_root``, record ``width``/``height``, and
    flag files that are missing or cannot be decoded.

    ``image_root`` is the directory the CheXpert ``Path`` values are relative to (the parent
    of ``CheXpert-v1.0-small/``). When it is ``None`` the probe is skipped and the manifest
    is returned unchanged with ``dimensions_available=False`` -- the path a CSV-only
    checkout and the tests take.

    Only the image header is parsed (plus a structural ``verify()``); pixels are never
    decoded, so this stays cheap over the full dataset.
    """
    out = manifest.copy()
    if image_root is None:
        return out, ImageProbeReport(0, 0, 0, False, [], [])

    root = Path(image_root)
    widths: list[int | None] = []
    heights: list[int | None] = []
    missing: list[str] = []
    unreadable: list[str] = []

    for rel in out[path_column].astype(str):
        fpath = root / rel
        if not fpath.is_file():
            missing.append(rel)
            widths.append(None)
            heights.append(None)
            continue
        try:
            with Image.open(fpath) as im:
                im.verify()
            with Image.open(fpath) as im:
                width, height = im.size
        except Exception:  # noqa: BLE001 - any decode failure means the file is unusable
            unreadable.append(rel)
            widths.append(None)
            heights.append(None)
            continue
        widths.append(int(width))
        heights.append(int(height))

    out["width"] = pd.array(widths, dtype="Int64")
    out["height"] = pd.array(heights, dtype="Int64")
    return out, ImageProbeReport(
        n_checked=len(out),
        n_missing=len(missing),
        n_unreadable=len(unreadable),
        dimensions_available=True,
        missing_paths=missing,
        unreadable_paths=unreadable,
    )


# ---------------------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------------------


def write_manifest(
    manifest: pd.DataFrame,
    path: str | Path,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write ``manifest.parquet`` plus a JSON sidecar.

    The sidecar carries the target/concept contract (``manifest.attrs`` is *not* preserved
    by parquet) and any extra ``meta`` -- summaries, provenance, image report.
    :func:`load_manifest` restores the contract from it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(path, index=False)

    sidecar: dict[str, Any] = {
        "target": manifest.attrs.get("target"),
        "concepts": list(manifest.attrs.get("concepts", [])),
        "n_rows": int(len(manifest)),
        "columns": list(manifest.columns),
    }
    if meta:
        sidecar.update(meta)
    path.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2, default=str), encoding="utf-8"
    )
    return path


def load_manifest(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read a manifest written by :func:`write_manifest`, restoring ``attrs['target']`` and
    ``attrs['concepts']`` from the sidecar. Raises if the sidecar is absent -- the manifest
    is unusable without the concept/target contract.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest parquet not found: {path}")
    sidecar_path = path.with_suffix(".json")
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"Manifest sidecar not found: {sidecar_path}. It carries the target/concept "
            "contract that parquet cannot store; the manifest is unusable without it."
        )
    manifest = pd.read_parquet(path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    manifest.attrs["target"] = sidecar.get("target")
    manifest.attrs["concepts"] = list(sidecar.get("concepts", []))
    return manifest, sidecar
