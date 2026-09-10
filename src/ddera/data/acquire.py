"""Phase-1 data acquisition orchestration (ADR-002/003/004/005/011).

Ties the already-tested building blocks together into the pipeline stage the project plan
calls "01 -- acquisition":

    a locally-obtained CheXpert download
        -> manifest.parquet          (ddera.data.chexpert)
        -> patient-level splits.parquet   (ddera.data.splits)

This module does **not** download CheXpert. CheXpert requires a Stanford AIMI research-use
agreement (ADR-002); the user obtains and extracts it, then points ``--dest`` at the
directory. :func:`inspect_download` verifies the expected structure and reports what it
finds.

Which ADR contributes what here:

* **ADR-002** -- CheXpert is the source; ``valid.csv`` (234-study radiologist consensus) is
  written as a separate external set (``external.parquet``) and never split.
* **ADR-003** -- the target is ``ConceptSpec.target``. The escalation trigger
  (test positives < ``min_test_positives``) is *evaluated and reported*, never acted on --
  there is no trained model yet, so the AUROC-CI half of the trigger cannot be computed.
* **ADR-004** -- the target column has ``u_ignore`` + blank->negative applied. That drops
  rows, so it is a cohort decision and belongs here. Concept labels are persisted **raw**;
  their uncertainty policy is applied later at training time and must stay configurable
  (u_mask primary, u_zeros / u_ones as the sensitivity analysis).
* **ADR-005** -- frontal views only, patient-level split, ``valid.csv`` held out.
* **ADR-011** -- split integrity is judged by :func:`check_split_integrity`'s z-score test,
  not a flat prevalence tolerance.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ddera.config import ConceptSpec
from ddera.data.chexpert import (
    ImageProbeReport,
    ManifestSummary,
    build_manifest,
    probe_image_dimensions,
    write_manifest,
)
from ddera.data.labels import apply_uncertainty_policy, rows_to_keep
from ddera.data.splits import (
    SplitReport,
    check_split_integrity,
    patient_level_split,
    write_splits,
)

CHEXPERT_SUBDIR = "CheXpert-v1.0-small"

ACQUISITION_HELP = (
    "CheXpert is not bundled with DDERA and this tool does not download it. Accept the "
    "Stanford AIMI research-use agreement at "
    "https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2 , "
    "download and extract CheXpert-v1.0-small, then pass its location with --dest."
)


# ---------------------------------------------------------------------------------------
# Download inspection
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CheXpertLayout:
    """What :func:`inspect_download` found under a candidate CheXpert directory."""

    root: Path
    train_csv: Path | None
    valid_csv: Path | None
    image_root: Path | None  # directory the CSV 'Path' values resolve against

    @property
    def ok(self) -> bool:
        return self.train_csv is not None

    @property
    def has_images(self) -> bool:
        return self.image_root is not None

    def describe(self) -> str:
        return "\n".join(
            [
                f"CheXpert download @ {self.root}",
                f"  train.csv : {self.train_csv or 'NOT FOUND'}",
                f"  valid.csv : {self.valid_csv or 'not found (external eval set skipped)'}",
                f"  images    : {self.image_root or 'not found (CSV-only; probing disabled)'}",
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "train_csv": str(self.train_csv) if self.train_csv else None,
            "valid_csv": str(self.valid_csv) if self.valid_csv else None,
            "image_root": str(self.image_root) if self.image_root else None,
            "has_images": self.has_images,
        }


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.is_file():
            return p
    return None


def inspect_download(dest: str | Path) -> CheXpertLayout:
    """Locate ``train.csv`` / ``valid.csv`` and the image tree under ``dest``.

    Accepts either the directory that directly contains ``train.csv`` or one that contains a
    ``CheXpert-v1.0-small/`` subdirectory. Never downloads or writes anything.
    """
    root = Path(dest)
    candidates = [root, root / CHEXPERT_SUBDIR]

    train_csv = _first_existing(c / "train.csv" for c in candidates)
    valid_csv = _first_existing(c / "valid.csv" for c in candidates)

    # CSV 'Path' values look like 'CheXpert-v1.0-small/train/patientNNNNN/...', so the image
    # root is whichever directory makes that relative path resolve.
    image_root: Path | None = None
    if (root / CHEXPERT_SUBDIR / "train").is_dir():
        image_root = root
    elif root.name == CHEXPERT_SUBDIR and (root / "train").is_dir():
        image_root = root.parent

    return CheXpertLayout(root, train_csv, valid_csv, image_root)


# ---------------------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------------------


@dataclass
class ProcessedArtifacts:
    """Everything :func:`build_processed_dataset` produced, for the CLI and the EDA notebook."""

    manifest_path: Path
    splits_path: Path
    external_path: Path | None
    manifest_summary: ManifestSummary
    split_report: SplitReport
    integrity: dict[str, Any]
    image_report: ImageProbeReport
    escalation: dict[str, Any]
    n_rows_after_target_policy: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "splits_path": str(self.splits_path),
            "external_path": str(self.external_path) if self.external_path else None,
            "manifest_summary": self.manifest_summary.to_dict(),
            "split_report": self.split_report.to_dict(),
            "integrity": self.integrity,
            "image_report": self.image_report.to_dict(),
            "escalation": self.escalation,
            "n_rows_after_target_policy": self.n_rows_after_target_policy,
        }

    def summary(self) -> str:
        lines = [
            f"manifest -> {self.manifest_path}  "
            f"({self.n_rows_after_target_policy:,} rows after the target policy)",
            f"splits   -> {self.splits_path}",
        ]
        if self.external_path:
            lines.append(f"external -> {self.external_path}  (valid.csv; held out, never split)")
        lines += ["", self.split_report.summary(), ""]
        lines.append(
            self.image_report.summary()
            if self.image_report.dimensions_available
            else "image probe: skipped (no image tree or --no-check-images)"
        )
        lines.append("")
        lines.append(
            f"ADR-003 escalation: {self.escalation['n_test_positives']} test positives "
            f"(trigger < {self.escalation['min_test_positives']}) -> "
            f"{'WOULD ESCALATE (advisory)' if self.escalation['would_escalate'] else 'ok'}"
        )
        if not self.integrity["leakage_free"]:
            lines.append("PATIENT LEAKAGE: " + "; ".join(self.integrity["issues"]))
        return "\n".join(lines)


def _binarize_target(frame: Any, target: str, spec: ConceptSpec) -> Any:
    """Apply ADR-004's target policy: drop uncertain-target rows, then map to 0/1."""
    raw = frame[f"raw_{target}"].to_numpy()
    keep = rows_to_keep(
        raw,
        policy=spec.uncertainty.target_policy,
        blank_policy=spec.uncertainty.blank_policy,
    )
    frame = frame.loc[keep].reset_index(drop=True)
    labels, _ = apply_uncertainty_policy(
        frame[f"raw_{target}"].to_numpy(),
        policy=spec.uncertainty.target_policy,
        blank_policy=spec.uncertainty.blank_policy,
    )
    frame["target"] = labels.astype(int)
    return frame


def build_processed_dataset(
    dest: str | Path,
    spec: ConceptSpec,
    out_dir: str | Path,
    *,
    seed: int = 42,
    check_images: bool | None = None,
    drop_unreadable: bool = False,
    include_valid: bool = True,
) -> ProcessedArtifacts:
    """Run the Phase-1 pipeline end to end and write the artifacts.

    Args:
        dest: an extracted CheXpert directory (see :func:`inspect_download`).
        spec: the concept/target contract, from ``configs/concepts/chexpert_v1.yaml``.
        out_dir: where ``manifest.parquet`` / ``splits.parquet`` (+ JSON sidecars) go.
        seed: patient-split seed.
        check_images: probe image files. ``None`` = auto (on when an image tree is present).
        drop_unreadable: drop rows whose image is missing/corrupt (always reported either way).
        include_valid: also write ``external.parquet`` from ``valid.csv`` when present.

    Raises:
        FileNotFoundError: no ``train.csv`` under ``dest``. Message includes acquisition help.
        RuntimeError: patient leakage detected -- no artifacts are written.
    """
    layout = inspect_download(dest)
    if not layout.ok:
        raise FileNotFoundError(f"No train.csv found under {Path(dest)!r}. {ACQUISITION_HELP}")

    out_dir = Path(out_dir)
    target = spec.target
    concepts = list(spec.concepts)

    manifest, summary = build_manifest(
        layout.train_csv,
        target=target,
        concepts=concepts,
        views=spec.cohort.views,
    )
    manifest = _binarize_target(manifest, target, spec)
    n_after_policy = len(manifest)

    do_check = layout.has_images if check_images is None else check_images
    manifest, image_report = probe_image_dimensions(
        manifest, layout.image_root if do_check else None
    )
    if do_check and drop_unreadable and (image_report.n_missing or image_report.n_unreadable):
        bad = set(image_report.missing_paths) | set(image_report.unreadable_paths)
        manifest = manifest.loc[~manifest["path"].isin(bad)].reset_index(drop=True)

    manifest.attrs["target"] = target
    manifest.attrs["concepts"] = concepts
    manifest_path = write_manifest(
        manifest,
        out_dir / "manifest.parquet",
        meta={
            "source_csv": str(layout.train_csv),
            "concept_spec": spec.name,
            "views": list(spec.cohort.views),
            "uncertainty": dict(vars(spec.uncertainty)),
            "manifest_summary": summary.to_dict(),
            "n_rows_after_target_policy": n_after_policy,
            "image_report": image_report.to_dict(),
        },
    )

    split_df, split_report = patient_level_split(
        manifest,
        patient_col="patient_id",
        target_col="target",
        ratios=dict(spec.cohort.split_ratios),
        seed=seed,
    )
    integrity = check_split_integrity(split_df, target_col="target")
    if not integrity["leakage_free"]:
        raise RuntimeError(
            "Patient leakage detected in the split -- refusing to write splits.parquet:\n  "
            + "\n  ".join(integrity["issues"])
        )
    splits_path = write_splits(
        split_df, out_dir / "splits.parquet", report=split_report, integrity=integrity
    )

    n_test_pos = int(split_df.loc[split_df["split"] == "test", "target"].sum())
    would_escalate, reason = spec.escalation.should_escalate(n_test_pos, auroc_ci_width=0.0)
    escalation = {
        "n_test_positives": n_test_pos,
        "min_test_positives": spec.escalation.min_test_positives,
        "would_escalate": bool(would_escalate),
        "reason": reason,
        "note": "Advisory. The AUROC-CI half of the ADR-003 trigger needs a trained model.",
    }

    external_path: Path | None = None
    if include_valid and layout.valid_csv is not None:
        external, _ = build_manifest(
            layout.valid_csv, target=target, concepts=concepts, views=spec.cohort.views
        )
        external = _binarize_target(external, target, spec)
        external.attrs["target"] = target
        external.attrs["concepts"] = concepts
        external_path = write_manifest(
            external,
            out_dir / "external.parquet",
            meta={
                "source_csv": str(layout.valid_csv),
                "concept_spec": spec.name,
                "role": (
                    "held-out external evaluation set (ADR-005): the official 234-study "
                    "radiologist-consensus set. Never split, never trained on."
                ),
            },
        )

    return ProcessedArtifacts(
        manifest_path=manifest_path,
        splits_path=splits_path,
        external_path=external_path,
        manifest_summary=summary,
        split_report=split_report,
        integrity=integrity,
        image_report=image_report,
        escalation=escalation,
        n_rows_after_target_policy=n_after_policy,
    )
