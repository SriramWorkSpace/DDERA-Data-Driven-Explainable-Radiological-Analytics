"""Patient-level dataset splitting (ADR-005).

The single most common source of overstated results in medical imaging is splitting on
images rather than patients. One patient contributes several studies and views; if those
land in both train and test, the model can recognise the patient rather than the pathology,
and every metric is inflated.

DDERA therefore splits on **patients**, never images, and
:func:`assert_no_patient_leakage` is asserted in the test suite rather than left to
discipline.

Stratification is done at patient level on a patient-level summary of the target: a patient
counts as positive if *any* of their studies is positive. This keeps prevalence comparable
across splits without ever letting a patient straddle two of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

DEFAULT_RATIOS: dict[str, float] = {"train": 0.70, "val": 0.10, "test": 0.20}


@dataclass(frozen=True)
class SplitReport:
    """Summary of a split, for logging into the run config and the dashboard."""

    n_patients: dict[str, int]
    n_images: dict[str, int]
    prevalence: dict[str, float]
    ratios_requested: dict[str, float]
    ratios_achieved: dict[str, float]
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def summary(self) -> str:
        lines = [f"{'split':<8}{'patients':>10}{'images':>10}{'prevalence':>13}{'img frac':>10}"]
        for split in self.n_patients:
            lines.append(
                f"{split:<8}{self.n_patients[split]:>10,}{self.n_images[split]:>10,}"
                f"{self.prevalence[split]:>13.4f}{self.ratios_achieved[split]:>10.3f}"
            )
        return "\n".join(lines)


def patient_level_split(
    df: pd.DataFrame,
    *,
    patient_col: str = "patient_id",
    target_col: str = "target",
    ratios: dict[str, float] | None = None,
    seed: int = 42,
    split_col: str = "split",
) -> tuple[pd.DataFrame, SplitReport]:
    """Split a manifest into train/val/test with **no patient appearing in two splits**.

    Args:
        df: image-level manifest; one row per image.
        patient_col: column identifying the patient.
        target_col: binary target column, used only to stratify.
        ratios: patient-level proportions. Must sum to 1.
        seed: RNG seed.
        split_col: name of the column to add.

    Returns:
        ``(df_with_split_column, report)``. The input frame is not mutated.

    Raises:
        ValueError: if ratios are invalid, or a class is too rare to stratify.
    """
    ratios = dict(ratios or DEFAULT_RATIOS)
    _validate_ratios(ratios)
    for col in (patient_col, target_col):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found. Available: {list(df.columns)}")

    # Patient-level target: positive if ANY of the patient's studies is positive.
    patients = (
        df.groupby(patient_col, sort=True)[target_col]
        .max()
        .reset_index()
        .rename(columns={target_col: "patient_target"})
    )
    if len(patients) < 3:
        raise ValueError(f"Need at least 3 patients to split, got {len(patients)}.")

    counts = patients["patient_target"].value_counts()
    if len(counts) < 2:
        raise ValueError(
            f"All patients share target value {counts.index[0]}; stratified splitting is "
            "impossible. Check the target column and the uncertainty policy."
        )
    if counts.min() < 3:
        raise ValueError(
            f"Rarest class has only {counts.min()} patients; cannot stratify into 3 splits. "
            "Consider the ADR-003 escalation rule."
        )

    ids = patients[patient_col].to_numpy()
    strata = patients["patient_target"].to_numpy()

    # Two-stage: first carve off test, then split the remainder into train/val. Ratios are
    # renormalised at the second stage so the achieved proportions match the request.
    first = StratifiedShuffleSplit(n_splits=1, test_size=ratios["test"], random_state=seed)
    rest_idx, test_idx = next(first.split(ids, strata))

    val_fraction = ratios["val"] / (ratios["train"] + ratios["val"])
    second = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_rel, val_rel = next(second.split(ids[rest_idx], strata[rest_idx]))

    assignment: dict[str, str] = {}
    for pid in ids[rest_idx][train_rel]:
        assignment[pid] = "train"
    for pid in ids[rest_idx][val_rel]:
        assignment[pid] = "val"
    for pid in ids[test_idx]:
        assignment[pid] = "test"

    out = df.copy()
    out[split_col] = out[patient_col].map(assignment)

    if out[split_col].isna().any():
        n = int(out[split_col].isna().sum())
        raise RuntimeError(f"{n} rows were not assigned a split. This is a bug in splitting.")

    assert_no_patient_leakage(out, patient_col=patient_col, split_col=split_col)

    total = len(out)
    report = SplitReport(
        n_patients={s: int(g[patient_col].nunique()) for s, g in out.groupby(split_col)},
        n_images={s: int(len(g)) for s, g in out.groupby(split_col)},
        prevalence={s: float(g[target_col].mean()) for s, g in out.groupby(split_col)},
        ratios_requested=ratios,
        ratios_achieved={s: len(g) / total for s, g in out.groupby(split_col)},
        seed=seed,
    )
    return out, report


def assert_no_patient_leakage(
    df: pd.DataFrame,
    *,
    patient_col: str = "patient_id",
    split_col: str = "split",
) -> None:
    """Raise if any patient appears in more than one split.

    This is the guarantee the whole evaluation rests on. It is asserted, not assumed.
    """
    by_split = {s: set(g[patient_col]) for s, g in df.groupby(split_col)}
    names = sorted(by_split)
    problems: list[str] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = by_split[a] & by_split[b]
            if shared:
                sample = sorted(shared)[:5]
                problems.append(f"{len(shared)} patient(s) in both {a!r} and {b!r} (e.g. {sample})")
    if problems:
        raise AssertionError("PATIENT LEAKAGE DETECTED:\n  " + "\n  ".join(problems))


def check_split_integrity(
    df: pd.DataFrame,
    *,
    patient_col: str = "patient_id",
    target_col: str = "target",
    split_col: str = "split",
    z_threshold: float = 3.0,
) -> dict[str, Any]:
    """Full integrity report: leakage, coverage, and prevalence drift.

    Prevalence drift is judged **against sampling noise**, not against a flat percentage.
    A fixed relative tolerance is the wrong instrument here: a 10% val split of 300 patients
    holds ~30 patients, where ordinary binomial noise easily moves prevalence by 10-15%
    relative. A flat threshold would fire constantly on correct splits, and constant false
    alarms are how a real one gets ignored.

    So each split's prevalence is converted to a z-score against the pooled rate, using the
    **patient count** as the effective sample size rather than the image count. Images from
    one patient are correlated, so the image count would overstate the information available
    and make the test too sensitive; the patient count is the conservative choice.

    A split is flagged only when ``|z| > z_threshold`` -- i.e. the drift is larger than
    chance comfortably explains, which is what would indicate an actual stratification bug.
    """
    result: dict[str, Any] = {"leakage_free": True, "issues": []}

    try:
        assert_no_patient_leakage(df, patient_col=patient_col, split_col=split_col)
    except AssertionError as exc:
        result["leakage_free"] = False
        result["issues"].append(str(exc))

    expected = {"train", "val", "test"}
    present = set(df[split_col].dropna().unique())
    if missing := expected - present:
        result["issues"].append(f"Missing splits: {sorted(missing)}")

    grouped = df.groupby(split_col)
    prevalences = {s: float(g[target_col].mean()) for s, g in grouped}
    n_patients = {s: int(g[patient_col].nunique()) for s, g in grouped}

    result["prevalence"] = prevalences
    result["n_patients"] = n_patients
    result["n_images"] = {s: int(len(g)) for s, g in grouped}

    if len(prevalences) > 1:
        pooled = float(df[target_col].mean())
        z_scores: dict[str, float] = {}
        for split, prevalence in prevalences.items():
            effective_n = max(n_patients[split], 1)
            se = np.sqrt(max(pooled * (1.0 - pooled), 1e-12) / effective_n)
            z_scores[split] = float((prevalence - pooled) / se) if se > 0 else 0.0

        lo, hi = min(prevalences.values()), max(prevalences.values())
        result["prevalence_pooled"] = pooled
        result["prevalence_relative_drift"] = (hi - lo) / hi if hi > 0 else 0.0
        result["prevalence_z_scores"] = z_scores

        for split, z in z_scores.items():
            if abs(z) > z_threshold:
                result["issues"].append(
                    f"Split {split!r} prevalence {prevalences[split]:.4f} deviates from the "
                    f"pooled rate {pooled:.4f} by z={z:.2f} (threshold {z_threshold}), which "
                    f"is more than sampling noise over {n_patients[split]} patients explains. "
                    "Check the stratification."
                )

    result["ok"] = result["leakage_free"] and not result["issues"]
    return result


def _validate_ratios(ratios: dict[str, float]) -> None:
    required = {"train", "val", "test"}
    if set(ratios) != required:
        raise ValueError(
            f"ratios must have exactly the keys {sorted(required)}, got {sorted(ratios)}"
        )
    total = sum(ratios.values())
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"ratios must sum to 1.0, got {total}")
    if any(v <= 0 for v in ratios.values()):
        raise ValueError(f"All ratios must be positive, got {ratios}")
