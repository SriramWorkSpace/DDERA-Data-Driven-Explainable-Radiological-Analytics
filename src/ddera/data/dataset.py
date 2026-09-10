"""Torch datasets over the Phase-1 artifacts (``splits.parquet``).

:class:`CheXpertImageDataset` returns ``(image, concepts, concept_mask, target)`` for one
split. :class:`CachedFeatureDataset` (Phase 2b) returns a cached encoder feature vector in
place of the image, on the identical label contract.

The concept uncertainty policy (ADR-004) is applied **here**, from the ``ConceptSpec``, once
per split. Phase 1 persisted concept labels **raw** (1 / 0 / -1 / NaN) precisely so this
stays configurable: ``u_mask`` is the default, ``u_zeros`` / ``u_ones`` are the sensitivity
analysis. The ``target`` column is already binary -- Phase 1 applied ``u_ignore`` to it,
which is a row drop and therefore a cohort decision made upstream.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ddera.config import ConceptSpec
from ddera.data.chexpert import concept_matrix
from ddera.data.labels import encode_concept_matrix
from ddera.data.splits import load_splits

VALID_SPLITS = ("train", "val", "test")


class EncodedSplit:
    """Policy-encoded concept labels + target + provenance for one split.

    Shared by both datasets so the label handling is defined exactly once. The heavy work
    (applying the uncertainty policy to the whole ``(N, K)`` matrix) happens here at
    construction, not per ``__getitem__``.
    """

    def __init__(
        self,
        splits_path: str | Path,
        split: str,
        concept_spec: ConceptSpec,
        *,
        concept_policy: str | None = None,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
        frame, _ = load_splits(splits_path)
        if "split" not in frame.columns:
            raise ValueError(f"{splits_path} has no 'split' column; is it a splits.parquet?")
        sub = frame.loc[frame["split"] == split].reset_index(drop=True)
        if len(sub) == 0:
            raise ValueError(f"No rows for split {split!r} in {splits_path}")

        policy = concept_policy or concept_spec.uncertainty.concept_policy
        raw = concept_matrix(sub, concept_spec.concepts)
        labels, mask = encode_concept_matrix(
            raw, policy=policy, blank_policy=concept_spec.uncertainty.blank_policy
        )

        self.split = split
        self.concept_names: list[str] = list(concept_spec.concepts)
        self.concept_policy = policy
        self.concept_labels = labels.astype(np.float32)
        self.concept_mask = mask.astype(np.float32)
        self.targets = sub["target"].to_numpy(dtype=np.float32)
        self.paths: list[str] = sub["path"].astype(str).tolist()
        self.patient_ids: list[str] = sub["patient_id"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.paths)

    def label_bundle(self, i: int) -> dict[str, torch.Tensor | int]:
        return {
            "concepts": torch.from_numpy(self.concept_labels[i]),
            "concept_mask": torch.from_numpy(self.concept_mask[i]),
            "target": torch.tensor(self.targets[i]),
            "row": i,
        }


class _LabelledDataset(Dataset):
    """Common plumbing: an :class:`EncodedSplit` plus the passthrough accessors."""

    def __init__(self, encoded: EncodedSplit) -> None:
        self._enc = encoded

    def __len__(self) -> int:
        return len(self._enc)

    @property
    def paths(self) -> list[str]:
        return self._enc.paths

    @property
    def patient_ids(self) -> list[str]:
        return self._enc.patient_ids

    @property
    def targets(self) -> np.ndarray:
        return self._enc.targets

    @property
    def concept_names(self) -> list[str]:
        return self._enc.concept_names

    @property
    def concept_labels(self) -> np.ndarray:
        return self._enc.concept_labels

    @property
    def concept_mask(self) -> np.ndarray:
        return self._enc.concept_mask


class CheXpertImageDataset(_LabelledDataset):
    """One split of ``splits.parquet`` as ``(image, concepts, concept_mask, target)``.

    Args:
        splits_path: the Phase-1 ``splits.parquet``.
        split: ``"train"`` / ``"val"`` / ``"test"``.
        concept_spec: the concept/target contract (``configs/concepts/chexpert_v1.yaml``).
        data_root: directory the manifest ``path`` values resolve against -- i.e. the
            ``--dest`` passed to ``scripts/get_data.py`` (contains ``CheXpert-v1.0-small/``).
        transform: an albumentations pipeline from :mod:`ddera.data.transforms`.
        concept_policy: override the spec's concept policy (for the u_zeros / u_ones
            sensitivity analysis). ``None`` uses the spec.
    """

    def __init__(
        self,
        splits_path: str | Path,
        split: str,
        concept_spec: ConceptSpec,
        *,
        data_root: str | Path,
        transform,
        concept_policy: str | None = None,
    ) -> None:
        super().__init__(
            EncodedSplit(splits_path, split, concept_spec, concept_policy=concept_policy)
        )
        self.data_root = Path(data_root)
        self.transform = transform

    def __getitem__(self, i: int) -> dict:
        import cv2  # local import: keeps `import ddera.data.dataset` cheap for label-only use

        fpath = self.data_root / self._enc.paths[i]
        image = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(
                f"Row {i}: image not readable at {fpath}. Run `scripts/get_data.py "
                "--check-images --drop-unreadable`, or fix the download."
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image=image)["image"]
        return {"image": tensor, **self._enc.label_bundle(i)}
