"""CheXpertImageDataset over Phase-1 splits.parquet.

Built on the ``synthetic_processed`` fixture (a synthetic CheXpert tree with real generated
JPEGs, run through the Phase-1 pipeline). Focus: the ADR-004 concept policy is applied here
from the spec, the target passes through unchanged, and splits stay patient-disjoint.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddera.data.chexpert import concept_matrix
from ddera.data.dataset import CheXpertImageDataset, EncodedSplit
from ddera.data.splits import load_splits
from ddera.data.transforms import build_eval_transform


def _train_ds(synthetic_processed, **kw):
    out, root, spec = synthetic_processed
    return CheXpertImageDataset(
        out / "splits.parquet",
        "train",
        spec,
        data_root=root,
        transform=build_eval_transform(),
        **kw,
    )


def _first_uncertain_cell(out, spec, split="train"):
    df, _ = load_splits(out / "splits.parquet")
    sub = df[df["split"] == split].reset_index(drop=True)
    raw = concept_matrix(sub, spec.concepts)
    loc = np.argwhere(raw == -1.0)
    assert len(loc) > 0, "synthetic split should contain uncertain concept labels"
    return int(loc[0][0]), int(loc[0][1])


class TestStructure:
    def test_len_matches_split_row_count(self, synthetic_processed):
        out, _, _ = synthetic_processed
        df, _ = load_splits(out / "splits.parquet")
        assert len(_train_ds(synthetic_processed)) == int((df["split"] == "train").sum())

    def test_item_keys_shapes_and_ranges(self, synthetic_processed):
        item = _train_ds(synthetic_processed)[0]
        assert set(item) == {"image", "concepts", "concept_mask", "target", "row"}
        assert item["image"].shape == (3, 224, 224)
        assert item["concepts"].shape == (12,)
        assert item["concept_mask"].shape == (12,)
        assert set(np.unique(item["concepts"].numpy())) <= {0.0, 1.0}
        assert set(np.unique(item["concept_mask"].numpy())) <= {0.0, 1.0}
        assert item["target"].item() in (0.0, 1.0)

    def test_bad_split_name_raises(self, synthetic_processed):
        out, _, spec = synthetic_processed
        with pytest.raises(ValueError, match="split must be one of"):
            EncodedSplit(out / "splits.parquet", "holdout", spec)


class TestADR004Policy:
    def test_u_mask_zeroes_label_and_mask_for_uncertain(self, synthetic_processed):
        out, _, spec = synthetic_processed
        i, j = _first_uncertain_cell(out, spec)
        enc = EncodedSplit(out / "splits.parquet", "train", spec)  # default: u_mask
        assert enc.concept_mask[i, j] == 0.0
        assert enc.concept_labels[i, j] == 0.0

    def test_u_ones_override_flips_uncertain_to_positive(self, synthetic_processed):
        out, _, spec = synthetic_processed
        i, j = _first_uncertain_cell(out, spec)
        enc = EncodedSplit(out / "splits.parquet", "train", spec, concept_policy="u_ones")
        assert enc.concept_mask[i, j] == 1.0
        assert enc.concept_labels[i, j] == 1.0

    def test_target_passes_through_from_parquet(self, synthetic_processed):
        out, root, spec = synthetic_processed
        df, _ = load_splits(out / "splits.parquet")
        sub = df[df["split"] == "val"].reset_index(drop=True)
        ds = CheXpertImageDataset(
            out / "splits.parquet", "val", spec, data_root=root, transform=build_eval_transform()
        )
        for k in range(min(5, len(ds))):
            assert ds[k]["target"].item() == pytest.approx(float(sub["target"].iloc[k]))


class TestProvenance:
    def test_train_and_test_are_patient_disjoint(self, synthetic_processed):
        out, root, spec = synthetic_processed
        tf = build_eval_transform()
        train = CheXpertImageDataset(
            out / "splits.parquet", "train", spec, data_root=root, transform=tf
        )
        test = CheXpertImageDataset(
            out / "splits.parquet", "test", spec, data_root=root, transform=tf
        )
        assert set(train.patient_ids).isdisjoint(test.patient_ids)

    def test_missing_image_raises_with_the_path(self, synthetic_processed):
        out, root, _ = synthetic_processed
        ds = _train_ds(synthetic_processed)
        (root / ds.paths[0]).unlink()
        with pytest.raises(FileNotFoundError, match="not readable"):
            _ = ds[0]
