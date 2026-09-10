"""Feature cache + fingerprint / stale-cache protection (ADR-008).

Uses a random-init encoder (``weights=None``) over the synthetic image dataset. The
stale-cache rejection test is the project-plan Phase-2 GATE item.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ddera.config import EncoderConfig
from ddera.data.dataset import CachedFeatureDataset, CheXpertImageDataset, EncodedSplit
from ddera.data.transforms import build_eval_transform
from ddera.features.cache import FeatureCache, Fingerprint, StaleCacheError
from ddera.models.encoder import DenseNet121Encoder
from ddera.seed import set_seed


@pytest.fixture
def image_ds(synthetic_processed):
    out, root, spec = synthetic_processed
    ds = CheXpertImageDataset(
        out / "splits.parquet", "train", spec, data_root=root, transform=build_eval_transform()
    )
    return ds, out / "splits.parquet", spec


@pytest.fixture
def encoder():
    set_seed(0)
    return DenseNet121Encoder(EncoderConfig(weights=None), frozen=True)


@pytest.fixture
def built_cache(tmp_path, image_ds, encoder):
    ds, _, _ = image_ds
    cache = FeatureCache(tmp_path / "features")
    fp = cache.extract(encoder, ds, "train", batch_size=8)
    return cache, fp, ds


class TestExtractAndLoad:
    def test_writes_the_three_artifacts(self, built_cache, tmp_path):
        root = tmp_path / "features"
        assert (root / "train.npy").exists()
        assert (root / "train_index.parquet").exists()
        assert (root / "fingerprint.json").exists()

    def test_load_returns_float16_memmap_of_the_right_shape(self, built_cache, encoder):
        cache, _, ds = built_cache
        features, index_df, _ = cache.load("train", expected=Fingerprint.from_encoder(encoder))
        assert isinstance(features, np.memmap)
        assert features.dtype == np.float16
        assert features.shape == (len(ds), 1024)
        assert index_df["path"].astype(str).tolist() == ds.paths

    def test_cached_values_match_a_fresh_forward_pass(self, built_cache, encoder):
        cache, _, ds = built_cache
        features, _, _ = cache.load("train")
        with torch.no_grad():
            fresh = encoder(torch.stack([ds[k]["image"] for k in range(len(ds))])).numpy()
        assert np.allclose(features.astype(np.float32), fresh, atol=3e-2, rtol=3e-2)

    def test_missing_split_raises_file_not_found(self, built_cache):
        cache, _, _ = built_cache
        with pytest.raises(FileNotFoundError):
            cache.load("test")


class TestStaleCacheRejection:
    def test_different_encoder_weights_are_rejected(self, built_cache):
        cache, _, _ = built_cache
        set_seed(999)
        other = DenseNet121Encoder(EncoderConfig(weights=None))
        with pytest.raises(StaleCacheError, match="do not match"):
            cache.load("train", expected=Fingerprint.from_encoder(other))

    def test_different_resolution_is_rejected(self, built_cache, encoder):
        cache, fp, _ = built_cache
        bumped = Fingerprint(
            arch=fp.arch,
            weights=fp.weights,
            weights_sha256=fp.weights_sha256,
            resolution=320,
            channels=fp.channels,
            feature_dim=fp.feature_dim,
            normalization_mean=fp.normalization_mean,
            normalization_std=fp.normalization_std,
        )
        with pytest.raises(StaleCacheError):
            cache.load("train", expected=bumped)

    def test_missing_fingerprint_file_is_stale(self, built_cache, tmp_path):
        cache, _, _ = built_cache
        (tmp_path / "features" / "fingerprint.json").unlink()
        with pytest.raises(StaleCacheError, match="missing"):
            cache.load("train")

    def test_is_fresh_true_then_false_after_tampering(self, built_cache, encoder, tmp_path):
        cache, _, _ = built_cache
        assert cache.is_fresh(encoder, "train") is True
        fpath = tmp_path / "features" / "fingerprint.json"
        data = json.loads(fpath.read_text())
        data["resolution"] = 999
        fpath.write_text(json.dumps(data))
        assert cache.is_fresh(encoder, "train") is False


class TestCachedFeatureDataset:
    def test_returns_features_with_the_same_label_contract(self, built_cache, tmp_path, image_ds):
        _, splits_path, spec = image_ds
        ds = CachedFeatureDataset(tmp_path / "features", "train", splits_path, spec)
        item = ds[0]
        assert set(item) == {"features", "concepts", "concept_mask", "target", "row"}
        assert item["features"].shape == (1024,)
        assert item["features"].dtype == torch.float32

        ref = EncodedSplit(splits_path, "train", spec)
        assert np.array_equal(ds[3]["concepts"].numpy(), ref.concept_labels[3])
        assert np.array_equal(ds[3]["concept_mask"].numpy(), ref.concept_mask[3])

    def test_rejects_a_split_without_a_cache(self, built_cache, tmp_path, image_ds):
        _, splits_path, spec = image_ds
        with pytest.raises(FileNotFoundError):
            CachedFeatureDataset(tmp_path / "features", "val", splits_path, spec)
