"""Shared pytest fixtures.

All fixtures are built on :mod:`ddera.data.synthetic`, where the true reasoner weights and
true concepts are known. That is what lets the tests assert the harness **recovers** the
right answer rather than merely that it runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ddera.data.synthetic import make_synthetic_cbm, write_synthetic_chexpert_tree  # noqa: E402
from ddera.xai.intervention import LinearReasoner  # noqa: E402


@pytest.fixture(scope="session")
def synth():
    """Default dataset: noisy concept predictor, no leakage past the bottleneck."""
    return make_synthetic_cbm(
        n_patients=300,
        studies_per_patient=2,
        n_concepts=8,
        concept_noise=0.8,
        leak_strength=0.0,
        seed=0,
    )


@pytest.fixture(scope="session")
def synth_clean():
    """A near-perfect concept predictor: interventions should have almost nothing to fix."""
    return make_synthetic_cbm(
        n_patients=300,
        studies_per_patient=2,
        n_concepts=8,
        concept_noise=0.01,
        concept_separability=6.0,
        leak_strength=0.0,
        seed=1,
    )


@pytest.fixture(scope="session")
def synth_leaky():
    """Target information injected into features that bypasses the concepts entirely."""
    return make_synthetic_cbm(
        n_patients=300,
        studies_per_patient=2,
        n_concepts=8,
        concept_noise=0.8,
        leak_strength=4.0,
        seed=0,
    )


@pytest.fixture(scope="session")
def synth_uncertain():
    """Raw labels containing uncertain (-1) and blank (NaN) entries."""
    return make_synthetic_cbm(
        n_patients=120,
        studies_per_patient=2,
        n_concepts=6,
        uncertain_rate=0.15,
        blank_rate=0.10,
        seed=3,
    )


@pytest.fixture
def synthetic_processed(tmp_path):
    """Phase-1 artifacts from a synthetic CheXpert tree *with* generated JPEGs.

    Returns ``(processed_dir, data_root, ConceptSpec)`` where ``processed_dir`` holds
    ``manifest.parquet`` / ``splits.parquet`` and ``data_root`` is what the manifest paths
    resolve against. Used by the Phase-2 dataset / transforms / reporting tests.
    """
    from ddera.config import ConceptSpec
    from ddera.data.acquire import build_processed_dataset

    raw = write_synthetic_chexpert_tree(
        tmp_path / "chexpert", n_patients=64, with_images=True, seed=0
    )
    spec = ConceptSpec.from_yaml("configs/concepts/chexpert_v1.yaml")
    out = tmp_path / "processed"
    build_processed_dataset(raw, spec, out, seed=42, check_images=True)
    return out, raw, spec


@pytest.fixture(scope="session")
def synthetic_cached(tmp_path_factory):
    """A synthetic feature cache + splits.parquet for the Phase-3 training/eval tests.

    Returns ``(cache_dir, splits_parquet, ConceptSpec)``. Built once per session: synthetic
    tree -> Phase-1 pipeline -> DenseNet (random init) feature extraction for train + val,
    at 64px for speed.
    """
    from ddera.config import ConceptSpec, EncoderConfig
    from ddera.data.acquire import build_processed_dataset
    from ddera.data.dataset import CheXpertImageDataset
    from ddera.data.transforms import build_eval_transform
    from ddera.features.cache import FeatureCache
    from ddera.models.encoder import DenseNet121Encoder

    root = tmp_path_factory.mktemp("synthetic_cached")
    raw = write_synthetic_chexpert_tree(
        root / "chexpert", n_patients=60, with_images=True, image_size=64, seed=0
    )
    spec = ConceptSpec.from_yaml("configs/concepts/chexpert_v1.yaml")
    processed = root / "processed"
    build_processed_dataset(raw, spec, processed, seed=42, check_images=True)
    splits = processed / "splits.parquet"

    enc_cfg = EncoderConfig(weights=None, resolution=64)
    encoder = DenseNet121Encoder(enc_cfg, frozen=True)
    cache = FeatureCache(root / "features")
    tfm = build_eval_transform(enc_cfg)
    for split in ("train", "val"):
        ds = CheXpertImageDataset(splits, split, spec, data_root=raw, transform=tfm)
        cache.extract(encoder, ds, split, batch_size=16)
    return cache.root, splits, spec


@pytest.fixture
def reasoner(synth):
    """The TRUE linear reasoner that generated the synthetic targets."""
    return LinearReasoner(synth.weights, synth.bias, synth.concept_names)


@pytest.fixture
def predict_fn(reasoner):
    """The ``concepts -> probabilities`` callable the XAI harness consumes."""
    return reasoner.as_predict_fn()
