"""DenseNet-121 feature encoder (ADR-001).

Constructed with ``weights=None`` throughout so the tests never touch the network. The
fingerprint hashes the actual parameters, so a seeded build is reproducible and a
different seed is a different encoder.
"""

from __future__ import annotations

import pytest
import torch

from ddera.config import EncoderConfig
from ddera.models.encoder import DenseNet121Encoder
from ddera.seed import set_seed


@pytest.fixture(scope="module")
def frozen_encoder() -> DenseNet121Encoder:
    set_seed(0)
    return DenseNet121Encoder(EncoderConfig(weights=None), frozen=True)


class TestForward:
    def test_maps_image_batch_to_1024d(self, frozen_encoder):
        out = frozen_encoder(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 1024)
        assert torch.isfinite(out).all()

    def test_eval_forward_is_deterministic(self, frozen_encoder):
        x = torch.randn(2, 3, 224, 224)
        assert torch.equal(frozen_encoder(x), frozen_encoder(x))

    def test_accepts_the_configured_resolution(self):
        set_seed(0)
        enc = DenseNet121Encoder(EncoderConfig(weights=None, resolution=256))
        assert enc(torch.randn(1, 3, 256, 256)).shape == (1, 1024)


class TestFrozenPolicy:
    def test_frozen_encoder_has_no_trainable_params(self, frozen_encoder):
        assert all(not p.requires_grad for p in frozen_encoder.parameters())
        assert frozen_encoder.training is False

    def test_unfrozen_encoder_is_trainable(self):
        set_seed(0)
        enc = DenseNet121Encoder(EncoderConfig(weights=None), frozen=False)
        assert any(p.requires_grad for p in enc.parameters())


class TestFingerprint:
    def test_has_the_adr008_fields(self, frozen_encoder):
        fp = frozen_encoder.fingerprint()
        for key in ("weights_sha256", "resolution", "normalization_mean", "feature_dim"):
            assert key in fp

    def test_same_seed_same_hash_different_seed_different_hash(self):
        set_seed(0)
        a = DenseNet121Encoder(EncoderConfig(weights=None)).weights_sha256()
        set_seed(0)
        b = DenseNet121Encoder(EncoderConfig(weights=None)).weights_sha256()
        set_seed(1)
        c = DenseNet121Encoder(EncoderConfig(weights=None)).weights_sha256()
        assert a == b
        assert a != c


class TestConfigGuards:
    def test_non_densenet_arch_rejected(self):
        with pytest.raises(ValueError, match="densenet121"):
            EncoderConfig(arch="resnet50")
