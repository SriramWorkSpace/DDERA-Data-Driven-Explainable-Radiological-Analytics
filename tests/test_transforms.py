"""Preprocessing / augmentation pipelines (ADR-006).

The ADR-006 tests are the mechanical enforcement of "no horizontal flip": both a structural
check (no reflection transform in the pipeline) and an empirical one (a left-right
asymmetric image is never mirrored by the training augmentation).
"""

from __future__ import annotations

import albumentations as A
import numpy as np
import pytest
import torch

from ddera.config import EncoderConfig, TransformConfig
from ddera.data.transforms import (
    FORBIDDEN_TRANSFORMS,
    assert_no_reflection,
    build_eval_transform,
    build_train_transform,
)


def _uint8_img(h: int = 300, w: int = 320) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


class TestEvalTransform:
    def test_output_shape_and_dtype(self):
        out = build_eval_transform(EncoderConfig())(image=_uint8_img())["image"]
        assert out.shape == (3, 224, 224)
        assert out.dtype == torch.float32

    def test_is_deterministic(self):
        transform = build_eval_transform(EncoderConfig())
        img = _uint8_img()
        assert torch.equal(transform(image=img)["image"], transform(image=img)["image"])

    def test_resolution_comes_from_config(self):
        out = build_eval_transform(EncoderConfig(resolution=320))(image=_uint8_img())["image"]
        assert out.shape == (3, 320, 320)

    def test_imagenet_normalisation_is_applied(self):
        enc = EncoderConfig()
        img = np.full((64, 64, 3), 127, np.uint8)
        out = build_eval_transform(enc)(image=img)["image"]
        expected = [
            (127 / 255 - m) / s
            for m, s in zip(enc.normalization_mean, enc.normalization_std, strict=True)
        ]
        for c in range(3):
            assert out[c].mean().item() == pytest.approx(expected[c], abs=1e-3)


class TestTrainTransform:
    def test_output_shape(self):
        out = build_train_transform(TransformConfig(), EncoderConfig())(image=_uint8_img())["image"]
        assert out.shape == (3, 224, 224)

    def test_is_stochastic(self):
        transform = build_train_transform(TransformConfig(augment_prob=1.0), EncoderConfig())
        img = _uint8_img()
        outs = [transform(image=img)["image"] for _ in range(5)]
        assert not all(torch.equal(outs[0], o) for o in outs[1:])


class TestADR006NoReflection:
    def test_forbidden_set_covers_the_flips(self):
        assert {
            "HorizontalFlip",
            "VerticalFlip",
            "Transpose",
            "RandomRotate90",
        } <= FORBIDDEN_TRANSFORMS

    def test_built_pipelines_pass_the_guard(self):
        assert_no_reflection(build_eval_transform(EncoderConfig()))
        assert_no_reflection(build_train_transform(TransformConfig(), EncoderConfig()))

    def test_guard_rejects_a_pipeline_with_a_flip(self):
        bad = A.Compose([A.HorizontalFlip(p=1.0), A.Resize(224, 224)])
        with pytest.raises(AssertionError, match="ADR-006"):
            assert_no_reflection(bad)

    def test_training_augmentation_never_mirrors_left_right(self):
        # Left half bright, right half dark. Small affine jitter must preserve that ordering
        # for every draw -- a horizontal flip would invert it.
        img = np.zeros((224, 224, 3), np.uint8)
        img[:, :112] = 240
        transform = build_train_transform(TransformConfig(augment_prob=1.0), EncoderConfig())
        for k in range(30):
            # albumentations draws from the legacy global RNG; seeding it is how we vary the
            # augmentation deterministically across iterations (same rationale as ddera.seed).
            np.random.seed(k)  # noqa: NPY002
            out = transform(image=img)["image"]
            left = out[:, :, :112].mean().item()
            right = out[:, :, 112:].mean().item()
            assert left > right, f"iteration {k}: left {left:.3f} !> right {right:.3f}"
