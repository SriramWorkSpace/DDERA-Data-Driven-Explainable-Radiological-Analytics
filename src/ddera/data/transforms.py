"""Image preprocessing and augmentation (ADR-006).

Training pipeline: small affine jitter (``Affine``: +/-7 deg rotation, +/-5 % scale, small
translation) plus brightness/contrast jitter, then resize to the encoder resolution and
ImageNet normalisation. Evaluation pipeline: resize + normalise only.

**No horizontal flip, no vertical flip, no transpose, no 90-degree rotation.** Chest anatomy
is not left-right symmetric: the cardiac silhouette sits left of midline and its apparent
size relative to the thorax is exactly what the Cardiomegaly and Enlarged Cardiomediastinum
concepts measure. Reflecting an X-ray produces an anatomically impossible image (effectively
situs inversus) and teaches the encoder that laterality is irrelevant, corrupting two of the
twelve concepts. :func:`assert_no_reflection` enforces this on every pipeline built here.
"""

from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from ddera.config import EncoderConfig, TransformConfig

#: Albumentations transforms that reflect or re-orient the image. ADR-006 forbids all of them.
FORBIDDEN_TRANSFORMS: frozenset[str] = frozenset(
    {"HorizontalFlip", "VerticalFlip", "Flip", "Transpose", "RandomRotate90", "D4"}
)


def assert_no_reflection(pipeline: A.Compose) -> None:
    """Raise if an ADR-006-forbidden transform is present in ``pipeline``.

    Called on every pipeline this module builds, so a reflection augmentation cannot be
    introduced without a test failure.
    """
    present = {type(t).__name__ for t in pipeline.transforms}
    bad = present & FORBIDDEN_TRANSFORMS
    if bad:
        raise AssertionError(
            f"ADR-006 violation: reflection / re-orientation transform(s) {sorted(bad)} in "
            "the augmentation pipeline. Chest anatomy is not left-right symmetric."
        )


def build_eval_transform(encoder: EncoderConfig | None = None) -> A.Compose:
    """Deterministic: resize to the encoder resolution, ImageNet-normalise, to CHW tensor.

    Input is an HxWx3 ``uint8`` RGB array; output is a ``float32`` ``(3, R, R)`` tensor.
    """
    enc = encoder or EncoderConfig()
    pipeline = A.Compose(
        [
            A.Resize(enc.resolution, enc.resolution),
            A.Normalize(mean=enc.normalization_mean, std=enc.normalization_std),
            ToTensorV2(),
        ]
    )
    assert_no_reflection(pipeline)
    return pipeline


def build_train_transform(
    transform: TransformConfig | None = None,
    encoder: EncoderConfig | None = None,
) -> A.Compose:
    """Affine jitter + brightness/contrast, then the eval steps. ADR-006 compliant.

    Geometry and photometry come from ``transform``; resolution and normalisation from
    ``encoder`` (the same constants the feature-cache fingerprint records).
    """
    tcfg = transform or TransformConfig()
    enc = encoder or EncoderConfig()
    pipeline = A.Compose(
        [
            A.Affine(
                scale=(1.0 - tcfg.scale_limit, 1.0 + tcfg.scale_limit),
                translate_percent=(-tcfg.shift_limit, tcfg.shift_limit),
                rotate=(-tcfg.rotate_limit_deg, tcfg.rotate_limit_deg),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=tcfg.augment_prob,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=tcfg.brightness_limit,
                contrast_limit=tcfg.contrast_limit,
                p=tcfg.augment_prob,
            ),
            A.Resize(enc.resolution, enc.resolution),
            A.Normalize(mean=enc.normalization_mean, std=enc.normalization_std),
            ToTensorV2(),
        ]
    )
    assert_no_reflection(pipeline)
    return pipeline
