"""``python -m ddera.train`` -- run one experiment end to end.

    python -m ddera.train --config configs/experiment/m2_sequential.yaml
    python -m ddera.train --config configs/experiment/m2_sequential.yaml --fast-dev-run
    python -m ddera.train --config configs/experiment/m2_sequential.yaml --synthetic

``--synthetic`` builds a small synthetic CheXpert tree, runs the Phase-1 pipeline and (for
frozen-encoder variants) extracts a feature cache, then trains and evaluates on that -- and
also fits a quick B0 baseline and re-infers concepts under three input perturbations, so
the run reaches all eight metric families. Every number it prints and writes is labelled
SYNTHETIC: it is a mechanism demonstration, not a result (CLAUDE.md section 8).

Real runs need ``splits_parquet`` / ``feature_cache`` in the config (``scripts/get_data.py``
plus feature extraction). ``--baseline-run <dir>`` supplies a B0 run whose predictions give
the completeness interpretability-cost.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import albumentations as A  # noqa: E402
import cv2  # noqa: E402
import torch  # noqa: E402
from albumentations.pytorch import ToTensorV2  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ddera.config import (  # noqa: E402
    RUNS_ROOT,
    ConceptSpec,
    EncoderConfig,
    ExperimentConfig,
    ModelConfig,
)
from ddera.data.dataset import CachedFeatureDataset, CheXpertImageDataset  # noqa: E402
from ddera.data.transforms import build_eval_transform  # noqa: E402
from ddera.models.cbm import EncoderWrapped, build_model  # noqa: E402
from ddera.models.encoder import DenseNet121Encoder  # noqa: E402
from ddera.reporting.runs import load_run, log_run, new_run_id  # noqa: E402
from ddera.seed import make_generator, seed_worker, set_seed  # noqa: E402
from ddera.train.evaluate import evaluate_model  # noqa: E402
from ddera.train.loop import train_model  # noqa: E402

_REGIME = {"m1_independent": "independent", "m2_sequential": "sequential", "m3_joint": "joint"}
_FROZEN_VARIANTS = {"m1_independent", "m2_sequential", "m4_hybrid"}


def _loader(dataset, cfg: ExperimentConfig, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator(cfg.seed),
    )


def _perturbing_transform(kind: str, enc: EncoderConfig) -> A.Compose:
    """An eval transform preceded by one fixed, deterministic perturbation (for stability)."""
    pre = {
        "rotate": A.Affine(rotate=(6, 6), border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),
        "brightness": A.RandomBrightnessContrast(
            brightness_limit=(0.25, 0.25), contrast_limit=0.0, p=1.0
        ),
        "noise": A.GaussNoise(std_range=(0.1, 0.15), p=1.0),
    }[kind]
    return A.Compose(
        [
            pre,
            A.Resize(enc.resolution, enc.resolution),
            A.Normalize(mean=enc.normalization_mean, std=enc.normalization_std),
            ToTensorV2(),
        ]
    )


def _synthetic(cfg: ExperimentConfig, workdir: Path, spec: ConceptSpec):
    from ddera.data.acquire import build_processed_dataset
    from ddera.data.synthetic import write_synthetic_chexpert_tree
    from ddera.features.cache import FeatureCache

    enc = EncoderConfig(weights=None, resolution=96)
    raw = write_synthetic_chexpert_tree(
        workdir / "chexpert",
        n_patients=200,
        with_images=True,
        image_size=96,
        signal_strength=1.5,
        seed=cfg.seed,
    )
    processed = workdir / "processed"
    build_processed_dataset(raw, spec, processed, seed=cfg.seed, check_images=True)
    splits = processed / "splits.parquet"
    tfm = build_eval_transform(enc)
    encoder = DenseNet121Encoder(enc, frozen=True)

    frozen = cfg.model.variant in _FROZEN_VARIANTS
    if frozen:
        cache = FeatureCache(workdir / "features")
        for split in ("train", "val"):
            cache.extract(
                encoder,
                CheXpertImageDataset(splits, split, spec, data_root=raw, transform=tfm),
                split,
                batch_size=16,
            )
        train_ds = CachedFeatureDataset(cache.root, "train", splits, spec)
        val_ds = CachedFeatureDataset(cache.root, "val", splits, spec)
        features = cache.load("val")[0][:].astype("float32")
    else:
        train_ds = CheXpertImageDataset(splits, "train", spec, data_root=raw, transform=tfm)
        val_ds = CheXpertImageDataset(splits, "val", spec, data_root=raw, transform=tfm)
        features = None

    val_images = CheXpertImageDataset(splits, "val", spec, data_root=raw, transform=tfm)
    pert = {
        k: CheXpertImageDataset(
            splits, "val", spec, data_root=raw, transform=_perturbing_transform(k, enc)
        )
        for k in ("rotate", "brightness", "noise")
    }
    return {
        "train": train_ds,
        "val": val_ds,
        "val_images": val_images,
        "perturbations": pert,
        "features": features,
        "encoder": encoder if frozen else None,
        "train_encoder": None if frozen else DenseNet121Encoder(enc, frozen=False),
        "enc_cfg": enc,
        "splits": splits,
        "raw": raw,
        "spec": spec,
    }


def _real(cfg: ExperimentConfig, spec: ConceptSpec):
    if not cfg.splits_parquet:
        raise SystemExit(
            "This config has no data locations. Set splits_parquet / feature_cache "
            "(scripts/get_data.py + feature extraction), or run with --synthetic."
        )
    splits = Path(cfg.splits_parquet)
    frozen = cfg.model.variant in _FROZEN_VARIANTS
    if frozen:
        if not cfg.feature_cache:
            raise SystemExit("Frozen-encoder variant needs feature_cache set in the config.")
        train_ds = CachedFeatureDataset(cfg.feature_cache, "train", splits, spec)
        val_ds = CachedFeatureDataset(cfg.feature_cache, "val", splits, spec)
        return {
            "train": train_ds,
            "val": val_ds,
            "val_images": None,
            "perturbations": {},
            "features": None,
            "encoder": None,
            "train_encoder": None,
            "enc_cfg": cfg.model.encoder,
            "splits": splits,
            "raw": None,
            "spec": spec,
        }
    tfm = build_eval_transform(cfg.model.encoder)
    train_ds = CheXpertImageDataset(splits, "train", spec, data_root=cfg.data_root, transform=tfm)
    val_ds = CheXpertImageDataset(splits, "val", spec, data_root=cfg.data_root, transform=tfm)
    return {
        "train": train_ds,
        "val": val_ds,
        "val_images": val_ds,
        "perturbations": {},
        "features": None,
        "encoder": None,
        "train_encoder": DenseNet121Encoder(cfg.model.encoder, frozen=cfg.model.freeze_encoder),
        "enc_cfg": cfg.model.encoder,
        "splits": splits,
        "raw": None,
        "spec": spec,
    }


def _fit_baseline(bundle: dict, cfg: ExperimentConfig, device: torch.device) -> object | None:
    """A quick B0 fit for the completeness reference (synthetic path only)."""
    b0_cfg = dataclasses.replace(
        cfg,
        model=ModelConfig(variant="b0", encoder=bundle["enc_cfg"], freeze_encoder=True),
        epochs=min(cfg.epochs, 15),
    )
    b0 = build_model(b0_cfg.model, encoder=nn.Identity())
    if bundle["encoder"] is not None:
        b0 = EncoderWrapped(bundle["encoder"], b0)
    train_model(
        b0,
        _loader(bundle["train"], cfg, shuffle=True),
        _loader(bundle["val"], cfg, shuffle=False),
        b0_cfg,
        device=device,
        regime="sequential",
        fast_dev_run=False,
    )
    preds, _, _ = evaluate_model(
        b0,
        _loader(bundle["val"], cfg, shuffle=False),
        spec=bundle["spec"],
        device=device,
        split="val",
    )
    return preds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--fast-dev-run", action="store_true", help="1 epoch, 2 batches")
    parser.add_argument("--synthetic", action="store_true", help="train on a synthetic tree")
    parser.add_argument(
        "--out", type=Path, default=RUNS_ROOT, help="runs root (default: %(default)s)"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--variant", default=None, help="override cfg.model.variant")
    parser.add_argument(
        "--baseline-run", type=Path, default=None, help="a B0 run dir for completeness"
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args(argv)

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.epochs is not None:
        cfg = dataclasses.replace(cfg, epochs=args.epochs)
    if args.variant is not None:
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, variant=args.variant))
    set_seed(cfg.seed)
    spec = ConceptSpec.from_yaml(
        cfg.concepts if "/" in cfg.concepts else f"concepts/{cfg.concepts}.yaml"
    )

    if args.device == "cpu" or (args.synthetic and args.device == "auto"):
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda:0")
    else:
        from ddera.device import get_device

        device = get_device()

    tmp: tempfile.TemporaryDirectory | None = None
    if args.synthetic:
        tmp = tempfile.TemporaryDirectory()
        bundle = _synthetic(cfg, Path(tmp.name), spec)
    else:
        bundle = _real(cfg, spec)

    train_encoder = bundle["train_encoder"]
    model = build_model(
        cfg.model, encoder=train_encoder if train_encoder is not None else nn.Identity()
    )
    regime = _REGIME.get(cfg.model.variant, "sequential")

    print(
        f"{'[SYNTHETIC] ' if args.synthetic else ''}training {cfg.model.variant} ({regime}) "
        f"on {len(bundle['train'])} train / {len(bundle['val'])} val  [device={device}]"
    )
    history = train_model(
        model,
        _loader(bundle["train"], cfg, shuffle=True),
        _loader(bundle["val"], cfg, shuffle=False),
        cfg,
        device=device,
        regime=regime,
        fast_dev_run=args.fast_dev_run,
    )

    # Evaluation: image path when we have images (enables the stability family).
    eval_model = model
    if bundle["val_images"] is not None and bundle["encoder"] is not None:
        eval_model = EncoderWrapped(bundle["encoder"], model)
    eval_ds = bundle["val_images"] if bundle["val_images"] is not None else bundle["val"]
    pert_loaders = {
        k: _loader(ds, cfg, shuffle=False) for k, ds in bundle["perturbations"].items()
    } or None

    baseline_predictions = None
    if args.baseline_run is not None:
        baseline_predictions = load_run(args.baseline_run).get("predictions")
    elif args.synthetic and not args.fast_dev_run and cfg.model.variant != "b0":
        baseline_predictions = _fit_baseline(bundle, cfg, device)

    predictions, metrics, concept_weights = evaluate_model(
        eval_model,
        _loader(eval_ds, cfg, shuffle=False),
        spec=spec,
        device=device,
        split="val",
        perturbation_loaders=pert_loaders,
        features=bundle["features"],
        baseline_predictions=baseline_predictions,
        seed=cfg.seed,
    )
    metrics["training"] = history.to_dict()
    metrics["synthetic"] = bool(args.synthetic)
    config_dict = cfg.to_dict()
    config_dict["_synthetic"] = bool(args.synthetic)
    config_dict["_resolved_encoder"] = bundle["enc_cfg"].to_dict()

    run_dir = Path(args.out) / new_run_id(cfg.model.variant)
    paths = log_run(
        run_dir,
        config=config_dict,
        metrics=metrics,
        predictions=predictions,
        concept_weights=concept_weights,
        checkpoint=model.state_dict(),
        partial=(metrics["status"] != "complete"),
        index_csv=Path(args.out) / "runs_index.csv",
    )
    if tmp is not None:
        tmp.cleanup()

    auroc = metrics["predictive"].get("auroc")
    banner = "  ** SYNTHETIC -- mechanism demo, NOT a result **" if args.synthetic else ""
    print(
        f"run -> {paths.root}\n  status={metrics['status']}  val AUROC={auroc:.3f}"
        f"  families={metrics['families_present']}{banner}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
