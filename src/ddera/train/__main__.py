"""``python -m ddera.train`` -- run one experiment end to end.

    python -m ddera.train --config configs/experiment/m2_sequential.yaml
    python -m ddera.train --config configs/experiment/m2_sequential.yaml --fast-dev-run
    python -m ddera.train --config configs/experiment/m2_sequential.yaml --synthetic

``--synthetic`` builds a small synthetic CheXpert tree, runs the Phase-1 pipeline and (for
frozen-encoder variants) extracts a feature cache, then trains and evaluates on that. Every
number it prints and writes is labelled SYNTHETIC -- it is a mechanism demonstration, not a
result (CLAUDE.md section 8). Real runs need ``splits_parquet`` / ``feature_cache`` set in
the config (produced by ``scripts/get_data.py`` + feature extraction).

Phase 3 writes a **partial** run (predictive, calibration, concept quality, a first
intervention curve, faithfulness). Leakage / completeness / stability arrive with the
Phase 5 protocol.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ddera.config import RUNS_ROOT, ConceptSpec, EncoderConfig, ExperimentConfig  # noqa: E402
from ddera.data.dataset import CachedFeatureDataset, CheXpertImageDataset  # noqa: E402
from ddera.data.transforms import build_eval_transform  # noqa: E402
from ddera.models.cbm import build_model  # noqa: E402
from ddera.models.encoder import DenseNet121Encoder  # noqa: E402
from ddera.reporting.runs import log_run, new_run_id  # noqa: E402
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


def _synthetic_datasets(cfg: ExperimentConfig, workdir: Path, spec: ConceptSpec):
    """Synthetic tree -> Phase-1 pipeline -> (feature cache) -> train/val datasets."""
    from ddera.data.acquire import build_processed_dataset
    from ddera.data.synthetic import write_synthetic_chexpert_tree
    from ddera.features.cache import FeatureCache

    enc_cfg = EncoderConfig(weights=None, resolution=96)  # small + offline for the demo path
    raw = write_synthetic_chexpert_tree(
        workdir / "chexpert", n_patients=64, with_images=True, image_size=96, seed=cfg.seed
    )
    processed = workdir / "processed"
    build_processed_dataset(raw, spec, processed, seed=cfg.seed, check_images=True)
    splits = processed / "splits.parquet"
    tfm = build_eval_transform(enc_cfg)

    if cfg.model.variant in _FROZEN_VARIANTS:
        cache = FeatureCache(workdir / "features")
        encoder = DenseNet121Encoder(enc_cfg, frozen=True)
        for split in ("train", "val"):
            ds = CheXpertImageDataset(splits, split, spec, data_root=raw, transform=tfm)
            cache.extract(encoder, ds, split, batch_size=16)
        train = CachedFeatureDataset(cache.root, "train", splits, spec)
        val = CachedFeatureDataset(cache.root, "val", splits, spec)
        return train, val, None, enc_cfg

    encoder = DenseNet121Encoder(enc_cfg, frozen=False)
    train = CheXpertImageDataset(splits, "train", spec, data_root=raw, transform=tfm)
    val = CheXpertImageDataset(splits, "val", spec, data_root=raw, transform=tfm)
    return train, val, encoder, enc_cfg


def _real_datasets(cfg: ExperimentConfig, spec: ConceptSpec):
    if not cfg.splits_parquet:
        raise SystemExit(
            "This config has no data locations. Set splits_parquet / feature_cache (from "
            "scripts/get_data.py + feature extraction), or run with --synthetic."
        )
    splits = Path(cfg.splits_parquet)
    if cfg.model.variant in _FROZEN_VARIANTS:
        if not cfg.feature_cache:
            raise SystemExit("Frozen-encoder variant needs feature_cache set in the config.")
        train = CachedFeatureDataset(cfg.feature_cache, "train", splits, spec)
        val = CachedFeatureDataset(cfg.feature_cache, "val", splits, spec)
        return train, val, None, cfg.model.encoder
    tfm = build_eval_transform(cfg.model.encoder)
    root = cfg.data_root
    train = CheXpertImageDataset(splits, "train", spec, data_root=root, transform=tfm)
    val = CheXpertImageDataset(splits, "val", spec, data_root=root, transform=tfm)
    encoder = DenseNet121Encoder(cfg.model.encoder, frozen=cfg.model.freeze_encoder)
    return train, val, encoder, cfg.model.encoder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--fast-dev-run", action="store_true", help="1 epoch, 2 batches -- a smoke test"
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="train on a synthetic tree (SYNTHETIC)"
    )
    parser.add_argument(
        "--out", type=Path, default=RUNS_ROOT, help="runs root (default: %(default)s)"
    )
    parser.add_argument("--epochs", type=int, default=None, help="override cfg.epochs")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args(argv)

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.epochs is not None:
        cfg = dataclasses.replace(cfg, epochs=args.epochs)
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
        train_ds, val_ds, encoder, enc_cfg = _synthetic_datasets(cfg, Path(tmp.name), spec)
    else:
        train_ds, val_ds, encoder, enc_cfg = _real_datasets(cfg, spec)

    model = build_model(cfg.model, encoder=encoder if encoder is not None else nn.Identity())
    regime = _REGIME.get(cfg.model.variant, "sequential")

    print(
        f"{'[SYNTHETIC] ' if args.synthetic else ''}training {cfg.model.variant} "
        f"({regime}) on {len(train_ds)} train / {len(val_ds)} val  [device={device}]"
    )
    history = train_model(
        model,
        _loader(train_ds, cfg, shuffle=True),
        _loader(val_ds, cfg, shuffle=False),
        cfg,
        device=device,
        regime=regime,
        fast_dev_run=args.fast_dev_run,
    )

    predictions, metrics, concept_weights = evaluate_model(
        model, _loader(val_ds, cfg, shuffle=False), spec=spec, device=device, split="val"
    )
    metrics["training"] = history.to_dict()
    metrics["synthetic"] = bool(args.synthetic)
    config_dict = cfg.to_dict()
    config_dict["_synthetic"] = bool(args.synthetic)
    config_dict["_resolved_encoder"] = enc_cfg.to_dict()

    run_dir = Path(args.out) / new_run_id(cfg.model.variant)
    paths = log_run(
        run_dir,
        config=config_dict,
        metrics=metrics,
        predictions=predictions,
        concept_weights=concept_weights,
        checkpoint=model.state_dict(),
        partial=True,
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
