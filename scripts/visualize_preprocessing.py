"""Phase-2 progress figures: augmentation, preprocessed samples, encoder feature space,
feature-cache round-trip, and the linear concept probe (GATE 2 sanity check).

    # real data (after scripts/get_data.py):
    python scripts/visualize_preprocessing.py --processed data/processed --data-root data/chexpert

    # before real data -- synthetic tree + random-init encoder, every figure stamped SYNTHETIC:
    python scripts/visualize_preprocessing.py --synthetic-demo --out reports/figures/demo

With ``--synthetic-demo`` the encoder is randomly initialised (``weights=None``): the
feature-space and probe figures then only demonstrate that the machinery runs -- they are
NOT a concept-quality result and are labelled accordingly.

Exit codes: 0 ok  ·  2 artifacts / images not found.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ddera.config import (  # noqa: E402
    FIGURES_ROOT,
    PROCESSED_DATA_ROOT,
    ConceptSpec,
    EncoderConfig,
)
from ddera.data.dataset import CheXpertImageDataset  # noqa: E402
from ddera.data.transforms import build_eval_transform, build_train_transform  # noqa: E402
from ddera.features.cache import FeatureCache, Fingerprint  # noqa: E402
from ddera.features.probe import linear_probe_concept_auroc  # noqa: E402
from ddera.models.encoder import DenseNet121Encoder  # noqa: E402
from ddera.reporting import plots  # noqa: E402
from ddera.reporting.theme import apply_theme, save_figure  # noqa: E402
from ddera.seed import set_seed  # noqa: E402

DEFAULT_CONCEPTS = "configs/concepts/chexpert_v1.yaml"


def _build_synthetic(dest: Path, concepts: str) -> tuple[Path, Path]:
    from ddera.data.acquire import build_processed_dataset
    from ddera.data.synthetic import write_synthetic_chexpert_tree

    raw = write_synthetic_chexpert_tree(
        dest / "chexpert", n_patients=90, with_images=True, image_size=224, seed=0
    )
    out = dest / "processed"
    build_processed_dataset(raw, ConceptSpec.from_yaml(concepts), out, seed=42, check_images=True)
    return out, raw


def _first_rgb_image(dataset: CheXpertImageDataset) -> np.ndarray:
    path = dataset.data_root / dataset.paths[0]
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--processed", type=Path, default=PROCESSED_DATA_ROOT)
    parser.add_argument(
        "--data-root", type=Path, default=None, help="dir the manifest paths resolve against"
    )
    parser.add_argument("--concepts", default=DEFAULT_CONCEPTS)
    parser.add_argument("--cache", type=Path, default=None, help="an existing feature cache dir")
    parser.add_argument("--out", type=Path, default=FIGURES_ROOT / "preprocessing")
    parser.add_argument("--synthetic-demo", action="store_true")
    args = parser.parse_args(argv)

    set_seed(42)
    apply_theme()
    synthetic = args.synthetic_demo
    tmp: tempfile.TemporaryDirectory | None = None

    processed, data_root = args.processed, args.data_root
    if synthetic:
        tmp = tempfile.TemporaryDirectory()
        processed, data_root = _build_synthetic(Path(tmp.name), args.concepts)

    splits_path = processed / "splits.parquet"
    if not splits_path.exists() or data_root is None:
        print(
            f"ERROR: need {splits_path} and --data-root (or use --synthetic-demo).",
            file=sys.stderr,
        )
        return 2

    spec = ConceptSpec.from_yaml(args.concepts)
    enc_cfg = EncoderConfig(weights=None) if synthetic else EncoderConfig()
    eval_tf = build_eval_transform(enc_cfg)
    train_ds = CheXpertImageDataset(
        splits_path, "train", spec, data_root=data_root, transform=eval_tf
    )

    args.out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    sample = _first_rgb_image(train_ds)
    fig = plots.plot_augmentation_grid(sample, build_train_transform(None, enc_cfg), n=8)
    written.append(save_figure(fig, args.out / "augmentation_grid.png", synthetic=synthetic))

    fig = plots.plot_preprocessed_samples(train_ds, n=8, encoder=enc_cfg)
    written.append(save_figure(fig, args.out / "preprocessed_samples.png", synthetic=synthetic))

    # --- feature cache + probe -----------------------------------------------------------
    cache_dir = args.cache
    encoder = DenseNet121Encoder(enc_cfg, frozen=True)
    if cache_dir is None:
        cache_dir = Path(tmp.name) / "features" if tmp else args.out / "_features"
        FeatureCache(cache_dir).extract(encoder, train_ds, "train", batch_size=16)
        eval_ds = CheXpertImageDataset(
            splits_path, "val", spec, data_root=data_root, transform=eval_tf
        )
        FeatureCache(cache_dir).extract(encoder, eval_ds, "val", batch_size=16)

    cache = FeatureCache(cache_dir)
    feats_tr, idx_tr, _ = cache.load("train", expected=Fingerprint.from_encoder(encoder))

    # cache round-trip sanity: cached vs a fresh forward pass on the first few rows
    import torch

    with torch.no_grad():
        fresh = encoder(torch.stack([train_ds[k]["image"] for k in range(min(16, len(train_ds)))]))
    fig = plots.plot_feature_cache_sanity(feats_tr[: fresh.shape[0]], fresh.numpy())
    written.append(save_figure(fig, args.out / "feature_cache_sanity.png", synthetic=synthetic))

    fig = plots.plot_feature_space(feats_tr, train_ds.targets, colour_label="target")
    written.append(save_figure(fig, args.out / "feature_space_pca.png", synthetic=synthetic))

    try:
        feats_ev, _, _ = cache.load("val", expected=Fingerprint.from_encoder(encoder))
        from ddera.data.dataset import EncodedSplit

        ev_enc = EncodedSplit(splits_path, "val", spec)
        probe = linear_probe_concept_auroc(
            feats_tr,
            train_ds.concept_labels,
            train_ds.concept_mask,
            feats_ev,
            ev_enc.concept_labels,
            ev_enc.concept_mask,
            list(spec.concepts),
        )
        fig = plots.plot_concept_probe_auroc(probe)
        written.append(save_figure(fig, args.out / "concept_probe_auroc.png", synthetic=synthetic))
    except FileNotFoundError:
        print("  (no val cache -> skipping the concept probe figure)")

    if tmp is not None:
        tmp.cleanup()
    tag = "  (SYNTHETIC)" if synthetic else ""
    for p in written:
        print(f"  wrote {p}")
    print(f"\n{len(written)} figures -> {args.out}{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
