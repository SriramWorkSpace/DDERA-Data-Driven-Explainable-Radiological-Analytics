"""Phase-1/2 progress figures: dataset, split and concept-label distributions.

Reads the Phase-1 artifacts (``manifest.parquet``, ``splits.parquet``) and writes PNGs into
``--out``. With ``--synthetic-demo`` it first builds a synthetic CheXpert tree and runs the
Phase-1 pipeline on it, so figures can be produced before the real dataset is in place --
every such figure is stamped ``SYNTHETIC``.

    python scripts/visualize_dataset.py --processed data/processed
    python scripts/visualize_dataset.py --synthetic-demo --out reports/figures/demo

Exit codes: 0 ok  ·  2 artifacts not found (and --synthetic-demo not given).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ddera.config import FIGURES_ROOT, PROCESSED_DATA_ROOT, ConceptSpec  # noqa: E402
from ddera.data.chexpert import load_manifest  # noqa: E402
from ddera.data.splits import load_splits  # noqa: E402
from ddera.reporting import plots  # noqa: E402
from ddera.reporting.theme import apply_theme, save_figure  # noqa: E402
from ddera.seed import set_seed  # noqa: E402

DEFAULT_CONCEPTS = "configs/concepts/chexpert_v1.yaml"


def _build_synthetic(dest: Path, concepts: str) -> Path:
    """Synthetic CheXpert tree -> Phase-1 pipeline -> processed dir. Returns the dir."""
    from ddera.data.acquire import build_processed_dataset
    from ddera.data.synthetic import write_synthetic_chexpert_tree

    raw = write_synthetic_chexpert_tree(dest / "chexpert", n_patients=120, seed=0)
    out = dest / "processed"
    build_processed_dataset(raw, ConceptSpec.from_yaml(concepts), out, seed=42, check_images=False)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--processed", type=Path, default=PROCESSED_DATA_ROOT)
    parser.add_argument("--concepts", default=DEFAULT_CONCEPTS)
    parser.add_argument("--out", type=Path, default=FIGURES_ROOT / "dataset")
    parser.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="build a synthetic dataset and plot that (figures stamped SYNTHETIC)",
    )
    args = parser.parse_args(argv)

    set_seed(42)
    apply_theme()

    synthetic = args.synthetic_demo
    tmp: tempfile.TemporaryDirectory | None = None
    processed = args.processed
    if synthetic:
        tmp = tempfile.TemporaryDirectory()
        processed = _build_synthetic(Path(tmp.name), args.concepts)

    manifest_path = processed / "manifest.parquet"
    splits_path = processed / "splits.parquet"
    if not manifest_path.exists() or not splits_path.exists():
        print(
            f"ERROR: {manifest_path} / {splits_path} not found. Run scripts/get_data.py "
            "first, or pass --synthetic-demo.",
            file=sys.stderr,
        )
        return 2

    spec = ConceptSpec.from_yaml(args.concepts)
    manifest, _ = load_manifest(manifest_path)
    splits_df, _ = load_splits(splits_path)
    observations = [spec.target, *spec.concepts]

    figures = {
        "split_summary.png": plots.plot_split_summary(splits_df),
        "target_prevalence_by_split.png": plots.plot_target_prevalence_by_split(splits_df),
        "label_distribution.png": plots.plot_label_distribution(manifest, observations),
        "concept_cooccurrence.png": plots.plot_concept_cooccurrence(manifest, spec.concepts),
        "concept_mask_coverage.png": plots.plot_mask_coverage(manifest, spec),
    }
    for name, fig in figures.items():
        written = save_figure(fig, args.out / name, synthetic=synthetic)
        print(f"  wrote {written}")

    if tmp is not None:
        tmp.cleanup()
    tag = "  (SYNTHETIC)" if synthetic else ""
    print(f"\n{len(figures)} figures -> {args.out}{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
