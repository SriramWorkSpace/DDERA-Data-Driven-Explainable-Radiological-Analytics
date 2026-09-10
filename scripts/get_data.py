"""Phase-1 CheXpert acquisition.

Verify a *local* CheXpert download, build the manifest and patient-level splits, and persist
them as parquet (+ JSON sidecars). Wires together ``ddera.data.chexpert``,
``ddera.data.labels`` and ``ddera.data.splits`` under the contract in
``configs/concepts/chexpert_v1.yaml``.

This script does NOT download CheXpert -- it needs a Stanford AIMI research-use agreement
(ADR-002). Point ``--dest`` at your extracted ``CheXpert-v1.0-small`` directory.

    python scripts/get_data.py --dest data/chexpert
    python scripts/get_data.py --dest data/chexpert --no-check-images --out data/processed

Exit codes: 0 ok  ·  1 split-integrity failure (or --strict issue)  ·  2 dataset not found.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ddera.config import PROCESSED_DATA_ROOT, ConceptSpec  # noqa: E402
from ddera.data.acquire import (  # noqa: E402
    ACQUISITION_HELP,
    build_processed_dataset,
    inspect_download,
)
from ddera.seed import set_seed  # noqa: E402

DEFAULT_CONCEPTS = "configs/concepts/chexpert_v1.yaml"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dest",
        required=True,
        type=Path,
        help="extracted CheXpert dir (contains train.csv or CheXpert-v1.0-small/)",
    )
    p.add_argument(
        "--concepts", default=DEFAULT_CONCEPTS, help="concept spec YAML (default: %(default)s)"
    )
    p.add_argument(
        "--out",
        type=Path,
        default=PROCESSED_DATA_ROOT,
        help="output dir for processed artifacts (default: %(default)s)",
    )
    p.add_argument("--seed", type=int, default=42, help="patient-split seed (default: %(default)s)")

    images = p.add_mutually_exclusive_group()
    images.add_argument(
        "--check-images",
        dest="check_images",
        action="store_true",
        default=None,
        help="probe image files for existence/decodability/size (default: auto)",
    )
    images.add_argument(
        "--no-check-images",
        dest="check_images",
        action="store_false",
        help="skip image probing (CSV-only)",
    )
    p.add_argument(
        "--drop-unreadable",
        action="store_true",
        help="drop rows whose image is missing/corrupt (still reported)",
    )
    p.add_argument(
        "--no-valid",
        dest="include_valid",
        action="store_false",
        help="do not build external.parquet from valid.csv",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on any integrity issue, not just patient leakage",
    )
    p.add_argument("--json", type=Path, help="also write the run summary as JSON here")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    set_seed(args.seed)

    layout = inspect_download(args.dest)
    print(layout.describe())
    if not layout.ok:
        print(f"\nERROR: {ACQUISITION_HELP}", file=sys.stderr)
        return 2

    spec = ConceptSpec.from_yaml(args.concepts)
    print(f"\nconcept spec: {spec.name}  target={spec.target!r}  n_concepts={spec.n_concepts}")

    try:
        artifacts = build_processed_dataset(
            args.dest,
            spec,
            args.out,
            seed=args.seed,
            check_images=args.check_images,
            drop_unreadable=args.drop_unreadable,
            include_valid=args.include_valid,
        )
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:  # patient leakage -> nothing was written
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print("\n" + artifacts.summary())

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(artifacts.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        print(f"\nsummary JSON -> {args.json}")

    issues = artifacts.integrity.get("issues", [])
    if issues:
        print("\nintegrity issues (not patient leakage):")
        for issue in issues:
            print(f"  - {issue}")
        if args.strict:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
