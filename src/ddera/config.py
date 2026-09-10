"""Configuration loading.

Every experiment is fully described by YAML. Nothing is hardcoded in a notebook, and the
resolved config is copied into the run directory so any result can be reproduced exactly.

The concept specification (``ConceptSpec``) is deliberately domain-agnostic: swapping
``configs/concepts/chexpert_v1.yaml`` for ``vindr_v1.yaml`` is the *only* change Phase 8
is permitted to require. If a new domain needs more than that, the generality claim
(Invariant 9) has failed and we report that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs"
DATA_ROOT = REPO_ROOT / "data"
#: Derived data-pipeline artifacts (manifest.parquet, splits.parquet, ...). Gitignored via
#: ``/data/``. Kept separate from the raw download so ``data/chexpert/`` stays untouched.
PROCESSED_DATA_ROOT = DATA_ROOT / "processed"
#: Cached encoder features (ADR-008): ``data/features/<split>.npy`` + index + fingerprint.
#: Under ``/data/`` so it never enters git.
FEATURES_ROOT = DATA_ROOT / "features"
EXPERIMENT_ROOT = REPO_ROOT / "experiments"
RUNS_ROOT = EXPERIMENT_ROOT / "runs"
#: Exported figures for notebooks, scripts and the progress report. The directory is
#: tracked; the generated image files are gitignored (reproducible from the pipeline).
REPORTS_ROOT = REPO_ROOT / "reports"
FIGURES_ROOT = REPORTS_ROOT / "figures"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file, resolving bare names against ``configs/``."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        candidate = CONFIG_ROOT / p
        if candidate.exists():
            p = candidate
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with p.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class UncertaintySpec:
    """How to treat CheXpert-style uncertain (-1) and blank labels. See ADR-004."""

    concept_policy: str = "u_mask"
    target_policy: str = "u_ignore"
    blank_policy: str = "negative"
    sensitivity_analysis: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EscalationSpec:
    """Pre-committed rule for switching target when the primary one is too sparse (ADR-003)."""

    min_test_positives: int = 250
    max_auroc_ci_width: float = 0.10
    fallback_targets: list[str] = field(default_factory=list)

    def should_escalate(self, n_test_positives: int, auroc_ci_width: float) -> tuple[bool, str]:
        """Apply the rule. Returns (escalate, human-readable reason)."""
        reasons = []
        if n_test_positives < self.min_test_positives:
            reasons.append(f"only {n_test_positives} test positives (< {self.min_test_positives})")
        if auroc_ci_width > self.max_auroc_ci_width:
            reasons.append(f"AUROC 95% CI width {auroc_ci_width:.3f} (> {self.max_auroc_ci_width})")
        return bool(reasons), "; ".join(reasons) if reasons else "criteria met, no escalation"


@dataclass(frozen=True)
class CohortSpec:
    """Which images are in scope, and how they are split. See ADR-005."""

    views: list[str] = field(default_factory=lambda: ["AP", "PA"])
    split_level: str = "patient"
    split_ratios: dict[str, float] = field(
        default_factory=lambda: {"train": 0.70, "val": 0.10, "test": 0.20}
    )
    stratify_on: str = "target"
    external_eval: str | None = None


@dataclass(frozen=True)
class ConceptSpec:
    """The concept/target contract for one domain.

    This is the object the whole codebase reasons about. Nothing downstream knows or cares
    that the concepts happen to be radiographic findings.
    """

    name: str
    domain: str
    dataset: str
    target: str
    concepts: list[str]
    uncertainty: UncertaintySpec = field(default_factory=UncertaintySpec)
    escalation: EscalationSpec = field(default_factory=EscalationSpec)
    cohort: CohortSpec = field(default_factory=CohortSpec)
    expected_leakage_watchlist: list[str] = field(default_factory=list)
    excluded: list[dict[str, str]] = field(default_factory=list)

    @property
    def n_concepts(self) -> int:
        return len(self.concepts)

    def __post_init__(self) -> None:
        if self.target in self.concepts:
            raise ValueError(
                f"Target {self.target!r} also appears in the concept list. That makes the "
                "bottleneck circular and violates Invariant 4."
            )
        if len(set(self.concepts)) != len(self.concepts):
            dupes = [c for c in self.concepts if self.concepts.count(c) > 1]
            raise ValueError(f"Duplicate concepts: {sorted(set(dupes))}")
        if not self.concepts:
            raise ValueError("A concept bottleneck with zero concepts is not a bottleneck.")
        unknown = set(self.expected_leakage_watchlist) - set(self.concepts)
        if unknown:
            raise ValueError(f"Leakage watchlist names non-existent concepts: {sorted(unknown)}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ConceptSpec:
        raw = load_yaml(path)
        return cls(
            name=raw["name"],
            domain=raw["domain"],
            dataset=raw["dataset"],
            target=raw["target"],
            concepts=list(raw["concepts"]),
            uncertainty=UncertaintySpec(**raw.get("uncertainty", {})),
            escalation=EscalationSpec(
                min_test_positives=raw.get("escalation", {})
                .get("trigger", {})
                .get("min_test_positives", 250),
                max_auroc_ci_width=raw.get("escalation", {})
                .get("trigger", {})
                .get("max_auroc_ci_width", 0.10),
                fallback_targets=raw.get("escalation", {}).get("fallback_targets", []),
            ),
            cohort=CohortSpec(
                views=raw.get("cohort", {}).get("views", ["AP", "PA"]),
                split_level=raw.get("cohort", {}).get("split_level", "patient"),
                split_ratios=raw.get("cohort", {}).get(
                    "split_ratios", {"train": 0.70, "val": 0.10, "test": 0.20}
                ),
                stratify_on=raw.get("cohort", {}).get("stratify_on", "target"),
                external_eval=raw.get("cohort", {}).get("external_eval"),
            ),
            expected_leakage_watchlist=raw.get("expected_leakage_watchlist", []),
            excluded=raw.get("excluded", []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "dataset": self.dataset,
            "target": self.target,
            "concepts": self.concepts,
            "n_concepts": self.n_concepts,
            "uncertainty": self.uncertainty.__dict__,
            "cohort": self.cohort.__dict__,
            "expected_leakage_watchlist": self.expected_leakage_watchlist,
        }


@dataclass(frozen=True)
class EncoderConfig:
    """Vision encoder specification (ADR-001).

    Also the source of the feature-cache fingerprint (ADR-008): a change to any field here
    invalidates every cached feature, and ``ddera.features.cache`` refuses the stale cache.
    """

    arch: str = "densenet121"
    #: torchvision weights enum name, or ``None`` for random init (used only in tests).
    weights: str | None = "IMAGENET1K_V1"
    resolution: int = 224
    channels: int = 3
    feature_dim: int = 1024
    normalization_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalization_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        if self.arch != "densenet121":
            raise ValueError(f"Only 'densenet121' is supported in v1 (ADR-001); got {self.arch!r}.")
        if self.channels != 3:
            raise ValueError("DenseNet-121 ImageNet weights expect 3 input channels.")
        if len(self.normalization_mean) != 3 or len(self.normalization_std) != 3:
            raise ValueError("normalization_mean and normalization_std must each have 3 values.")
        if self.feature_dim != 1024:
            raise ValueError(f"densenet121 produces 1024-d features, not {self.feature_dim}.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> EncoderConfig:
        raw = load_yaml(path)
        norm = raw.get("normalization", {})
        return cls(
            arch=raw.get("arch", "densenet121"),
            weights=raw.get("weights", "IMAGENET1K_V1"),
            resolution=int(raw.get("resolution", 224)),
            channels=int(raw.get("channels", 3)),
            feature_dim=int(raw.get("feature_dim", 1024)),
            normalization_mean=tuple(norm.get("mean", (0.485, 0.456, 0.406))),
            normalization_std=tuple(norm.get("std", (0.229, 0.224, 0.225))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "arch": self.arch,
            "weights": self.weights,
            "resolution": self.resolution,
            "channels": self.channels,
            "feature_dim": self.feature_dim,
            "normalization_mean": list(self.normalization_mean),
            "normalization_std": list(self.normalization_std),
        }


@dataclass(frozen=True)
class TransformConfig:
    """Training augmentation knobs (ADR-006).

    There are deliberately **no** horizontal/vertical flip or transpose parameters: chest
    anatomy is not left-right symmetric and flipping corrupts the Cardiomegaly / Enlarged
    Cardiomediastinum concepts. ``ddera.data.transforms`` also asserts this structurally.
    """

    rotate_limit_deg: float = 7.0  # ADR-006: +/- 7 degrees
    scale_limit: float = 0.05  # ADR-006: +/- 5 %
    shift_limit: float = 0.05
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    augment_prob: float = 0.5

    @classmethod
    def from_yaml(cls, path: str | Path) -> TransformConfig:
        raw = load_yaml(path)
        return cls(
            rotate_limit_deg=float(raw.get("rotate_limit_deg", 7.0)),
            scale_limit=float(raw.get("scale_limit", 0.05)),
            shift_limit=float(raw.get("shift_limit", 0.05)),
            brightness_limit=float(raw.get("brightness_limit", 0.2)),
            contrast_limit=float(raw.get("contrast_limit", 0.2)),
            augment_prob=float(raw.get("augment_prob", 0.5)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rotate_limit_deg": self.rotate_limit_deg,
            "scale_limit": self.scale_limit,
            "shift_limit": self.shift_limit,
            "brightness_limit": self.brightness_limit,
            "contrast_limit": self.contrast_limit,
            "augment_prob": self.augment_prob,
        }


#: Model variants (ARCHITECTURE section 5). B0 and M4 are the only ones with a path from
#: encoder features to the target -- both exist solely to quantify the bottleneck's cost.
MODEL_VARIANTS = (
    "b0",  # black-box baseline: encoder -> Linear(1024 -> 1)
    "m1_independent",  # CBM, reasoner trained on ground-truth concepts
    "m2_sequential",  # CBM, reasoner trained on predicted concepts (detached)
    "m3_joint",  # CBM, joint L_target + lambda * L_concept (Phase 4)
    "m4_hybrid",  # CBM + residual channel of width k (Phase 4)
)


@dataclass(frozen=True)
class ModelConfig:
    """Which model to build. See ARCHITECTURE section 5 and ADR-001."""

    variant: str = "m2_sequential"
    n_concepts: int = 12
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    #: M1 / M2 / M4 keep the encoder frozen (ADR-008 cache path); B0 / M3 fine-tune it.
    freeze_encoder: bool = True
    reasoner_bias: bool = True
    concept_head_dropout: float = 0.0
    #: M4 only -- width of the residual channel that bypasses the bottleneck (Phase 4).
    residual_dim: int = 0

    def __post_init__(self) -> None:
        if self.variant not in MODEL_VARIANTS:
            raise ValueError(
                f"Unknown model variant {self.variant!r}; expected one of {MODEL_VARIANTS}"
            )
        if self.n_concepts < 1:
            raise ValueError("A concept bottleneck needs at least one concept.")
        if self.residual_dim and self.variant != "m4_hybrid":
            raise ValueError("residual_dim is only meaningful for the m4_hybrid variant.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        raw = load_yaml(path)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelConfig:
        enc = raw.get("encoder")
        encoder = (
            EncoderConfig(**_encoder_kwargs(enc)) if isinstance(enc, dict) else EncoderConfig()
        )
        return cls(
            variant=raw.get("variant", "m2_sequential"),
            n_concepts=int(raw.get("n_concepts", 12)),
            encoder=encoder,
            freeze_encoder=bool(raw.get("freeze_encoder", True)),
            reasoner_bias=bool(raw.get("reasoner_bias", True)),
            concept_head_dropout=float(raw.get("concept_head_dropout", 0.0)),
            residual_dim=int(raw.get("residual_dim", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "n_concepts": self.n_concepts,
            "encoder": self.encoder.to_dict(),
            "freeze_encoder": self.freeze_encoder,
            "reasoner_bias": self.reasoner_bias,
            "concept_head_dropout": self.concept_head_dropout,
            "residual_dim": self.residual_dim,
        }


def _encoder_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    norm = raw.get("normalization", {})
    kwargs = {
        k: raw[k] for k in ("arch", "weights", "resolution", "channels", "feature_dim") if k in raw
    }
    if "mean" in norm:
        kwargs["normalization_mean"] = tuple(norm["mean"])
    if "std" in norm:
        kwargs["normalization_std"] = tuple(norm["std"])
    return kwargs


@dataclass(frozen=True)
class ExperimentConfig:
    """A full experiment, resolved from ``configs/experiment/<name>.yaml``.

    The resolved config is copied verbatim into the run directory (``config.yaml``) so any
    result reproduces exactly. Data locations are ``None`` in the checked-in configs and
    filled once Phase 1/2 have run on the real download; ``--synthetic`` bypasses them.
    """

    name: str
    model: ModelConfig = field(default_factory=ModelConfig)
    seed: int = 42
    concepts: str = "chexpert_v1"  # ConceptSpec name or path
    transforms: str = "transforms_v1"
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 5
    concept_loss_weight: float = 1.0  # lambda for the joint regime (Phase 4)
    grad_accum_steps: int = 1
    num_workers: int = 0
    splits_parquet: str | None = None
    data_root: str | None = None
    feature_cache: str | None = None
    #: Scale lever (subset the cohort). Using this on real data requires its own ADR
    #: (Invariant 10 / ADR-009): it changes scale, never methodology.
    subset_patients: int | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        raw = load_yaml(path)
        model = raw.get("model")
        return cls(
            name=raw["name"],
            model=ModelConfig.from_dict(model) if isinstance(model, dict) else ModelConfig(),
            seed=int(raw.get("seed", 42)),
            concepts=raw.get("concepts", "chexpert_v1"),
            transforms=raw.get("transforms", "transforms_v1"),
            epochs=int(raw.get("epochs", 20)),
            batch_size=int(raw.get("batch_size", 64)),
            lr=float(raw.get("lr", 1e-3)),
            weight_decay=float(raw.get("weight_decay", 1e-4)),
            early_stop_patience=int(raw.get("early_stop_patience", 5)),
            concept_loss_weight=float(raw.get("concept_loss_weight", 1.0)),
            grad_accum_steps=int(raw.get("grad_accum_steps", 1)),
            num_workers=int(raw.get("num_workers", 0)),
            splits_parquet=raw.get("splits_parquet"),
            data_root=raw.get("data_root"),
            feature_cache=raw.get("feature_cache"),
            subset_patients=raw.get("subset_patients"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model.to_dict(),
            "seed": self.seed,
            "concepts": self.concepts,
            "transforms": self.transforms,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "early_stop_patience": self.early_stop_patience,
            "concept_loss_weight": self.concept_loss_weight,
            "grad_accum_steps": self.grad_accum_steps,
            "num_workers": self.num_workers,
            "splits_parquet": self.splits_parquet,
            "data_root": self.data_root,
            "feature_cache": self.feature_cache,
            "subset_patients": self.subset_patients,
        }
