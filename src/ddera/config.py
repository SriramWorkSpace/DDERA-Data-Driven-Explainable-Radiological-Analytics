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
EXPERIMENT_ROOT = REPO_ROOT / "experiments"
RUNS_ROOT = EXPERIMENT_ROOT / "runs"


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
