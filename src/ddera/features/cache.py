"""Frozen-encoder feature cache (ADR-008).

Run the frozen DenseNet-121 once over each split and store the 1024-d vectors as a
``float16`` memmap, with a ``<split>_index.parquet`` (row -> source path) and a
``fingerprint.json`` alongside. Every frozen-encoder CBM variant then trains on these
instead of re-running the CNN.

:meth:`FeatureCache.load` recomputes the expected fingerprint from the current encoder and
refuses a cache that does not match: silently reusing features from a different encoder,
resolution or normalisation would invalidate every downstream metric. M3 (joint) and B0
fine-tune the encoder -- their features are NOT these, which the fingerprint enforces rather
than trusting to discipline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ddera.models.encoder import DenseNet121Encoder
from ddera.seed import make_generator, seed_worker


class StaleCacheError(RuntimeError):
    """Raised when a cached feature set does not match the current encoder fingerprint."""


@dataclass(frozen=True)
class Fingerprint:
    """The encoder identity a cached feature set was produced with (ADR-008)."""

    arch: str
    weights: str | None
    weights_sha256: str
    resolution: int
    channels: int
    feature_dim: int
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]

    @classmethod
    def from_encoder(cls, encoder: DenseNet121Encoder) -> Fingerprint:
        fp = encoder.fingerprint()
        return cls(
            arch=fp["arch"],
            weights=fp["weights"],
            weights_sha256=fp["weights_sha256"],
            resolution=int(fp["resolution"]),
            channels=int(fp["channels"]),
            feature_dim=int(fp["feature_dim"]),
            normalization_mean=tuple(fp["normalization_mean"]),
            normalization_std=tuple(fp["normalization_std"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> Fingerprint:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            arch=raw["arch"],
            weights=raw["weights"],
            weights_sha256=raw["weights_sha256"],
            resolution=int(raw["resolution"]),
            channels=int(raw["channels"]),
            feature_dim=int(raw["feature_dim"]),
            normalization_mean=tuple(raw["normalization_mean"]),
            normalization_std=tuple(raw["normalization_std"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "arch": self.arch,
            "weights": self.weights,
            "weights_sha256": self.weights_sha256,
            "resolution": self.resolution,
            "channels": self.channels,
            "feature_dim": self.feature_dim,
            "normalization_mean": list(self.normalization_mean),
            "normalization_std": list(self.normalization_std),
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def matches(self, other: Fingerprint) -> bool:
        return self.to_dict() == other.to_dict()


class FeatureCache:
    """Reads and writes ``<root>/<split>.npy`` + ``<split>_index.parquet`` + ``fingerprint.json``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _split_files(self, split: str) -> tuple[Path, Path]:
        return self.root / f"{split}.npy", self.root / f"{split}_index.parquet"

    @property
    def fingerprint_path(self) -> Path:
        return self.root / "fingerprint.json"

    def extract(
        self,
        encoder: DenseNet121Encoder,
        dataset: Any,
        split: str,
        *,
        batch_size: int = 32,
        num_workers: int = 0,
        device: str | torch.device | None = None,
        seed: int = 42,
        autocast: bool | None = None,
    ) -> Fingerprint:
        """Run ``encoder`` over ``dataset`` (a :class:`~ddera.data.dataset.CheXpertImageDataset`)
        in fixed order and write the cache. Returns the fingerprint it stamped.
        """
        device = torch.device(device) if device is not None else torch.device("cpu")
        encoder = encoder.to(device).eval()
        if autocast is None:
            autocast = device.type == "cuda"

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # order must match index.parquet and the split
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=make_generator(seed),
        )

        n = len(dataset)
        feats = np.empty((n, encoder.feature_dim), dtype=np.float16)
        pos = 0
        with torch.no_grad():
            for batch in loader:
                x = batch["image"].to(device)
                if autocast:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        out = encoder(x)
                else:
                    out = encoder(x)
                out = out.float().cpu().numpy().astype(np.float16)
                feats[pos : pos + len(out)] = out
                pos += len(out)
        if pos != n:
            raise RuntimeError(f"Extracted {pos} rows, expected {n}.")

        npy_path, idx_path = self._split_files(split)
        self.root.mkdir(parents=True, exist_ok=True)
        memmap = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.float16, shape=feats.shape)
        memmap[:] = feats
        memmap.flush()
        del memmap

        pd.DataFrame({"row": np.arange(n), "path": list(dataset.paths)}).to_parquet(
            idx_path, index=False
        )
        fingerprint = Fingerprint.from_encoder(encoder)
        fingerprint.to_json(self.fingerprint_path)
        return fingerprint

    def load(
        self, split: str, *, expected: Fingerprint | None = None
    ) -> tuple[np.memmap, pd.DataFrame, Fingerprint]:
        """Return ``(features_memmap, index_df, fingerprint)``.

        Raises :class:`StaleCacheError` if ``expected`` is given and does not match, if the
        fingerprint file is missing, or if the feature/index lengths disagree.
        """
        npy_path, idx_path = self._split_files(split)
        if not npy_path.exists():
            raise FileNotFoundError(f"No cached features for split {split!r} at {npy_path}")
        if not self.fingerprint_path.exists():
            raise StaleCacheError(f"{self.fingerprint_path} is missing; rebuild the cache.")

        fingerprint = Fingerprint.from_json(self.fingerprint_path)
        if expected is not None and not fingerprint.matches(expected):
            raise StaleCacheError(
                "Cached features do not match the current encoder -- rebuild the cache.\n"
                f"  cached  : {fingerprint.to_dict()}\n"
                f"  expected: {expected.to_dict()}"
            )

        features = np.load(npy_path, mmap_mode="r")
        index_df = pd.read_parquet(idx_path)
        if len(features) != len(index_df):
            raise StaleCacheError(
                f"feature rows ({len(features)}) != index rows ({len(index_df)}) for {split!r}"
            )
        return features, index_df, fingerprint

    def is_fresh(self, encoder: DenseNet121Encoder, split: str) -> bool:
        try:
            self.load(split, expected=Fingerprint.from_encoder(encoder))
            return True
        except (FileNotFoundError, StaleCacheError):
            return False
