"""Training loop: train / validate / early-stop, regime-aware (ARCHITECTURE stage 6/7).

One loop serves B0 and every CBM regime. For B0 the loss is plain target BCE; for a CBM it
is :func:`ddera.train.losses.cbm_loss` under the variant's regime. The M1 / M2 / M3
gradient behaviour (ground-truth vs predicted concepts, detach vs joint) is baked into
:class:`ddera.models.cbm.ConceptBottleneckModel`, so the loop does not branch on it beyond
picking the loss.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ddera.config import ExperimentConfig
from ddera.train.amp import AmpContext
from ddera.train.losses import cbm_loss, target_bce


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    val_loss: float
    val_target_loss: float
    val_concept_loss: float


@dataclass
class TrainHistory:
    epochs: list[EpochStats] = field(default_factory=list)
    best_epoch: int = 0
    best_val: float = float("inf")
    stopped_early: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": [vars(e) for e in self.epochs],
            "best_epoch": self.best_epoch,
            "best_val": self.best_val,
            "stopped_early": self.stopped_early,
        }


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _loss_for(model: nn.Module, batch: dict, *, regime: str, concept_weight: float) -> dict:
    out = model(batch)
    if "concept_logits" not in out:  # B0 black box
        loss = target_bce(out["target_logit"], batch["target"])
        zero = torch.zeros((), device=loss.device)
        return {"loss": loss, "target_loss": loss.detach(), "concept_loss": zero}
    return cbm_loss(out, batch, regime=regime, concept_weight=concept_weight)


@torch.no_grad()
def _validate(
    model, loader, device, *, regime, concept_weight, max_batches=None
) -> tuple[float, float, float]:
    model.eval()
    tot = tot_t = tot_c = n = 0.0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        parts = _loss_for(
            model, _to_device(batch, device), regime=regime, concept_weight=concept_weight
        )
        tot += float(parts["loss"])
        tot_t += float(parts["target_loss"])
        tot_c += float(parts["concept_loss"])
        n += 1
    n = max(n, 1.0)
    return tot / n, tot_t / n, tot_c / n


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: ExperimentConfig,
    *,
    device: torch.device,
    regime: str,
    amp: AmpContext | None = None,
    fast_dev_run: bool = False,
    on_epoch: Callable[[EpochStats], None] | None = None,
) -> TrainHistory:
    """Train ``model`` and return the history. The model is left holding the best-val weights."""
    amp = amp or AmpContext.disabled()
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    epochs = 1 if fast_dev_run else cfg.epochs
    max_batches = 2 if fast_dev_run else None
    accum = max(1, cfg.grad_accum_steps)

    history = TrainHistory()
    best_state = copy.deepcopy(model.state_dict())
    patience = 0
    is_cbm = hasattr(model, "reasoner")  # BlackBoxModel has no reasoner

    for epoch in range(epochs):
        model.train()
        running = count = 0.0
        optimiser.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = _to_device(batch, device)
            with amp.autocast():
                parts = _loss_for(
                    model, batch, regime=regime, concept_weight=cfg.concept_loss_weight
                )
            (parts["loss"] / accum).backward()
            if (i + 1) % accum == 0:
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
            running += float(parts["loss"].detach())
            count += 1
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)

        val_loss, val_t, val_c = _validate(
            model,
            val_loader,
            device,
            regime=regime,
            concept_weight=cfg.concept_loss_weight,
            max_batches=max_batches,
        )
        stats = EpochStats(epoch, running / max(count, 1.0), val_loss, val_t, val_c)
        history.epochs.append(stats)
        if on_epoch is not None:
            on_epoch(stats)

        # Select on the target loss -- that is what the run is judged on.
        score = val_t if is_cbm else val_loss
        if score < history.best_val - 1e-6:
            history.best_val = score
            history.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if not fast_dev_run and patience >= cfg.early_stop_patience:
                history.stopped_early = True
                break

    model.load_state_dict(best_state)
    return history
