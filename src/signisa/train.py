"""Training loop: AdamW + warmup->cosine schedule, AMP on CUDA, early stopping
on val top-1. All knobs come from signisa.config.Config."""

import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .models import SignModel


def make_loader(dataset: Dataset, cfg: Config, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=False)


def warmup_cosine(cfg: Config, steps_per_epoch: int):
    warm = max(1, cfg.warmup_epochs * steps_per_epoch)
    total = max(warm + 1, cfg.epochs * steps_per_epoch)

    def factor(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / (total - warm)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


@torch.no_grad()
def top1_accuracy(model: SignModel, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        _, logits = model(x.to(device))  # no labels -> plain cosine logits for arcface too
        correct += (logits.argmax(dim=1).cpu() == y).sum().item()
        total += len(y)
    return correct / max(total, 1)


def train_model(cfg: Config, train_ds: Dataset, val_ds: Dataset,
                device: str = "cpu", max_steps: int | None = None) -> tuple[SignModel, dict]:
    """Returns (best model by val top-1, history). max_steps caps total steps for smoke tests."""
    torch.manual_seed(cfg.seed)
    for ds in (train_ds, val_ds):  # a v1 model must never silently consume v2 tensors
        version = getattr(ds, "landmark_version", None)
        assert version is None or version == cfg.landmark_version, (
            f"dataset is {version} but the model expects {cfg.landmark_version}")
    model = SignModel(cfg).to(device)
    train_loader = make_loader(train_ds, cfg, shuffle=True)
    val_loader = make_loader(val_ds, cfg, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(cfg, len(train_loader)))
    use_amp = cfg.amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    history = {"train_loss": [], "val_top1": []}
    best_top1, best_state, patience_left, step = -1.0, None, cfg.patience, 0
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                _, logits = model(x, y)
                loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1
            step += 1
            if max_steps and step >= max_steps:
                break
        history["train_loss"].append(epoch_loss / max(n_batches, 1))

        val_top1 = top1_accuracy(model, val_loader, device)
        history["val_top1"].append(val_top1)
        if val_top1 > best_top1:
            best_top1, patience_left = val_top1, cfg.patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left == 0:
                break
        if max_steps and step >= max_steps:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
