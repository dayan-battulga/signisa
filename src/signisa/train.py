"""Training loop: AdamW + warmup->cosine schedule, AMP on CUDA, early stopping
on val top-1. All knobs come from signisa.config.Config."""

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from .config import Config
from .data import pad_collate
from .models import SignModel


class LengthBucketSampler(Sampler):
    """Yield batches of similar-length sequences.

    Sequences are stored at native length, so pad-to-batch-max wastes compute — and
    skews the BatchNorms — in proportion to the length spread inside a batch. Real
    clips run 6 to 384 frames with a median near 40, so uniformly random batches would
    be mostly padding. Sorting inside a shuffled pool keeps batch composition close to
    random while collapsing that spread.
    """

    def __init__(self, lengths, batch_size: int, shuffle: bool, seed: int,
                 pool_batches: int = 50):
        self.lengths = np.asarray(lengths)
        self.batch_size, self.shuffle = batch_size, shuffle
        self.pool = batch_size * pool_batches
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        n = len(self.lengths)
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)
        batches = []
        for start in range(0, n, self.pool):
            chunk = order[start:start + self.pool]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            batches += [chunk[i:i + self.batch_size].tolist()
                        for i in range(0, len(chunk), self.batch_size)]
        if self.shuffle:
            self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)


def make_loader(dataset: Dataset, cfg: Config, shuffle: bool) -> DataLoader:
    common = dict(collate_fn=pad_collate, num_workers=cfg.num_workers,
                  pin_memory=torch.cuda.is_available())
    lengths = getattr(dataset, "lengths", None)  # a Subset (smoke tests) has none
    if lengths is None:
        return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle, **common)
    return DataLoader(dataset, batch_sampler=LengthBucketSampler(
        lengths, cfg.batch_size, shuffle, cfg.seed), **common)


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
    for x, mask, side, y in loader:
        # no labels -> plain cosine logits for arcface too
        _, logits = model(x.to(device), mask.to(device), side.to(device))
        correct += (logits.argmax(dim=1).cpu() == y).sum().item()
        total += len(y)
    return correct / max(total, 1)


def _report_projected_time(elapsed_s: float, epochs_done: int, cfg: Config) -> None:
    """Kaggle sessions are wall-clock capped, so say early whether the run fits."""
    per_epoch = elapsed_s / epochs_done
    hours = per_epoch * cfg.epochs / 3600
    print(f"throughput: {per_epoch:.1f} s/epoch -> projected {hours:.1f} h "
          f"for {cfg.epochs} epochs")
    if hours > cfg.session_budget_h:
        print(f"WARNING: over the {cfg.session_budget_h:g} h session budget — "
              f"restart with epochs <= {int(cfg.session_budget_h * cfg.epochs / hours)}")


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
    started = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for x, mask, side, y in train_loader:
            x, mask, side, y = x.to(device), mask.to(device), side.to(device), y.to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                _, logits = model(x, mask, side, y)
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
        if epoch == 2:  # three full epochs in, throughput has settled
            _report_projected_time(time.perf_counter() - started, 3, cfg)
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
