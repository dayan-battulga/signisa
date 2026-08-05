"""CPU smoke tests for the training stack: slim shards -> Dataset reconstruction,
augmentations, overfit sanity (both losses), and the end-to-end eval path."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import Subset

from signisa.config import AugmentConfig, Config
from signisa.data import ShardDataset, augmented
from signisa.eval import run_evaluation
from signisa.models import SignModel, parameter_count
from signisa.train import top1_accuracy, train_model

ROOT = Path(__file__).parent.parent
SAMPLES = ROOT / "data" / "samples"
pytestmark = pytest.mark.skipif(
    not SAMPLES.exists() or not any(SAMPLES.glob("*.parquet")), reason="no local samples")


@pytest.fixture(scope="session")
def tensors_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("tensors")
    subprocess.run(
        [sys.executable, "scripts/build_training_tensors.py",
         "--landmarks-dir", str(SAMPLES), "--out-dir", str(out)],
        check=True, cwd=ROOT)
    return out


def small_config(loss: str, **overrides) -> Config:
    defaults = dict(loss=loss, dim=64, n_conv_blocks=1, n_transformer_layers=1,
                    embed_dim=64, batch_size=10, num_workers=0, amp=False,
                    lr=1e-2, epochs=50, patience=100)
    defaults.update(overrides)
    return Config(**defaults)


def test_shards_store_4_channels_and_dataset_reconstructs(tensors_dir):
    from signisa.preprocess.kaggle import load_holistic
    from signisa.preprocess.pipeline import preprocess

    shard = np.load(tensors_dir / "shard_0000.npz")
    assert shard["tensors"].shape[1:] == (160, 65, 4)
    assert shard["tensors"].dtype == np.float16

    ds = ShardDataset(tensors_dir)
    tensor, label = ds[0]
    assert tensor.shape == (160, 65, 10) and tensor.dtype == torch.float32
    assert isinstance(label, int)

    row = ds.index.iloc[0]
    full = preprocess(load_holistic(SAMPLES / f"{row.sequence_id}.parquet"),
                      left_dominant=bool(row.mirrored)).tensor
    np.testing.assert_allclose(tensor.numpy(), full, atol=1e-2)  # float16 storage rounding


def test_augmentations_shapes_and_identity():
    rng = np.random.default_rng(3)
    coords = rng.normal(size=(160, 65, 3))
    confidence = np.ones((160, 65, 1))
    confidence[:, 5] = 0.0
    coords[:, 5] = 0.0

    off = AugmentConfig(mask_total_frac=0.0, rotation_deg=0.0, scale=0.0,
                        translation=0.0, node_dropout_p=0.0, noise_sigma=0.0)
    c2, f2 = augmented(coords.copy(), confidence.copy(), off, rng)
    np.testing.assert_allclose(c2, coords)
    np.testing.assert_allclose(f2, confidence)

    c3, f3 = augmented(coords.copy(), confidence.copy(), AugmentConfig(), rng)
    assert c3.shape == coords.shape and f3.shape == confidence.shape
    assert (c3[f3[..., 0] == 0.0] == 0.0).all()  # masked frames/nodes fully zeroed
    # never a flip: the mean x sign of a strongly right-biased cloud can't invert at +-5 deg
    biased = np.abs(coords)
    c4, _ = augmented(biased.copy(), np.ones((160, 65, 1)), AugmentConfig(), rng)
    assert c4[..., 0].sum() > 0


@pytest.mark.parametrize("loss", ["ce", "arcface"])
def test_overfit_ten_sequences(tensors_dir, loss):
    ds = Subset(ShardDataset(tensors_dir), range(10))
    cfg = small_config(loss)
    model, history = train_model(cfg, ds, ds, max_steps=50)
    assert history["train_loss"][-1] < history["train_loss"][0]
    loader = torch.utils.data.DataLoader(ds, batch_size=10)
    assert top1_accuracy(model, loader, "cpu") == 1.0


def test_end_to_end_mini_run(tensors_dir, tmp_path):
    ds = ShardDataset(tensors_dir)
    pids = sorted(ds.index.participant_id.unique())
    val_pids = pids[:2]
    cfg = small_config("ce", epochs=2, patience=10)
    train_ds = ShardDataset(tensors_dir, augment=True,
                            participants=[p for p in pids if p not in val_pids])
    val_ds = ShardDataset(tensors_dir, participants=val_pids)
    model, _ = train_model(cfg, train_ds, val_ds)

    metrics = run_evaluation(model, tensors_dir, ROOT / "data/meta/curriculum_db.json",
                             ROOT / "data/meta/training_labels.json", cfg, tmp_path,
                             val_participants=val_pids)
    report = (tmp_path / "metrics_report.md").read_text()
    assert "TAR@FAR" in report and "Cluster collapse" in report
    assert 0.0 <= metrics["tar_at_far"] <= 1.0
    assert 0.0 <= metrics["top1_closed_set"] <= 1.0

    trained = json.load((tmp_path / "curriculum_db_trained.json").open())
    assert trained["signs"].keys() == json.load(
        (ROOT / "data/meta/curriculum_db.json").open())["signs"].keys()
    filled = [s for s, e in trained["signs"].items() if e["centroid"] is not None]
    assert filled, "no centroids were filled"
    assert all(len(trained["signs"][s]["centroid"]) == cfg.embed_dim for s in filled)


def test_parameter_budget():
    assert parameter_count(SignModel(Config())) < 2_000_000
