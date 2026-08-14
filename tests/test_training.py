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

    # canonical-space flip == mirroring the raw sequence before preprocess — BIT-exact
    # on the float16 path (float16 rounding is sign-symmetric, so it commutes with the flip)
    from signisa.data import mirrored_stored
    flipped = mirrored_stored(ds.tensors[int(row.row)])
    other = preprocess(load_holistic(SAMPLES / f"{row.sequence_id}.parquet"),
                       left_dominant=not bool(row.mirrored)).tensor[..., [0, 1, 2, 9]]
    assert np.array_equal(flipped, other.astype(np.float16))


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
    # temporal mask budget: fully-masked frames never exceed mask_total_frac
    for _ in range(200):
        _, f = augmented(coords.copy(), np.ones((160, 65, 1)),
                         AugmentConfig(node_dropout_p=0.0), rng)
        assert (f[..., 0] == 0.0).all(axis=1).sum() <= int(0.4 * 160)
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
    assert "Per-participant" in report
    assert 0.0 <= metrics["tar_at_far"] <= 1.0
    assert 0.0 <= metrics["top1_closed_set"] <= 1.0

    assert set(metrics["per_participant"]) == set(val_pids)
    for v in metrics["per_participant"].values():
        assert v["mirrored_rate"] in (0.0, 1.0)  # vote is per participant, so uniform
    trials = metrics["trials"]
    assert list(trials.columns) == ["sequence_id", "participant", "attempt",
                                    "target", "score", "genuine"]
    assert set(trials.participant) <= set(val_pids)
    # every genuine trial has exactly one row per val curriculum attempt
    assert trials[trials.genuine].sequence_id.is_unique

    trained = json.load((tmp_path / "curriculum_db_trained.json").open())
    assert trained["signs"].keys() == json.load(
        (ROOT / "data/meta/curriculum_db.json").open())["signs"].keys()
    filled = [s for s, e in trained["signs"].items() if e["centroid"] is not None]
    assert filled, "no centroids were filled"
    assert all(len(trained["signs"][s]["centroid"]) == cfg.embed_dim for s in filled)
    # out-of-curriculum rivals get centroids too, so the decision policy can margin-check them
    assert "confusable_centroids" in trained
    assert not set(trained["confusable_centroids"]) & set(trained["signs"])


def test_parameter_budget():
    assert parameter_count(SignModel(Config())) < 2_000_000


def test_far_threshold_guarantee_and_overlap_estimate():
    from signisa.eval import far_threshold, overlap_estimate

    rng = np.random.default_rng(9)
    for n in (21, 41, 100, 101, 761):  # n = 1 mod 20 broke the quantile version
        scores = rng.normal(size=n)
        thr = far_threshold(scores, 0.05)
        assert (scores >= thr).mean() <= 0.05

    same = rng.normal(size=50)
    assert overlap_estimate(same, same.copy()) == 1.0
    genuine, confusable = rng.normal(10, 1, 50), rng.normal(0, 1, 50)
    assert overlap_estimate(genuine, confusable) == 0.0
    # inverted separation (impostors above genuine) counts as full overlap, not zero
    assert overlap_estimate(confusable, genuine) == 1.0
    # identical *distributions* at small n stay near 1 (histogram version read ~0.57)
    assert overlap_estimate(rng.normal(size=50), rng.normal(size=50)) > 0.8


def test_arcface_margin_soundness():
    from signisa.models import ArcFaceHead

    head = ArcFaceHead(8, 4, s=30.0, m=0.3)
    emb = torch.nn.functional.normalize(torch.randn(64, 8), dim=1)
    labels = torch.randint(0, 4, (64,))
    with torch.no_grad():
        plain = head(emb)
        margined = head(emb, labels)
    # margin can only lower the target logit, everywhere including cos near -1
    target_plain = plain.gather(1, labels[:, None])
    target_margined = margined.gather(1, labels[:, None])
    assert (target_margined <= target_plain + 1e-5).all()
    nontarget = ~torch.nn.functional.one_hot(labels, 4).bool()
    assert torch.allclose(plain[nontarget], margined[nontarget], atol=1e-5)
