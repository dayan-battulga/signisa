"""CPU smoke tests for the training stack: slim shards -> Dataset reconstruction,
augmentations, overfit sanity (both losses), and the end-to-end eval path."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import Subset

from signisa import SHARD_SCHEMA
from signisa.config import AugmentConfig, Config
from signisa.data import ShardDataset, augmented, pad_collate, side_features
from signisa.eval import run_evaluation
from signisa.models import SignModel, parameter_count
from signisa.train import make_loader, top1_accuracy, train_model

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


def test_ragged_shards_and_dataset_reconstructs(tensors_dir):
    from signisa.preprocess.kaggle import load_holistic
    from signisa.preprocess.pipeline import preprocess

    shard = np.load(tensors_dir / "shard_0000.npz")
    assert int(shard["schema_version"]) == SHARD_SCHEMA
    assert shard["frames"].shape[1:] == (65, 4) and shard["frames"].dtype == np.float16
    assert shard["lengths"].sum() == len(shard["frames"])
    assert len(set(shard["lengths"].tolist())) > 1, "samples should not all be the same length"

    ds = ShardDataset(tensors_dir)
    tensor, side, label = ds[0]
    row = ds.index.iloc[0]
    full = preprocess(load_holistic(SAMPLES / f"{row.sequence_id}.parquet"),
                      left_dominant=bool(row.mirrored)).tensor
    assert tensor.shape == full.shape and tensor.dtype == torch.float32
    assert isinstance(label, int)
    np.testing.assert_allclose(tensor.numpy(), full, atol=1e-2)  # float16 storage rounding
    np.testing.assert_allclose(side.numpy(), side_features(row.duration_s, row.peak_speed))

    # canonical-space flip == mirroring the raw sequence before preprocess — BIT-exact
    # on the float16 path (float16 rounding is sign-symmetric, so it commutes with the flip)
    from signisa.data import mirrored_stored
    flipped = mirrored_stored(ds.stored(0))
    other = preprocess(load_holistic(SAMPLES / f"{row.sequence_id}.parquet"),
                       left_dominant=not bool(row.mirrored)).tensor[..., [0, 1, 2, 9]]
    assert np.array_equal(flipped, other.astype(np.float16))


def stored_sequence(t: int = 100, n: int = 65) -> np.ndarray:
    rng = np.random.default_rng(3)
    stored = np.concatenate([rng.normal(size=(t, n, 3)), np.ones((t, n, 1))], axis=2)
    stored[:, 5] = 0.0  # a node the tracker never found
    return stored.astype(np.float32)


def test_augmentation_identity_and_budgets():
    rng = np.random.default_rng(3)
    stored = stored_sequence()
    side = side_features(3.0, 4.0)

    off = AugmentConfig(mask_total_frac=0.0, rotation_deg=0.0, scale=0.0, translation=0.0,
                        node_dropout_p=0.0, noise_sigma=0.0, crop_min_frac=1.0,
                        speed_min=1.0, speed_max=1.0)
    quiet, quiet_side = augmented(stored.copy(), side, off, rng)
    np.testing.assert_allclose(quiet, stored, atol=1e-6)
    np.testing.assert_allclose(quiet_side, side, rtol=1e-6)

    for _ in range(200):
        out, out_side = augmented(stored.copy(), side, AugmentConfig(node_dropout_p=0.0), rng)
        t = out.shape[0]
        assert 0.8 * 0.8 * 100 - 2 <= t <= 100 / 0.8 + 1        # speed-scale x crop bounds
        assert (out[..., 3] == 0.0).all(axis=1).sum() <= int(0.4 * t)  # temporal-mask budget
        assert (out[out[..., 3] == 0.0][..., :3] == 0.0).all()  # masked frames fully zeroed
        # duration follows the time warp; peak speed stays positive and finite
        assert 0.6 < out_side[0] < 4.0 and np.isfinite(out_side[1]) and out_side[1] > 0

    # augmentation alone never flips: +-5 deg can't invert a strongly right-biased cloud
    biased = stored.copy()
    biased[..., :3] = np.abs(biased[..., :3])
    flipless, _ = augmented(biased, side, AugmentConfig(), rng)
    assert flipless[..., 0].sum() > 0


def test_flip_augmentation_mirrors_and_is_off_when_disabled(tensors_dir):
    from signisa.data import mirrored_stored

    plain = ShardDataset(tensors_dir)
    quiet = AugmentConfig(mask_total_frac=0.0, rotation_deg=0.0, scale=0.0, translation=0.0,
                          node_dropout_p=0.0, noise_sigma=0.0, crop_min_frac=1.0,
                          speed_min=1.0, speed_max=1.0, flip_p=1.0)
    always = ShardDataset(tensors_dir, augment=True, aug_config=quiet)
    never = ShardDataset(tensors_dir, augment=True, aug_config=replace(quiet, flip_p=0.0))
    expected = mirrored_stored(plain.stored(0).astype(np.float32))
    np.testing.assert_allclose(always[0][0].numpy()[..., :3], expected[..., :3], atol=1e-5)
    np.testing.assert_allclose(never[0][0].numpy(), plain[0][0].numpy(), atol=1e-5)


def test_pad_collate_masks_real_frames(tensors_dir):
    ds = ShardDataset(tensors_dir)
    batch = [ds[i] for i in range(4)]
    x, mask, side, labels = pad_collate(batch)
    lengths = [sample[0].shape[0] for sample in batch]
    assert x.shape == (4, max(lengths), 65, 10)
    assert mask.shape == (4, max(lengths)) and side.shape == (4, 2) and labels.shape == (4,)
    for i, n in enumerate(lengths):
        assert mask[i].sum() == n
        assert torch.equal(x[i, :n], batch[i][0])
        assert (x[i, n:] == 0).all()

    # padding must not move the embedding: pooling and attention are mask-aware
    model = SignModel(small_config("ce")).eval()
    with torch.no_grad():
        batched = model.embedder(x, mask, side)
        alone = torch.cat([model.embedder(*pad_collate([b])[:3]) for b in batch])
    torch.testing.assert_close(batched, alone, atol=1e-4, rtol=1e-4)


def test_length_bucket_sampler_covers_every_index(tensors_dir):
    ds = ShardDataset(tensors_dir)
    cfg = small_config("ce", batch_size=4)
    for shuffle in (True, False):
        batches = list(make_loader(ds, cfg, shuffle).batch_sampler)
        assert sorted(i for b in batches for i in b) == list(range(len(ds)))
        assert all(len(b) <= 4 for b in batches)
    ordered = list(make_loader(ds, cfg, False).batch_sampler)
    assert np.concatenate(ordered).tolist() == sorted(
        range(len(ds)), key=lambda i: (ds.lengths[i], i))  # sorted by length, stable


@pytest.mark.parametrize("loss", ["ce", "arcface"])
def test_overfit_ten_sequences(tensors_dir, loss):
    ds = Subset(ShardDataset(tensors_dir), range(10))
    cfg = small_config(loss, epochs=200)
    model, history = train_model(cfg, ds, ds, max_steps=200)
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert top1_accuracy(model, make_loader(ds, cfg, shuffle=False), "cpu") == 1.0


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

    val_strs = {str(p) for p in val_pids}  # eval normalizes signer ids to str (ac_ merge)
    assert set(metrics["per_participant"]) == val_strs
    for v in metrics["per_participant"].values():
        assert v["mirrored_rate"] in (0.0, 1.0)  # vote is per participant, so uniform
    trials = metrics["trials"]
    assert list(trials.columns) == ["sequence_id", "participant", "attempt",
                                    "target", "score", "genuine", "domain"]
    assert set(trials.participant) <= val_strs
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


@pytest.fixture(scope="session")
def merged_dir(tmp_path_factory):
    """Kaggle samples + 2 synthetic ASL Citizen signers x 2 clips, one shard set."""
    from test_pipeline import synthetic_holistic

    tmp = tmp_path_factory.mktemp("citizen_merge")
    npz_dir = tmp / "extracted"
    npz_dir.mkdir()
    labels = json.load((ROOT / "data/meta/training_labels.json").open())
    mom, cat = labels["sign_to_class"]["mom"], labels["sign_to_class"]["cat"]
    rows = []
    for pid, label, name in [(101, mom, "mom"), (101, cat, "cat"),
                             (102, mom, "mom"), (102, cat, "cat")]:
        video = f"{pid}-{name.upper()}-{len(rows)}.mp4"
        np.savez_compressed(npz_dir / f"{Path(video).stem}.npz",
                            holistic=synthetic_holistic(24).astype(np.float16),
                            fps=24.0, n_frames=24, n_detected_frames=24,
                            mediapipe_version="stub")
        rows.append({"videofile": video, "gloss": name.upper(), "asllexcode": name,
                     "participant_id": f"ac_{pid}", "split": "train",
                     "canonical_label_id": label, "canonical_label": name})
    rows.append({"videofile": "999-GONE.mp4", "gloss": "MOM", "asllexcode": "mother",
                 "participant_id": "ac_999", "split": "train",
                 "canonical_label_id": mom, "canonical_label": "mom"})  # npz missing
    mapping = tmp / "mapping.csv"
    pd.DataFrame(rows).to_csv(mapping, index=False)

    out = tmp / "tensors"
    subprocess.run(
        [sys.executable, "scripts/build_training_tensors.py",
         "--landmarks-dir", str(SAMPLES), "--out-dir", str(out),
         "--citizen-npz-dir", str(npz_dir), "--citizen-mapping", str(mapping)],
        check=True, cwd=ROOT)
    return out


def test_citizen_build_rejects_stale_mapping_and_empty_npz_dir(merged_dir, tmp_path):
    src = merged_dir.parent
    good_mapping = pd.read_csv(src / "mapping.csv")

    stale = good_mapping.copy()
    stale["canonical_label_id"] += 1  # ids from an older labels file
    stale.to_csv(tmp_path / "stale.csv", index=False)
    for mapping, npz_dir, msg in [
            (tmp_path / "stale.csv", src / "extracted", "disagree with --labels"),
            (src / "mapping.csv", tmp_path / "empty", "is not a directory")]:
        proc = subprocess.run(
            [sys.executable, "scripts/build_training_tensors.py",
             "--landmarks-dir", str(SAMPLES), "--out-dir", str(tmp_path / "out"),
             "--citizen-npz-dir", str(npz_dir), "--citizen-mapping", str(mapping)],
            capture_output=True, text=True, cwd=ROOT)
        assert proc.returncode != 0 and msg in proc.stderr


def test_merged_index_domains_and_filtering(merged_dir):
    index = pd.read_csv(merged_dir / "index.csv")
    assert index.domain.value_counts().to_dict() == {"popsign": 30, "asl_citizen": 4}
    citizen = index[index.domain == "asl_citizen"]
    assert set(citizen.participant_id) == {"ac_101", "ac_102"}  # missing npz row dropped
    assert citizen.groupby("participant_id").mirrored.nunique().eq(1).all()  # per-signer vote

    ds = ShardDataset(merged_dir)
    assert len(ds) == 34
    tensor, side, label = ds[len(ds) - 1]  # a citizen row: same ragged contract
    assert tensor.shape == (ds.lengths[-1], 65, 10) and side.shape == (2,)

    for pids, expected in [(["ac_101"], 2), ({"ac_101", "ac_102"}, 4),
                           ([index.participant_id.iloc[0]], None)]:
        subset = ShardDataset(merged_dir, participants=pids)
        assert len(subset) == (expected or len(subset)) and len(subset) > 0
    # int and str spellings of a PopSign id select the same rows
    pop = index[index.domain == "popsign"].participant_id.iloc[0]
    assert len(ShardDataset(merged_dir, participants=[int(pop)])) == \
        len(ShardDataset(merged_dir, participants=[str(pop)]))


def test_merged_eval_reports_per_domain(merged_dir, tmp_path):
    from signisa.eval import default_val_participants, held_out_participants

    cfg = small_config("ce", n_val_participants=2, n_val_citizen_signers=1)
    model = SignModel(cfg).eval()  # untrained: the eval plumbing is what's under test
    metrics = run_evaluation(model, merged_dir, ROOT / "data/meta/curriculum_db.json",
                             ROOT / "data/meta/training_labels.json", cfg, tmp_path)

    assert set(metrics["per_domain"]) == {"popsign", "asl_citizen"}
    for v in metrics["per_domain"].values():
        assert v["n_genuine"] > 0 and v["n_signers"] > 0
    report = (tmp_path / "metrics_report.md").read_text()
    assert "## Per-domain" in report and "| asl_citizen |" in report

    # the PopSign selection is EXACTLY the historic one — numbers stay comparable
    index = pd.read_csv(merged_dir / "index.csv")
    pop_pids = index[index.domain == "popsign"].participant_id
    expected_pop = held_out_participants(pop_pids, 2, cfg.seed)
    chosen = default_val_participants(index, cfg)
    assert chosen[:2] == expected_pop
    assert sum(str(p).startswith("ac_") for p in chosen) == 1  # 1 of 2 citizen signers held out
    assert set(metrics["val_participants"]) == {str(p) for p in chosen}


def test_held_out_participants_matches_historic_selection():
    from signisa.eval import held_out_participants

    def historic(participant_ids, n, seed):  # the pre-domain implementation, verbatim
        pids = sorted(set(int(p) for p in participant_ids))
        rng = np.random.default_rng(seed)
        return sorted(int(p) for p in rng.choice(pids, size=n, replace=False))

    rng = np.random.default_rng(0)
    for _ in range(20):
        pids = rng.choice(30000, size=rng.integers(5, 25), replace=False).tolist()
        assert held_out_participants(pids, 4, 42) == historic(pids, 4, 42)
        assert held_out_participants([str(p) for p in pids], 4, 42) == historic(pids, 4, 42)


def test_stale_shard_schema_is_rejected(tensors_dir, tmp_path):
    import shutil

    shutil.copy(tensors_dir / "index.csv", tmp_path / "index.csv")
    old = np.load(tensors_dir / "shard_0000.npz")
    np.savez_compressed(tmp_path / "shard_0000.npz", frames=old["frames"],
                        lengths=old["lengths"], sequence_id=old["sequence_id"],
                        landmark_version=old["landmark_version"])  # no schema_version = v1
    with pytest.raises(AssertionError, match="shard schema 1"):
        ShardDataset(tmp_path)


def test_embedding_of_matches_the_batched_path(tensors_dir):
    """The inference CLIs build mask/side by hand — they must agree with pad_collate."""
    from signisa.models import embedding_of
    from signisa.preprocess.kaggle import load_holistic
    from signisa.preprocess.pipeline import preprocess

    ds = ShardDataset(tensors_dir)
    row = ds.index.iloc[0]
    model = SignModel(small_config("ce")).eval()
    pre = preprocess(load_holistic(SAMPLES / f"{row.sequence_id}.parquet"),
                     left_dominant=bool(row.mirrored))
    with torch.no_grad():  # index 1 is a different length, so entry 0 is genuinely padded
        batched = model.embedder(*pad_collate([ds[0], ds[1]])[:3])[0].numpy()
    np.testing.assert_allclose(embedding_of(model, pre), batched, atol=1e-2)


def test_parameter_budget():
    assert parameter_count(SignModel(Config())) < 2_000_000


@pytest.fixture(scope="session")
def tensors_dir_v2(tmp_path_factory):
    out = tmp_path_factory.mktemp("tensors_v2")
    subprocess.run(
        [sys.executable, "scripts/build_training_tensors.py", "--landmarks-dir", str(SAMPLES),
         "--out-dir", str(out), "--landmark-version", "v2", "--limit", "10"],
        check=True, cwd=ROOT)
    return out


def test_v2_tensors_dataset_and_model(tensors_dir_v2):
    shard = np.load(tensors_dir_v2 / "shard_0000.npz")
    assert shard["frames"].shape[1:] == (99, 4)
    assert str(shard["landmark_version"]) == "v2"

    ds = ShardDataset(tensors_dir_v2)
    assert ds.landmark_version == "v2"
    tensor, side, _ = ds[0]
    assert tensor.shape == (ds.lengths[0], 99, 10) and side.shape == (2,)

    cfg = small_config("ce", landmark_version="v2")
    _, logits = SignModel(cfg).eval()(*pad_collate([ds[0]])[:3])
    assert logits.shape == (1, 246)

    # v1 model must never silently consume v2 tensors
    with pytest.raises(AssertionError, match="expects v1"):
        train_model(small_config("ce", epochs=1), ds, ds)


def test_v2_lip_mirror_pairs_on_real_frames(tensors_dir_v2):
    from signisa.preprocess.landmarks import LANDMARK_SETS
    from signisa.preprocess.kaggle import load_holistic
    from signisa.preprocess.pipeline import mirrored, select_nodes

    v2 = LANDMARK_SETS["v2"]
    seq = select_nodes(load_holistic(SAMPLES / "1019555958.parquet"), "v2")
    lips = range(59, 99)
    full = seq[~np.isnan(seq[:, list(lips)]).any(axis=(1, 2))
               & ~np.isnan(seq[:, 42]).any(axis=1)]
    assert len(full) > 0
    nose_x = full[:, 42, 0]
    for i in lips:
        j = int(v2.mirror_perm[i])
        if j == i:  # midline lip point: stays near the nose line
            assert abs((full[:, i, 0] - nose_x).mean()) < 0.02
        else:       # paired lip points straddle the nose x
            assert (full[:, i, 0] - nose_x).mean() * (full[:, j, 0] - nose_x).mean() < 0
    # mirrored lip x-coordinates reflect: node i picks up the partner's negated x
    m = mirrored(full, "v2")
    np.testing.assert_allclose(m[:, list(lips), 0],
                               -full[:, v2.mirror_perm[list(lips)], 0], atol=1e-12)


def test_checkpoint_round_trip(tmp_path):
    from signisa.models import load_checkpoint, save_checkpoint

    model = SignModel(Config(loss="arcface", landmark_version="v2"))  # default dims:
    save_checkpoint(model, tmp_path / "m.pt")                         # loaders rebuild those
    loaded = load_checkpoint(tmp_path / "m.pt")
    assert loaded.cfg.loss == "arcface" and loaded.cfg.landmark_version == "v2"

    # legacy raw v1 state dicts still load (loss inferred from head.bias)
    legacy = SignModel(Config(loss="ce"))
    torch.save(legacy.state_dict(), tmp_path / "legacy.pt")
    old = load_checkpoint(tmp_path / "legacy.pt")
    assert old.cfg.loss == "ce" and old.cfg.landmark_version == "v1"


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
