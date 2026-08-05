"""Signer-independent verification evaluation (backlog Phase 1).

Centroids come from TRAIN-participant embeddings only. For every val-participant
attempt of a curriculum sign X: genuine trial = cos(attempt, centroid_X);
impostor trials = cos(attempt, centroid_Y) for Y in confusables(X) plus
n_random_impostors random non-confusable signs. Per-sign metrics group trials
by TARGET centroid (the threshold that gates attempts against that sign).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .data import ShardDataset
from .models import SignModel


def held_out_participants(participant_ids, n: int, seed: int) -> list[int]:
    pids = sorted(set(int(p) for p in participant_ids))
    rng = np.random.default_rng(seed)
    return sorted(int(p) for p in rng.choice(pids, size=n, replace=False))


@torch.no_grad()
def compute_embeddings(model: SignModel, dataset: ShardDataset, device: str,
                       batch_size: int) -> np.ndarray:
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return np.concatenate([model.embedder(x.to(device)).cpu().numpy() for x, _ in loader])


def build_centroids(embeddings: np.ndarray, label_ids: np.ndarray) -> dict[int, np.ndarray]:
    """Unit-normalized per-class mean embeddings."""
    centroids = {}
    for label in np.unique(label_ids):
        mean = embeddings[label_ids == label].mean(axis=0)
        centroids[int(label)] = mean / np.linalg.norm(mean)
    return centroids


def eer_of(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """(EER, threshold) via the sweep point where FAR and FRR cross."""
    thresholds = np.unique(np.concatenate([genuine, impostor]))
    far = (impostor[None, :] >= thresholds[:, None]).mean(axis=1)
    frr = (genuine[None, :] < thresholds[:, None]).mean(axis=1)
    i = int(np.abs(far - frr).argmin())
    return float((far[i] + frr[i]) / 2.0), float(thresholds[i])


def overlap_coefficient(a: np.ndarray, b: np.ndarray, bins: int = 40) -> float:
    """Histogram overlap in [0, 1]: 1 = identical distributions."""
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if lo == hi:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges, density=True)
    pb, _ = np.histogram(b, bins=edges, density=True)
    return float(np.minimum(pa, pb).sum() * (edges[1] - edges[0]))


def run_evaluation(model: SignModel, tensors_dir, curriculum_db_path, labels_path,
                   cfg: Config, out_dir, device: str = "cpu",
                   val_participants: list[int] | None = None) -> dict:
    """Full Task-3 evaluation; writes metrics_report.md + curriculum_db_trained.json."""
    tensors_dir, out_dir = Path(tensors_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db = json.load(Path(curriculum_db_path).open())
    labels = json.load(Path(labels_path).open())
    label_of = {c["id"]: c["label"] for c in labels["classes"]}
    id_of = {c["label"]: c["id"] for c in labels["classes"]}

    index = pd.read_csv(tensors_dir / "index.csv")
    if val_participants is None:
        val_participants = held_out_participants(
            index.participant_id, cfg.n_val_participants, cfg.seed)
    train_pids = sorted(set(index.participant_id) - set(val_participants))

    train_ds = ShardDataset(tensors_dir, participants=train_pids)
    val_ds = ShardDataset(tensors_dir, participants=val_participants)
    train_emb = compute_embeddings(model, train_ds, device, cfg.batch_size)
    val_emb = compute_embeddings(model, val_ds, device, cfg.batch_size)
    centroids = build_centroids(train_emb, train_ds.index.canonical_label_id.to_numpy())

    # closed-set top-1 over all classes (sanity anchor)
    ids = sorted(centroids)
    matrix = np.stack([centroids[i] for i in ids])              # (C, D)
    predicted = np.array(ids)[(val_emb @ matrix.T).argmax(axis=1)]
    val_labels = val_ds.index.canonical_label_id.to_numpy()
    top1 = float((predicted == val_labels).mean())

    # verification trials over curriculum signs
    confusable_ids = {id_of[s]: [id_of[c] for c in e["confusables"]]
                      for s, e in db["signs"].items()}
    rng = np.random.default_rng(cfg.seed)
    trials = []  # (attempt_label, target_label, score, is_genuine)
    skipped = 0
    for emb, label in zip(val_emb, val_labels):
        label = int(label)
        if label not in confusable_ids or label not in centroids:
            skipped += label in confusable_ids  # curriculum sign missing a centroid
            continue
        trials.append((label, label, float(emb @ centroids[label]), True))
        hard = [c for c in confusable_ids[label] if c in centroids]
        pool = [i for i in ids if i != label and i not in hard]
        rand = rng.choice(pool, size=min(cfg.n_random_impostors, len(pool)), replace=False)
        for target in hard + [int(r) for r in rand]:
            trials.append((label, target, float(emb @ centroids[target]), False))
    tdf = pd.DataFrame(trials, columns=["attempt", "target", "score", "genuine"])

    genuine_all = tdf[tdf.genuine].score.to_numpy()
    impostor_all = tdf[~tdf.genuine].score.to_numpy()
    far5_global = float(np.quantile(impostor_all, 1.0 - cfg.far_target))
    tar_at_far = float((genuine_all >= far5_global).mean())

    per_sign = {}
    for sign, entry in db["signs"].items():
        target = id_of[sign]
        g = tdf[(tdf.target == target) & tdf.genuine].score.to_numpy()
        i = tdf[(tdf.target == target) & ~tdf.genuine].score.to_numpy()
        if len(g) == 0 or len(i) == 0:
            per_sign[sign] = {"n_genuine": len(g), "n_impostor": len(i),
                              "eer": None, "eer_threshold": None, "far5_threshold": None}
            continue
        eer, thr = eer_of(g, i)
        per_sign[sign] = {"n_genuine": len(g), "n_impostor": len(i), "eer": eer,
                          "eer_threshold": thr,
                          "far5_threshold": float(np.quantile(i, 1.0 - cfg.far_target))}

    clusters = []
    for members in db["clusters"]:
        member_ids = {id_of[m] for m in members}
        g = tdf[tdf.genuine & tdf.target.isin(member_ids)].score.to_numpy()
        c = tdf[~tdf.genuine & tdf.target.isin(member_ids)
                & tdf.attempt.isin(member_ids)].score.to_numpy()
        overlap = overlap_coefficient(g, c) if len(g) and len(c) else None
        clusters.append({"members": members, "n_genuine": len(g), "n_confusable": len(c),
                         "overlap": overlap,
                         "collapsed": overlap is not None and overlap > cfg.cluster_overlap_flag})

    metrics = {
        "val_participants": val_participants, "n_train": len(train_ds), "n_val": len(val_ds),
        "top1_closed_set": top1, "tar_at_far": tar_at_far, "far_target": cfg.far_target,
        "global_far_threshold": far5_global, "n_genuine": len(genuine_all),
        "n_impostor": len(impostor_all), "skipped_missing_centroid": skipped,
        "per_sign": per_sign, "clusters": clusters,
    }
    _write_report(metrics, out_dir / "metrics_report.md", cfg)
    _write_trained_db(db, centroids, per_sign, id_of, out_dir / "curriculum_db_trained.json")
    return metrics


def _write_report(m: dict, path: Path, cfg: Config) -> None:
    eers = [v["eer"] for v in m["per_sign"].values() if v["eer"] is not None]
    lines = [
        "# Verification metrics (signer-independent)\n",
        f"- Val participants: {m['val_participants']} ({m['n_val']} sequences; "
        f"{m['n_train']} train sequences)",
        f"- **TAR@FAR={cfg.far_target:.0%}: {m['tar_at_far']:.1%}** "
        f"(success criterion: >90%) — {m['n_genuine']} genuine / {m['n_impostor']} impostor trials",
        f"- Closed-set top-1 over all {cfg.n_classes} classes: {m['top1_closed_set']:.1%}",
        f"- Mean per-sign EER: {np.mean(eers):.1%} (n={len(eers)})" if eers else "- No per-sign EERs",
        f"- Skipped curriculum attempts missing a centroid: {m['skipped_missing_centroid']}\n",
        "## Cluster collapse check (flag: overlap > "
        f"{cfg.cluster_overlap_flag:.0%})\n",
        "| cluster | genuine | confusable | overlap | collapsed |",
        "|---|---|---|---|---|",
    ]
    for c in m["clusters"]:
        ov = f"{c['overlap']:.1%}" if c["overlap"] is not None else "-"
        lines.append(f"| {' / '.join(c['members'])} | {c['n_genuine']} | {c['n_confusable']} "
                     f"| {ov} | {'**YES**' if c['collapsed'] else 'no'} |")
    lines += ["\n## Per-sign\n",
              "| sign | genuine | impostor | EER | eer_thr | far5_thr |", "|---|---|---|---|---|---|"]
    for sign, v in sorted(m["per_sign"].items()):
        if v["eer"] is None:
            lines.append(f"| {sign} | {v['n_genuine']} | {v['n_impostor']} | - | - | - |")
        else:
            lines.append(f"| {sign} | {v['n_genuine']} | {v['n_impostor']} | {v['eer']:.1%} "
                         f"| {v['eer_threshold']:.3f} | {v['far5_threshold']:.3f} |")
    path.write_text("\n".join(lines) + "\n")


def _write_trained_db(db: dict, centroids: dict, per_sign: dict, id_of: dict,
                      path: Path) -> None:
    out = json.loads(json.dumps(db))  # deep copy
    for sign, entry in out["signs"].items():
        label = id_of[sign]
        if label in centroids:
            entry["centroid"] = [round(float(v), 6) for v in centroids[label]]
        stats = per_sign.get(sign, {})
        entry["eer_threshold"] = stats.get("eer_threshold")
        entry["low_far_threshold"] = stats.get("far5_threshold")
    path.write_text(json.dumps(out) + "\n")
