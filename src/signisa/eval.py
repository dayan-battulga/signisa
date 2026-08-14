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
    """(EER, threshold) via the sweep point where FAR and FRR cross.

    O(thresholds x trials) memory — per-sign scale only; never call on pooled arrays.
    """
    thresholds = np.unique(np.concatenate([genuine, impostor]))
    far = (impostor[None, :] >= thresholds[:, None]).mean(axis=1)
    frr = (genuine[None, :] < thresholds[:, None]).mean(axis=1)
    i = int(np.abs(far - frr).argmin())
    return float((far[i] + frr[i]) / 2.0), float(thresholds[i])


def far_threshold(impostor: np.ndarray, far_target: float) -> float:
    """Smallest threshold with FAR <= far_target under the >= acceptance rule.

    Order-statistic based (ties included) — unlike an interpolated quantile,
    the guarantee holds at any n.
    """
    s = np.sort(impostor)
    k = int(np.floor(far_target * len(s)))  # max allowed false accepts
    bound = s[-1] if k == 0 else s[-(k + 1)]
    return float(np.nextafter(bound, np.inf))


def overlap_estimate(genuine: np.ndarray, confusable: np.ndarray) -> float:
    """2 * (1 - AUC), in [0, 1]: 1 = indistinguishable distributions, 0 = separated.

    Rank-based, so unbiased at small sample sizes (a 40-bin histogram overlap
    measures ~0.57 for identical 50-sample distributions and under-flags collapse).
    """
    greater = (genuine[:, None] > confusable[None, :]).mean()
    ties = (genuine[:, None] == confusable[None, :]).mean()
    auc = greater + 0.5 * ties
    return float(np.clip(2.0 * (1.0 - auc), 0.0, 1.0))


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
    trials = []
    skipped = 0
    val_seqs = val_ds.index.sequence_id.to_numpy()
    val_pids_col = val_ds.index.participant_id.to_numpy()
    for emb, label, seq, pid in zip(val_emb, val_labels, val_seqs, val_pids_col):
        label = int(label)
        if label not in confusable_ids or label not in centroids:
            skipped += label in confusable_ids  # curriculum sign missing a centroid
            continue
        trials.append((int(seq), int(pid), label, label, float(emb @ centroids[label]), True))
        hard = [c for c in confusable_ids[label] if c in centroids]
        pool = [i for i in ids if i != label and i not in hard]
        rand = rng.choice(pool, size=min(cfg.n_random_impostors, len(pool)), replace=False)
        for target in hard + [int(r) for r in rand]:
            trials.append((int(seq), int(pid), label, target,
                           float(emb @ centroids[target]), False))
    tdf = pd.DataFrame(trials, columns=["sequence_id", "participant", "attempt",
                                        "target", "score", "genuine"])

    genuine_all = tdf[tdf.genuine].score.to_numpy()
    impostor_all = tdf[~tdf.genuine].score.to_numpy()
    far5_global = far_threshold(impostor_all, cfg.far_target)
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
                          "far5_threshold": far_threshold(i, cfg.far_target)}

    # per-participant breakdown at the GLOBAL threshold — a bad or wrongly-mirrored
    # val signer shows up as an outlier row here
    per_participant = {}
    for pid in val_participants:
        sub = tdf[tdf.participant == pid]
        g = sub[sub.genuine].score.to_numpy()
        sign_eers = [eer_of(sg, si)[0] for t in sub.target.unique()
                     if len(sg := sub[(sub.target == t) & sub.genuine].score.to_numpy())
                     and len(si := sub[(sub.target == t) & ~sub.genuine].score.to_numpy())]
        vrows = index[index.participant_id == pid]
        per_participant[int(pid)] = {
            "n_genuine": len(g),
            "tar_at_far": float((g >= far5_global).mean()) if len(g) else None,
            "mean_eer": float(np.mean(sign_eers)) if sign_eers else None,
            "genuine_median": float(np.median(g)) if len(g) else None,
            "genuine_iqr": float(np.subtract(*np.percentile(g, [75, 25]))) if len(g) else None,
            "mirrored_rate": float(vrows.mirrored.mean()),
        }

    clusters = []
    for members in db["clusters"]:
        member_ids = {id_of[m] for m in members}
        g = tdf[tdf.genuine & tdf.target.isin(member_ids)].score.to_numpy()
        c = tdf[~tdf.genuine & tdf.target.isin(member_ids)
                & tdf.attempt.isin(member_ids)].score.to_numpy()
        overlap = overlap_estimate(g, c) if len(g) and len(c) else None
        clusters.append({"members": members, "n_genuine": len(g), "n_confusable": len(c),
                         "overlap": overlap,
                         "collapsed": overlap is not None and overlap > cfg.cluster_overlap_flag})

    metrics = {
        "val_participants": val_participants, "n_train": len(train_ds), "n_val": len(val_ds),
        "top1_closed_set": top1, "tar_at_far": tar_at_far, "far_target": cfg.far_target,
        "global_far_threshold": far5_global, "n_genuine": len(genuine_all),
        "n_impostor": len(impostor_all), "skipped_missing_centroid": skipped,
        "per_sign": per_sign, "clusters": clusters,
        "per_participant": per_participant, "trials": tdf,
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
        "## Per-participant (val, at the global threshold)\n",
        "| participant | genuine | TAR@FAR | mean EER | genuine median | IQR | mirrored |",
        "|---|---|---|---|---|---|---|",
        *[f"| {pid} | {v['n_genuine']} "
          f"| {v['tar_at_far']:.1%} | {v['mean_eer']:.1%} "
          f"| {v['genuine_median']:.3f} | {v['genuine_iqr']:.3f} | {v['mirrored_rate']:.0%} |"
          if v["tar_at_far"] is not None and v["mean_eer"] is not None
          else f"| {pid} | {v['n_genuine']} | - | - | - | - | {v['mirrored_rate']:.0%} |"
          for pid, v in m["per_participant"].items()],
        "\n## Cluster collapse check (flag: overlap > "
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
