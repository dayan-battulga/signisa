# CLAUDE.md — Signisa

Duolingo-style ASL learning app. The app always knows which sign the learner is attempting,
so the ML problem is **verification against a known target with phonological feedback** —
never open-vocabulary recognition. The exercise UI bounds each attempt (countdown → record
1–4 s → score); scoring runs on the completed window, so nothing needs to be causal.

## Ground-truth documents (read before designing anything)

- `docs/experiment-backlog.md` — current plan: phases, pass/kill criteria, cross-check flags. Start here.
- `docs/research/` — the four research reports behind every decision: task1 problem spec &
  metrics, task2 pose representation, task3a prior-art evidence, task3b system design.
  Cite these instead of re-deriving; don't re-litigate settled questions without new evidence.
- `memory.md` — rolling status + decision log. Read at session start, update at session end.

## Locked design decisions (change only with strong evidence; log any change in memory.md)

- **Input**: `(T=160, 65, 10)` tensor. 65 nodes = 21 left hand + 21 right hand + 7 body + 16 face
  (layout in `src/signisa/preprocess/landmarks.py`). 10 channels = xyz + velocity + bone + confidence.
- **Pipeline order** (implemented in `src/signisa/preprocess/pipeline.py`): mirror-if-left-dominant
  → short-gap fill (≤3 frames) → root-center (shoulder midpoint) → shoulder-width scale →
  canonical rotation → One Euro filter → zero-fill long gaps → resample to T=160 → derived
  channels. Raw duration + peak speed are kept as side features (speed can be lexically contrastive).
- **Isolated-sign verifier**: embedding model (ArcFace margin loss) + per-sign centroids.
  Accept = cosine threshold + margin-over-confusables + open-set rejection (OpenMax; fallback:
  energy-based). Not plain softmax accept/reject.
- **Confusables** come from ASL-LEX 2.0 phonological neighbors, stored per sign in curriculum_db.
- **Fingerspelling**: separate CTC model with FORCED alignment to the known target word —
  never open decoding.
- **Handedness**: canonical mirroring at inference (user toggle or wrist-velocity heuristic).
  No naive horizontal-flip augmentation.
- **Extractor**: MediaPipe Holistic tier first (matches Kaggle/PopSign training-data provenance;
  Kaggle asl-signs ships landmarks only). RTMW is a later accuracy tier, A/B'd on ASL Citizen.
- **Evaluation**: signer-independent splits ALWAYS. Report per-sign FAR/FRR/EER, never just accuracy.
- **Training venue**: Kaggle notebooks (free GPU, asl-signs pre-mounted). Local machine: data
  prep, tests, inference experiments.

## Repo layout

```
src/signisa/preprocess/   landmark selection, normalization chain, Kaggle loader (done, tested)
src/signisa/models/       ArcFace embedding model + phonological heads (Phase 1)
src/signisa/decision/     thresholds, margin-over-confusables, open-set rejection (Phase 1+)
scripts/                  coverage analysis, validation, curriculum_db builder
notebooks/                Kaggle training notebooks
data/meta/                small metadata only (class lists, CSVs) — tracked
data/                     everything else gitignored; datasets NEVER get committed
tests/                    pytest suite — must stay green
```

## Code standards

- Python 3.11, numpy/pandas; keep pure logic dependency-light and unit-testable.
- Naming: side-effecting functions verb-first (`build_curriculum_db`); pure value-returning
  functions read as noun phrases (`peak_speed_of`); booleans read as assertions (`is_left_dominant`).
- Small single-job functions; guard clauses over nested if/else; >3 params → group into a type.
- Rule of three before extracting a helper; no one-line single-caller helpers; no speculative
  generality (YAGNI); delete dead code — git is the history.
- Comment **why**, not what. Mark intentional seams with `# TODO:`.
- `pytest` green before every commit. Small commits; the message says why.

## Session protocol

1. Read `memory.md`, then the relevant phase in `docs/experiment-backlog.md`.
2. Do the work. Add or update tests; keep the suite green.
3. Update `memory.md`: what changed, decisions made, gotchas found, what's next.
4. Commit in small logical units; push to `origin main`.
