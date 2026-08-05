# Project memory

Created 2026-08-05. CLAUDE.md + docs/ arrived later the same day (they missed
the original push); the Phase 0 work below predates them but is consistent.

## Status

Session 2 (2026-08-05, later):
- Dominance detection done (`src/signisa/preprocess/dominance.py`): per-hand
  score = presence-fraction x mean wrist speed, 1.2x hysteresis -> left/right/
  ambiguous; per-participant majority vote (`majority_dominance`, ties ->
  right/no-mirror); per-sequence fallback reserved for unseen participants.
  `load_wrists` in kaggle.py reads only the 2 wrist rows/frame for a fast
  dominance pass. On the 30 samples: 5/15 participants left-dominant -> 9/30
  sequences mirrored (small-sample votes over 1-3 seqs; the full-data vote
  happens in build_training_tensors on Kaggle).
- Label collisions derived systematically (same resolved ASL-LEX entry):
  exactly 4 groups — awake/wake, cat/kitty, dog/puppy, nap/sleep -> **246
  canonical training classes** in `data/meta/training_labels.json`
  (classes + sign_to_class for all 250). Curriculum verified all-canonical.
- `data/meta/curriculum_db.json` built (backlog 0.4): 50 signs with phonology
  (handshape, major/minor location, movement, sign_type/one_handed), cluster
  members, canonicalized strong confusables within the 250, null slots for
  centroid/eer_threshold/low_far_threshold/weibull_params.
- `scripts/build_training_tensors.py`: two passes (wrist dominance vote, then
  mirror+preprocess) -> float16 (160,65,10) in ~500-seq .npz shards +
  index.csv (sequence_id, participant_id, canonical_label_id, duration_s,
  peak_speed, mirrored). Row i -> shard i//500. Local smoke: 72 seq/s overall
  => full 94,477 seqs ~22 min single-process; Kaggle full run pending.

Session 1:

- Phase 0.3 done: pipeline validated on 30 real Kaggle asl-signs parquets
  (10 signs x 3 signers). All outputs (160, 65, 10), NaN-free.
  `scripts/validate_pipeline.py` reruns the check.
- The 16 face-mesh indices in `src/signisa/preprocess/landmarks.py` were verified
  against real data: brows above eyes above mouth (MediaPipe y down), all 6
  left/right mirror pairs straddle the nose. **No index changes needed.**
- No pipeline bugs surfaced on real data; no fixes, so no new regression tests
  for Task 1. `tests/test_coverage.py` guards the coverage analysis instead.
- Phase 0.2 done: `scripts/coverage_analysis.py` ->
  `data/meta/coverage_report.md` + `data/meta/curriculum_candidates.csv`.

## Data locations

- `~/Desktop/dataset/asl-signs.zip` — full 37 GB Kaggle dump. Only metadata +
  30 sample parquets were extracted (to `~/Desktop/dataset/asl-signs/`); extract
  more with `unzip asl-signs.zip 'train_landmark_files/<pid>/<seq>.parquet'`.
- `~/Desktop/dataset/asl-lex/` — ASL-LEX 2.0 OSF export ("Data Files" only).
  The database is `Data Files/signdata.csv`, copied to
  `data/meta/asllex_signdata.csv`.
- `data/samples/*.parquet` — 30 sample sequences, gitignored, filename = sequence_id.

## Decisions

Locked (from CLAUDE.md — change only with strong evidence, log changes here):
- Verifier = ArcFace embedding + per-sign centroids; accept = cosine threshold
  + margin-over-confusables + open-set rejection. Never plain softmax.
- Fingerspelling = separate CTC model with FORCED alignment to the known word.
- Handedness = canonical mirroring at inference; no horizontal-flip augmentation.
- Extractor = MediaPipe Holistic tier first (matches Kaggle landmarks-only
  provenance); RTMW is a later accuracy tier A/B'd on ASL Citizen.
- Evaluation = signer-independent splits always; per-sign FAR/FRR/EER, never
  just accuracy. Training on Kaggle notebooks; local = prep/tests/inference.

- ASL-LEX gloss matching: 24-entry alias table in `coverage_analysis.py`
  (dad=FATHER, potty=TOILET, haveto=MUST, ...). 233/250 matched; the 17
  unmatched are listed in coverage_report.md. wake/kitty/puppy/nap alias to
  awake/cat/dog/sleep — same-sign label collisions, so those pairs show up as
  maximal strong confusables (visually identical). garbage deliberately
  unmatched: ASL-LEX 'trash' is the BASKET sign, 'throw_away' unconfirmed.
  store deliberately unmatched: would resolve to shop_1 = SHOPPING.
- Confusable definition: "any" tie = >=2 shared of {Handshape.2.0,
  MajorLocation.2.0, Movement.2.0}; "strong" (minimal-pair grade) = shared
  handshape AND major location. Curriculum clusters use strong ties.
- v1 curriculum: 50 signs hardcoded as `CURRICULUM_V1` in coverage_analysis.py,
  8 minimal-pair clusters incl. the 7-sign 5-hand-at-head family cluster
  (mom/dad/grandma/grandpa/mad/sad/sleep).
- 218/250 signs also in WLASL-2000 (alias-aware match).
- validate_pipeline's percent-missing uses `1 - mean(confidence)`, not
  `conf == 0`: resampling smears gap boundaries into fractional confidence, so
  exact-zero counting undercounts (caught by adversarial review on real data).
- eye→EYES / tooth→TEETH borrow plural Movement coding (Curved sweep); the
  handshape+location fields driving the strong-confusable graph are unaffected.

## Gotchas

- `asllex_signdata.csv` is **latin-1 encoded**, not UTF-8. 2723 rows, 4 dup
  EntryIDs (deduped on load). Variant glosses carry `_1`/`_2` suffixes.
- ASL-LEX `NeigborPairs.csv` (7.4M rows, in the OSF export) exists but we
  compute neighbors from signdata.csv features directly instead.
- Kaggle coverage is near-uniform (299–415 examples, 19–21 signers per sign) —
  coverage does not discriminate for curriculum selection.
- Many real sequences have one hand 100% absent (single-handed signs +
  left-dominant signers). Dominance detection now exists (session 2); note
  ~33% of sample participants vote left-dominant — likely selfie-mirroring in
  the Kaggle capture rather than true handedness. Irrelevant to us: canonical
  mirroring only needs data-space consistency, not true handedness.
- Kaggle parquets carry all 543 rows per frame with all-NaN coords when a
  detector missed — `load_wrists`' filtered read relies on this.
- Real sequences can be as short as 6 frames (~0.2 s) and as long as 223.

## Missing / absent from expected inputs

- CLAUDE.md, docs/experiment-backlog.md, memory.md: not in the repo (expected
  by the Phase 0 brief). This file now seeds memory.md.
- Everything expected in the datasets was present (sign map, train.csv,
  parquets, ASL-LEX database CSV). Nothing missing.
