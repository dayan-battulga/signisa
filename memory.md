# Project memory

Created 2026-08-05. CLAUDE.md + docs/ arrived later the same day (they missed
the original push); the Phase 0 work below predates them but is consistent.

## Status

Session 12 (2026-08-20, time-boxed chain-resumable extraction):
- **A shard hit Kaggle's 12 h cap and was SIGKILLed — killed commits publish NOTHING.**
  Estimation-based sharding is out; extraction is now time-boxed + chain-resumable.
- **--time-budget-h** (notebook TIME_BUDGET_H=10.5): checked after every completed
  clip; on expiry the queued futures are cancelled, the <= workers in-flight clips
  finish (atomic writes land and the next run skips them — verified in the chain
  smoke), done/remaining printed, exit 0 so the version SAVES.
- **--done-dir (repeatable)**: prior runs' outputs count as completed (npz AND their
  failures.csv). Chained Kaggle flow: publish the partial version, attach it plus all
  earlier links, re-run same SHARD_INDEX — only the remainder extracts. Smoke-tested
  three-link chain locally (budget-stop -> resume -> nothing pending).
  shard_manifest.json now carries n_remaining + chained_on; kaggle_prep warns when an
  attached shard chain still has clips remaining (attach its latest version). Prep's
  cross-dir stem-uniqueness assert already handles chained partials (later links
  never re-extract earlier links' clips).
- **Speed: frames downscaled to long side <= 640 (MAX_SIDE) before MediaPipe.**
  MEASURED trade-off on a 720p smoke clip: ~1.25x faster but 38% fewer detected
  frames (small far-away hands lose crop pixels). ASL Citizen's centered webcam
  framing should be far more forgiving, but A/B n_detected_frames on ~50 real clips
  (--max-side 0 vs 640) at the start of shard 0 before committing all 83k.
  **model_complexity does NOT exist in the Tasks API** (verified: HolisticLandmarker
  options are confidence thresholds only; the .task bundle bakes its models in) —
  input resolution is the only available speed knob.
- Early projection line kept, now informational: prints measured clips/s + projected
  shard hours + how many chained runs the budget implies.

Session 11 (2026-08-17, Kaggle extraction wrapper + multi-input prep):
- **notebooks/kaggle_extract.ipynb**: sharded ASL Citizen extraction on Kaggle CPU
  against the kaggle.com/datasets/abd0kamel/asl-citizen mirror. CONFIG = SHARD_INDEX /
  NUM_SHARDS (4) / WORKERS (4); run once per shard in separate sessions, publish each
  output, attach all to kaggle_prep. First cell is an integrity gate: globs the
  attached inputs, hard-fails under 80k videos or without all three split CSVs
  (partial mirror -> use the Microsoft download; never silently extract a subset).
  Split CSVs are copied into every shard output so prep needs only extraction
  outputs + asl-signs. Ends with shard_manifest.json + totals + <20 GB assert +
  an INCOMPLETE resume hint. Gate/discovery cells exercised locally against fake
  /kaggle trees (80k-file mirror, missing-CSV, partial-mirror, partial-shards).
- **extract_holistic.py** gained --shard-index/--num-shards (interleaved i::n over
  the FULL sorted list, so the partition never shifts with completion state —
  regression-tested) and a projected-shard-time print at clip 200 with a warning
  past --session-budget-h (11) so NUM_SHARDS can be raised early.
- **kaggle_prep.ipynb** now merges MULTIPLE inputs: globs
  /kaggle/input/*/shard_manifest.json for extraction outputs, notes partial shard
  sets, runs map_asl_citizen from the copied splits, passes one --citizen-npz-dir
  per shard dir (build_training_tensors accepts the flag repeatedly; stems are
  asserted unique ACROSS dirs — overlapping shards fail loudly), then asserts no
  cross-domain participant collisions and prints per-domain sequence/signer counts.
  With no extraction outputs attached it degrades to the plain PopSign build.

Session 10 (2026-08-17, ASL Citizen extraction infrastructure):
- **Strategy call: signer diversity promoted to the active phase** (PopSign has 21
  signers; ASL Citizen has 52 and raw video). 1c training may run in parallel;
  nothing here depends on its outcome.
- **Extractor reality: legacy mp.solutions.holistic is GONE** from every mediapipe
  installable on Python 3.13 (tried 1.0.1 and 0.10.35). All new extraction uses the
  Tasks-API HolisticLandmarker (13 MB .task bundle in data/models/, gitignored;
  create_landmarker prints the curl command when missing). This is a recorded
  extractor seam vs the Kaggle-provenance training data: every extracted npz stores
  mediapipe_version, and the per-domain eval section is the instrument that measures
  the shift. Tasks face mesh = 478 points (468 + 10 iris) — holistic_row keeps the
  first 468 so the Kaggle 543-row layout is unchanged.
- **src/signisa/preprocess/holistic.py**: create_landmarker / detect_frame /
  holistic_row / extract_video, lazy [live] imports. live_verify.py rewritten onto
  it (was dead code under any modern mediapipe).
- **scripts/extract_holistic.py**: multiprocess (Pool + imap_unordered, one fresh
  landmarker PER CLIP — detect_for_video requires monotonic timestamps per instance
  and tracking state must not leak across clips; smoke test caught this), atomic
  .part->rename writes so npz-exists IS the completion manifest, zero-detection
  clips rejected into failures.csv and skipped on restart (delete the CSV to retry).
  Full (T,543,3) float16 stored so future landmark sets derive without re-extraction.
  Smoke-tested on 3 local videos: positive / partial / zero-detection + resume.
- **scripts/map_asl_citizen.py**: split CSVs (Participant ID, Video file, Gloss,
  ASL-LEX Code) -> data/meta/asl_citizen_mapping.csv + asl_citizen_coverage.md.
  Match order: exact normalized ASL-LEX entry, then gloss/alias (Phase 0 ALIAS both
  directions) — but a row whose code names a DIFFERENT variant of our class
  (dog_2 vs our dog_1) is excluded as a variant mismatch (visually different sign),
  reported separately. CITIZEN_ALIAS placeholder to fill from the unmatched list
  after download. Signers namespaced "ac_<id>".
- **Merged tensors**: build_training_tensors.py takes --citizen-npz-dir +
  --citizen-mapping alongside the Kaggle parquets; index.csv gains a domain column
  (popsign | asl_citizen) and participant_id is written as str. Per-signer dominance
  vote for citizen clips via hand_dominance on the npz. Mapping rows without an npz
  (extraction failures) are skipped with a count.
- **Dual-domain eval**: held_out_participants is now type-preserving and
  bit-identical to the historic selection for numeric ids (regression-tested against
  the old implementation verbatim); default_val_participants = the SAME 4 PopSign
  signers + up to cfg.n_val_citizen_signers (5) Citizen signers (always leaving >=1
  in train). metrics/report gain a per-domain section, each domain at ITS OWN FAR
  threshold — the popsign row is the number comparable to all historic runs.
  Trials now carry str sequence_id/participant + a domain column.
- kaggle_train.ipynb split cell uses default_val_participants and str-safe
  train-pid filtering.
- Adversarial-review outcome (session 10, 4 lenses + 2 refuters per finding):
  9 confirmed / 8 refuted. Fixed: (1) kaggle_diagnose cells 6+7 int() sequence-id
  casts — crash on merged tensors (citizen ids are video stems) AND a str-vs-int64
  emb_of key regression that would np.stack([]) in the sad bimodal branch even on
  legacy tensors; ids are str-normalized at every notebook seam now. (2) extraction
  pool switched multiprocessing.Pool -> ProcessPoolExecutor: a native worker crash
  (mediapipe segfault/OOM) now raises BrokenProcessPool instead of hanging
  imap_unordered forever (refuter REPRODUCED the hang with a SIGKILL'd worker).
  (3) failures.csv header flushed at creation — a hard kill used to leave a
  headerless CSV whose first data row got eaten as the header on resume.
  (4) citizen_clips re-checks mapping label ids against THIS build's labels file
  (ids are positional; stale mapping = silently scrambled labels) and hard-fails on
  a nonexistent npz dir / zero joinable clips. (5) match_row: a coded Citizen row
  gloss-matching one of the 17 null-entry classes is a variant mismatch (STORE
  carrying shop_1 IS the SHOPPING sign — the exact case Phase 0 refused to alias).
  Notable refuted: wall-clock timestamp collisions in live_verify (a full inference
  sits between samples), fps>1000 metadata (not an ASL Citizen input), os.replace
  durability (power-loss window accepted).

Session 9 (2026-08-16, Phase 1c — ragged tensors + bundled recipe):
- **Lips (v2) killed by the pre-registered criterion**: 49.5% top-1 (-0.1 vs v1),
  TAR 73.0%. Five single levers now dead and the Kaggle winners hit ~0.82 at the
  same parameter scale, so the diagnosis phase is over: Phase 1c bundles the
  untested recipe pieces in ONE run and only ablates if it works.
- **Ragged variable-length storage (shard schema v2).** build_training_tensors no
  longer resamples to T=160; sequences keep their native length, capped at
  `pipeline.MAX_FRAMES = 384` (over-long clips resample DOWN — truncating a sign
  makes it a different sign). Shards store `frames` (all sequences concatenated) +
  `lengths` + `schema_version`; `signisa.data._load_shards` rebuilds slice offsets
  and asserts the schema (a pre-v2 shard raises "shard schema 1, expected 2").
  index.csv gains `n_frames`; on the 30 local samples fixed-160 storage would have
  been **3.8x larger** (frames min 6 / median 20 / max 223).
- **Batching**: `pad_collate` pads to the batch max and emits a real-frame mask;
  `train.LengthBucketSampler` sorts by length inside a shuffled 50-batch pool so
  padding (and the BatchNorm skew it causes) stays small. `eval.compute_embeddings`
  undoes the sampler permutation so embeddings stay in dataset order.
- **Model is mask-aware end to end**: attention gets `key_padding_mask`, pooling
  weights by confidence x mask, and padded frames are re-zeroed after the stem and
  after every conv block — the stem's bias made pads non-zero, and the k=17 depthwise
  convs read 8 frames past a sequence's end, so a short clip's tail depended on
  whatever else shared its batch (caught by the pad-parity test, ~1e-3 drift).
- **Side features reach the model**: `data.side_features` = (duration_s,
  log1p(peak_speed)) -> BatchNorm1d(2) -> concatenated before the final embedding
  Linear (`head` is now `dim + 2 -> embed_dim`). The rhythm signal resampling
  destroyed now arrives both ways (native length AND scalars).
- **New augmentations** (all in `augmented`, which now takes the stored (T,N,4)
  array + side and returns both): random temporal crop (80-100%), speed-scale
  0.8-1.2x, and random horizontal flip p=0.5 via MIRROR_PERM. Duration/peak-speed
  follow the time warps. The flip is train-only symmetrization — inference still
  runs canonical right-dominant, so the locked "no naive flip" rule (about
  inference-time handedness) is intact. All config-flagged; the report prints them.
- **Schedule**: EPOCHS=300, PATIENCE=30, ArcFace s=30 m=0.3 unchanged. After 3
  epochs `train_model` prints measured s/epoch + projected hours and warns past
  `cfg.session_budget_h` (11 h) with the epoch count that would fit.
- `preprocess.resampled` is now vectorized (was a per-column np.interp loop —
  400 calls per training sample once speed-scale runs in the DataLoader).
- **Success: top-1 >= 60% OR TAR@FAR5 >= 80%. Kill: top-1 < 55%** -> the recipe
  hypothesis dies and ASL Citizen (signer diversity) becomes the next phase.

Session 8 (2026-08-14, landmark v2 + live loop):
- **200-epoch ArcFace converged: 50.3% top-1 at epoch 166, flat tail, TAR
  73.4% — under-training eliminated as a cause.** Remaining levers: input
  representation (this session) and signer diversity (Phase 2 data).
- **Landmark set v2 (99 nodes)**: hands/body/brows/eyes unchanged, mouth
  upgraded from 6 to the standard 40-point MediaPipe lip set (outer+inner
  rings). Rationale (task3a): Kaggle winners carried 18-40 lip points because
  PopSign players mouth the words — the strongest evidence-backed lever left.
  landmarks.py now builds LandmarkSet objects (v1/v2) from one face-mesh
  symmetry table; v1 arrays are bit-identical to the old constants. Mirror
  perm derived generically; lip pairs verified on real frames (straddle test)
  + involution.
- **landmark_version recorded everywhere**: shards (npz field), index.csv
  column, checkpoints (save_checkpoint/load_checkpoint in signisa.models —
  legacy raw state dicts still load as v1), trained curriculum_db, metrics
  report (+ torch.__version__). Mismatch asserts at every seam: ShardDataset
  (shape vs version), train_model, run_evaluation, both verdict CLIs.
  Config.landmark_version drives n_nodes via __post_init__.
- kaggle_prep now has LANDMARK_VERSION="v2"; kaggle_train auto-detects the
  version from index.csv — no manual sync.
- **scripts/live_verify.py**: webcam -> 3-2-1 countdown -> ~2.5s capture ->
  MediaPipe Holistic -> Kaggle-layout (543,3) rows -> per-attempt dominance ->
  preprocess at the checkpoint's landmark version (measured fps, not nominal
  30) -> verdict. --save-dir persists landmarks + verdict JSON per attempt
  (first real learner data; Phase 2 seed). --smoke <parquet> runs the same
  path headless; cv2/mediapipe imported lazily ([live] extra).
- **Phase 1b target restated: >=80% TAR@FAR5** on unseen signers with the v2
  input (waypoint toward the backlog's >90% Phase 1 criterion).
- Adversarial-review outcome (session 8): the 40-point lip set verified as
  the exact FACEMESH_LIPS node set with genuine ring-order traversal; all 23
  mirror pairs geometrically confirmed on 457 real frames; v1 arrays
  bit-identical to pre-refactor; v2 mirrored_stored bit-exact like v1.
  Three real bugs fixed: diagnose round-2 cells were hardcoded v1 (now pass
  val_ds.landmark_version); mirrored_stored silently TRUNCATED v2 arrays
  under the v1 default (now shape-asserted, ditto pipeline.mirrored); a
  version-less legacy db passed the CLI assert with ANY model version — the
  one fully silent cross path since embeddings are 512-d in both versions
  (now: missing field == v1, must match the model). Also: unpaired face
  points must be declared midline (future-proofing assert);
  validate_pipeline.py documented as v1-only.

Session 7 (2026-08-14, 200-epoch run prep):
- **Round-2 diagnosis conclusions (Kaggle): orientation hypothesis CLEARED** —
  no val participant's genuine median improves under flipping, so the
  dominance vote is fine. Label noise is small: 1.8% suspect trials (well
  under PopSign's 19%), clean-TAR barely moves. sad's low scores are diffuse
  (no bimodality, low coherence) — scattered bad clips, not a second variant.
  **Primary cause of the ~73% wall = model quality / signer diversity**, so
  the next lever is longer training (200-epoch ArcFace) and later more signers.
- kaggle_train.ipynb prepped: CONFIG defaults now arcface/200 epochs, PATIENCE
  exposed (default 30 — 10 was too twitchy for a 200-epoch cosine), training
  cell prints best-epoch + last-20 val-top1 tail, and a "## Training" section
  is appended to metrics_report.md (best epoch, epochs run, early-stopped
  flag, tail) so the report itself shows whether 200 was enough.
- Confirmed (numerically, not just by reading): warmup_cosine spans
  cfg.epochs — warm ends at epoch 5, factor 0.52 at epoch 100, 0.0 at the
  final step of epoch 200. No hardcoded 60 anywhere in the schedule.

Session 6 (2026-08-14, diagnostic round 2 — orientation test + noise bound):
- **Round-1 diagnosis findings (Kaggle):** mirrored val signers ~82% TAR /
  7.7% EER vs unmirrored 68.7% / 56.7% -> dominance-vote error suspected, not
  model failure. Worst genuine trials look like wrong variants / mislabels
  (sad x24, water at 0.09, brother->boy at 0.99) — consistent with PopSign's
  documented ~19% noise.
- **Key fact proven + tested:** canonical-space mirroring of a stored tensor
  (MIRROR_PERM + negate x, z unchanged) is EXACTLY equivalent to mirroring the
  raw sequence before preprocess (0.0 error on real data — every pipeline step
  is mirror-equivariant; z re-derives as x*up). `signisa.data.mirrored_stored`.
  This lets the diagnose notebook test both orientations WITHOUT raw parquets.
- kaggle_diagnose.ipynb round-2 cells (same inputs, CPU): (1) orientation
  test — every val curriculum attempt embedded both ways, per-participant
  median/TAR stored vs flipped vs orientation-max + flip_wins rate, with
  auto-verdicts (wrong-vote if flipped TAR +10pp; mixed-orientation clips if
  flip_wins ~50% and orientation-max helps); (2) clean-TAR bound — suspect =
  genuine < 0.45 beaten by a non-confusable; rates overall/per-sign, >15%
  flagged, TAR recomputed without suspects at the same global threshold;
  (3) sad deep-dive — histogram, largest-gap bimodality, low-cluster rival
  concentration + internal embedding coherence (>~0.6 -> real second variant,
  low -> scattered mislabels) -> data-driven paragraph. Writes
  diagnosis2_report.md. All cells dry-run locally against a 30-sample mini-run.
- Caveat noted in report design: orientation-max TAR is computed against the
  fixed global threshold (impostor scores unchanged); at deployment
  orientation-max would shift impostors too — good enough for the decisive
  comparison, recalibrate before shipping it as the fix.
- Adversarial-review outcome (session 6): mirror equivalence upgraded to
  **bit-exact** — verified 30/30 samples both directions AND 13 adversarial
  fallback synthetics (missing shoulders/nose, degenerate width, T=1);
  the equivalence test now asserts exact float16 equality. Fixes applied:
  suspect definition requires best_impostor > score (top rival is a
  non-confusable at high base rate since the random pool excludes confusables
  — without the clause, "suspect" collapsed toward plain score<0.45 and
  inflated the noise estimate); orientation verdicts key on threshold-free
  genuine MEDIANS (TAR columns descriptive, caveat in report); sad histogram
  bins cover [-1, 1]. Stale 10-channel data/tensors_smoke deleted.

Session 5 (2026-08-14, decision layer — task3b Part 2, CPU-only):
- `src/signisa/decision/policy.py`: DecisionConfig (user_level 0..1,
  margin_delta 0.05, tau_bg 0.2 placeholder until Phase 4 OpenMax) +
  verify_attempt -> Verdict (JSON-serializable dataclass). Chain: garbage gate
  (max cosine over all curriculum centroids < tau_bg -> "not_signing") ->
  per-sign threshold eer + level*(far5 - eer), clamped stricter-ward when
  far5 < eer inverts at small n (threshold_clamped flag) -> score < threshold
  -> "inaccurate" -> margin-over-confusables (score - rival cosine >=
  margin_delta for every in-db confusable) -> "confusable" naming the
  offender; else accept "ok" with worst-margin details.
- Confusables outside the 50-sign curriculum have no centroid in the db and
  are skipped by the margin check — margin coverage grows with the curriculum.
- tests/test_decision.py: 8 tests on a hand-built orthogonal fake db (clean
  accept, confusable naming the right sign, garbage, below-threshold,
  user-level borderline flip, inverted-threshold clamp, untrained raise,
  JSON round-trip). No model needed.
- scripts/verify_attempt.py: parquet -> per-sequence dominance (the unseen-
  participant fallback) -> preprocess -> embedder -> verdict JSON. --untrained
  smoke-tested on data/samples: random embedding correctly rejected as
  "not_signing" by the garbage gate; user_level 1.0 + clamp path exercised.
- Smoke gotcha: a curriculum_db_trained.json from a mini run fills centroid
  and thresholds independently (centroid needs train examples, thresholds
  need val trials targeting the sign) — verify_attempt requires BOTH.
  Also: db centroid dims must match the checkpoint's embed_dim (a small-config
  db (64-d) can't be used with the default 512-d model).
- Adversarial-review fixes (session 5): zero/NaN embeddings raised (NaN sailed
  through every `<` reject gate to accept); eval now writes
  "confusable_centroids" for out-of-curriculum rivals — 194/280 confusable
  references point outside the 50-sign curriculum and were silently skipped
  by the margin check; user_level clamped to [0,1]; CLI --untrained/--checkpoint
  mutually exclusive + comprehensible error on architecture mismatch.
- **Trained dbs from the existing Kaggle run predate confusable_centroids** —
  rerunning kaggle_diagnose.ipynb (it calls run_evaluation) regenerates
  curriculum_db_trained.json with the section; no retraining needed.

Session 4 (2026-08-13, Phase 1 results + diagnostic):
- **Phase 1 Kaggle results (full asl-signs, signer-independent 4-of-21 val):**
  CE 72.9% TAR@FAR5 / 46.3% top-1 / 15.3% mean EER; ArcFace 72.6% / 48.4% /
  13.8%. Cluster overlaps improved nearly across the board under ArcFace
  (family 30.8% -> 19.6%); happy/please still flagged at 51.3%.
  **Success criterion (>90% TAR@FAR5) NOT met.** Both losses hit the same
  ~73% TAR wall -> suspected shared cause in the data or a val signer, not
  the loss function.
- Built notebooks/kaggle_diagnose.ipynb (CPU): inputs discovered by glob (no
  hardcoded mounts), loss inferred from state_dict keys ('head.bias' => ce).
  Reports TAR/mean-EER/genuine-spread by val participant with mirrored flags
  (bad-signer / wrong-mirroring hypothesis), per-sign TAR ascending, and the
  100 worst genuine trials with the centroid that beat them ->
  diagnosis_report.md + worst_genuine.csv. Analysis logic dry-run locally
  against a 30-sample mini-run before shipping.
- run_evaluation now reports the per-participant breakdown by default (in
  metrics + metrics_report.md) and returns the full trial table
  (metrics["trials"]: sequence_id, participant, attempt, target, score,
  genuine) for downstream diagnostics.

Session 3 (2026-08-05, Phase 1 prep — everything CPU-smoke-tested, nothing
trained for real yet):
- Tensor storage slimmed to (160, 65, 4) float16 = xyz + confidence (~8 GB for
  94k vs ~20 GB at 10 channels — Kaggle output cap). `signisa.data.ShardDataset`
  re-derives velocity+bones via the shared `with_derived_channels` helper so
  reconstruction matches preprocess() exactly (tested to float16 rounding).
- Train-time augmentations in data.py (off for val): temporal span masking
  (<=40% frames), affine jitter (rot +-5deg / scale +-5% / trans +-0.02),
  whole-sequence node dropout p=0.05, gaussian noise sigma=0.005 on present
  nodes. Augment BEFORE channel derivation. Never horizontal flip.
- Model: cnn_transformer per 3A Kaggle-1st pattern — Linear stem 650->192,
  3 depthwise-conv blocks (k=17, DropPath 0.2), 2 BatchNorm transformer
  layers, confidence-weighted mean pool, 512-d L2-normed embedding. 1.36M
  params (<2M budget). Heads: plain CE or ArcFace (s=30, m=0.3).
  All knobs in signisa.config.Config.
- Trainer: AdamW, warmup->cosine, AMP on CUDA, early stop on val top-1.
- Eval (signisa/eval.py): 4-of-21 seeded signer-independent split, centroids
  from train participants only, genuine vs confusable+20-random impostor
  trials, global TAR@FAR5, per-sign EER/thresholds, per-cluster histogram-
  overlap collapse check (flag > 50%), closed-set top-1 anchor. Writes
  metrics_report.md + curriculum_db_trained.json (centroids + thresholds).
- notebooks/kaggle_prep.ipynb (CPU: clone+install -> full tensor build,
  asserts < 15 GB) and kaggle_train.ipynb (GPU: CONFIG cell -> train -> eval
  -> model.pt). Thin wrappers; all logic lives in the package.
- CPU smoke: overfit-10-sequences hits 100% top-1 in <=50 steps for BOTH
  losses; end-to-end mini-run (2 epochs, 2-participant val) produces
  well-formed report + trained db. 25 tests green.
- Phase 1 success: >90% TAR@FAR5 unseen signers; kill: cluster collapse.
  Decided on the Kaggle run, not locally.

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
  detector missed — `load_wrists`' filtered read relies on this (now asserted).
- wrist_score known ceiling: speed comes only from consecutive present frames,
  so isolated-frame presence scores 0; per-participant majority voting absorbs
  it (verified sensible on all 30 real samples).
- load_asllex latent shadowing: bases 'what' and 'breakdown' are served by
  'W.H.A.T' / 'break_down' (punctuation sorts before letters). Neither is used
  by the 250; matters only if the vocabulary grows.
- ShardDataset holds the whole shard set in RAM behind an lru_cache so
  train/val/eval instances share ONE ~8 GB array; if Kaggle OOMs, switch the
  build to raw .npy + mmap (noted in data.py).
- kaggle_train.ipynb's TENSORS_DIR depends on the prep notebook's output slug —
  it's in the CONFIG cell for that reason.
- Augmentation RNG is seeded from torch per __getitem__ (DataLoader workers
  fork numpy state identically; torch reseeds per worker).
- Adversarial-review fixes (session 3): temporal-mask spans are clamped to the
  40% budget (loop overshot to 48% before); ArcFace head runs fp32 with
  autocast disabled (fp16 acos at cos=+-1 gave inf grads) and guards the
  theta > pi-m region with the standard linear-penalty fallback; FAR
  thresholds are order-statistic based (interpolated quantiles exceeded the
  5% target at n = 1 mod 20); cluster overlap is rank/AUC-based, 2*(1-AUC)
  (40-bin histogram overlap read ~0.57 for identical 50-sample dists and
  under-flagged collapse). Inverted separation (impostors above genuine)
  reports overlap 1.0 by design.
- Known accepted nitpicks: scheduler.step() advances on AMP-skipped steps
  (rare after the fp32 ArcFace fix); eer_of is O(thresholds x trials) memory —
  per-sign scale only, documented in its docstring.
- per-participant mean_eer degenerates to a 0/1 win-rate at smoke scale
  (1 genuine x 1 impostor per sign-group); meaningful only at full scale
  (~18 attempts/sign/participant). The diagnose notebook's "beaten_by" column
  is the top rival centroid — it only literally "beat" the genuine when
  margin < 0.
- Real sequences can be as short as 6 frames (~0.2 s) and as long as 223.
- Shards from before schema v2 are unreadable by design (the `tensors` key is gone,
  and the schema assert fires first) — rerun kaggle_prep, don't patch old outputs.
- The BatchNorms still see padded frames; LengthBucketSampler is what keeps that
  fraction small. If pad share ever grows (much longer clips, tiny batches),
  masked BatchNorm is the upgrade, not more bucketing.
- `SHARD_SCHEMA` lives in `signisa/__init__.py`, not `signisa.data`: the tensor-build
  script must stay importable without torch (it's the `[train]` extra).
- Checkpoints from before session 9 are unloadable — `embedder.head` grew by the two
  side-feature inputs (dim+2) and `side_norm` is new. Retrain, don't port.
- participant_id is a STRING everywhere from session 10 (merged indexes mix ac_
  strings with numeric PopSign ids). pandas re-infers int64 on pure-numeric CSVs, so
  every reader normalizes with astype(str); metrics keys / trials.participant are str.
- mediapipe pinned by reality, not pyproject: [live] needs a Tasks-API build
  (>=0.10.30); mp.solutions imports are dead. HolisticLandmarker VIDEO mode requires
  strictly increasing timestamps PER INSTANCE — never reuse a landmarker across clips.
- The extraction failures.csv doubles as a skip-list on restart; deleting it retries
  failures. Zero-detection is deterministic, so retries only matter after code changes.

## Missing / absent from expected inputs

- CLAUDE.md, docs/experiment-backlog.md, memory.md: not in the repo (expected
  by the Phase 0 brief). This file now seeds memory.md.
- Everything expected in the datasets was present (sign map, train.csv,
  parquets, ASL-LEX database CSV). Nothing missing.
