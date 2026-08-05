# Signisa — Experiment Backlog (post-3A/3B review gate, 2026-08-05)

Gate verdict: **PASSED — skip the Task 6 architecture deep-dive, move to build.**
Task 6 trigger (per 3B): run only if the ArcFace + ST-GCN model hits a latency/memory wall during TFLite/ONNX edge conversion; scope it narrowly to replacing temporal pooling with a linear-time SSM (Mamba).

## The committed design (from 3B, grounded in 3A)

- **Isolated signs**: ST-GCN backbone on (B, T=160, 65, 10) → 512-d embedding trained with ArcFace margin loss; verification = cosine similarity vs. per-sign centroid. Latent space factorized into 4 phonological subspaces (handshape / location / movement / palm orientation) with auxiliary heads trained on ASL-LEX/Sem-Lex phoneme labels → hierarchical single-error feedback.
- **Decision policy**: per-sign thresholds interpolated EER→low-FAR by user level; margin-over-confusables (Δ≈0.05) vs. ASL-LEX minimal-pair centroids; OpenMax/Weibull open-set rejection (fallback: energy-based OOD). All parameters in a `curriculum_db.json`.
- **Fingerspelling**: separate CTC model, **forced alignment** restricted to the known target word → per-letter scores/timing.
- **Learner shift**: temporal interpolation to fixed T=160; per-user calibration after ~20 passed signs; canonical mirroring for left-handed users (X-flip before filtering/normalization); L2 eval set protocol (full: 30 users × 50 signs × 3 attempts).
- **Fallback formulation**: ST-GCN features + soft-DTW vs. reference template (also the feedback-localization mechanism if attention mapping underperforms).

## Cross-check flags (carry into experiments)

1. **Extractor sequencing (biggest catch)**: 3B "strictly disqualifies" MediaPipe and mandates RTMW primary — but 3A's entire Kaggle evidence base was achieved ON MediaPipe landmarks, and the Kaggle asl-signs training data is **landmarks-only (no raw video)** — it cannot be re-extracted with RTMW. An RTMW-primary model forfeits that dataset and creates a train/inference extractor mismatch with PopSign-provenance data. → Build the MediaPipe tier first (browser-native, matches training-data provenance); treat RTMW as the accuracy-tier upgrade validated in Phase 1 A/B on ASL Citizen (raw video available for re-extraction). This inverts 3B's tier order for the MVP without contradicting its two-tier design.
2. **Interpolation vs. movement tension**: T=160 interpolation normalizes speed but may erase lexically contrastive speed/tension cues (3A's own WAKE vs. AWAKE example differs in movement micro-tension). → Keep raw duration + peak-velocity scalars as side features alongside the interpolated tensor.
3. **Flip reconciliation**: Kaggle winners used horizontal-flip *augmentation* successfully for 250-class classification; 3B rejects it for verification and prescribes canonical mirroring at inference (PopSign's preprocessing). Both are defensible in context — adopt canonical mirroring; optionally test flip-aug in Phase 1 as an ablation.
4. **3B citation padding**: several off-domain citations (EMG gestures, bearing faults, plant disease). The concepts (ArcFace, prototypical nets, OpenMax) are textbook-solid, but treat 3B's specific numbers as unverified — especially the "~67% → >82% forced-alignment gain" (sourced to an old HMM-era Georgia Tech thesis; plausible, unverified).
5. **Verified**: ASL Citizen 63% top-1 / 91% R@10 (real published baselines); Sem-Lex ~85% phoneme accuracy; Kaggle 1st-place 1D-CNN+Transformer ~1.8M params / 17 ms TFLite (hoyso48 writeup); PhonSSM paper exists (arXiv 2604.08761). PopSign v1.0 (210k videos, 47 signers, signdata.cc.gatech.edu) is a real downloadable source.
6. **Morphology risk (3B open question)**: shoulder-width normalization is untested on children/atypical bodies; per-user calibration partially mitigates. Revisit if kid users matter.
7. **Z-axis risk**: palm-orientation head depends on webcam depth estimates — expect this head to be the weakest; Phase 3 measures it separately.

## Backlog

### Phase 0 — Data + pipeline foundations (start now; ~1 week)
- 0.1 Acquire: Kaggle asl-signs landmarks; ASL Citizen (download/agreement); PopSign v1.0 (signdata.cc.gatech.edu); ASL-LEX 2.0 (OSF); WLASL optional (expect link rot).
- 0.2 **Coverage analysis** (code, not research): intersect a candidate beginner curriculum (~100–300 words) with dataset vocabularies → pick the v1 50-sign curriculum where every sign has enough examples AND its ASL-LEX minimal pairs are also trainable. Output: curriculum list + per-sign example counts + confusable sets.
- 0.3 Preprocessing pipeline per Task 2 spec: landmark subset → 65-node mapping → root-centering → median shoulder-width scaling → yaw correction → One Euro → confidence masking (<0.3 → zero) → T=160 cubic-spline interpolation (+ duration/velocity side features) → (B,160,65,10). Unit tests: left-hand mirroring, missing-node masks, re-run determinism.
- 0.4 `curriculum_db.json` scaffold: per-sign centroid slot, eer/low-FAR thresholds, Weibull params, minimal-pair list.

### Phase 1 — Formulation baseline (3B Stage 1)
- Train ArcFace embedding model on the 50-sign curriculum (fluent data, signer-independent split). Backbone A/B: ST-GCN (3B's pick) vs. Kaggle 1st-place 1D-CNN+Transformer recipe (~2M params) — cheap to run both; the backbone question is genuinely unresolved.
- Compare against softmax TCN baseline. **Success: >90% true-accept at 5% false-accept on held-out fluent signers. Kill: embedding collapse on minimal pairs.**

### Phase 2 — Learner domain shift (3B Stage 2)
- Smoke test first: 5–8 friends × 20 signs × 3 attempts (before committing to the full 30×50×3 protocol). Include instructed near-misses (wrong location/handshape) as targeted negatives.
- **Success: <10% absolute EER degradation vs. fluent test. Kill: >30% false-reject on legible novice attempts** → escalate mitigation (fine-tune on L2, revisit interpolation).

### Phase 3 — Sub-lexical feedback heads (3B Stage 3)
- Add 4 phonological auxiliary heads (ASL-LEX/Sem-Lex labels). **Success: ≥80–85% correct blame on the induced-error set; report palm-orientation head separately (Z-axis risk). Kill: heads over-correlate → fall back to soft-DTW segment blame.**

### Phase 4 — Open-set rejection (3B Stage 4)
- OpenMax/Weibull vs. simple energy/max-cosine baseline on idle motion, random gestures, face-touching. **Success: >95% rejection of non-signing with <2% hit to valid acceptance.**

### Phase 5 — Fingerspelling forced alignment (parallelizable)
- 27-symbol CTC model (Kaggle asl-fingerspelling recipes); restricted-path forced alignment against the known word; per-letter PLLR scores. Validate the claimed forced-alignment gain empirically.

### Phase 6 — Browser demo loop
- MediaPipe tasks-vision (JS/WASM) → preprocessing (TS port or WASM) → ONNX Runtime Web / TFJS model → decision policy → feedback UI. Countdown → record 1–4 s → score. **Budget: model <50 ms; verdict <500 ms after attempt end.**

## Deferred
- Task 6 deep-dive (only on latency-wall trigger). Original prompt 4 (streaming/continuous) until conversation practice. RTMW accuracy tier after Phase 2. NMM/facial-grammar feedback post-v1.
