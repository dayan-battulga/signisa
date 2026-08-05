# Signisa

Known-target ASL sign validation for a Duolingo-style learning app. The app always knows
which sign the learner is attempting; the model verifies the attempt against that target
and returns phonological feedback — verification, not open-vocabulary recognition.

Design docs (Tasks 1, 2, 3A, 3B + experiment backlog) live in the attached Claude project.

## Layout

```
src/signisa/preprocess/   landmark selection, normalization chain, Kaggle loader
src/signisa/models/       ArcFace embedding model + phonological heads (Phase 1)
src/signisa/decision/     thresholds, margin-over-confusables, open-set rejection (Phase 1+)
scripts/                  coverage analysis, curriculum_db builder
notebooks/                Kaggle training notebooks
data/meta/                small metadata (class lists); big datasets stay out of git
tests/                    pytest suite (synthetic data)
```

## Pipeline (implemented)

Raw MediaPipe Holistic landmarks `(T, 543, 3)` →
mirror if left-dominant → short-gap fill → root-center (shoulder mid)
→ shoulder-width scale → canonical rotation → One Euro filter → resample to `T=160`
→ `(160, 65, 10)` tensor: xyz + velocity + bone vectors + confidence,
plus raw-duration / peak-speed side features.

## Develop

```
pip install -e ".[dev]"
pytest
```
