# MRES-FER — Micro-expression guided macro-expression recognition (MMEW)

This report describes what was implemented, how each component maps to the
framework diagram from the professor, what the dataset assumptions are, and how
to run the pipeline command by command.

---

## 1. Research question

> Can features extracted for **micro-expression** detection guide and improve
> **macro-expression** recognition?

Hypothesis: annotated micro-expression data is scarce, but a transformer can
learn **short-term, fine-grained facial dynamics** that are useful for the
global (macro) expression — *without* requiring those dynamics to correspond
exactly to annotated micro-expressions. The micro-expression corpus is
therefore used as a source of *dynamics*, not as a second classification task.

---

## 2. Framework → code mapping

```
VIDEO
  |
FRAMES                       [optional face alignment / crop — offline]
  |
ViT features  ...........................  feature_extractor.py (frozen ViT-B/16, run once)
  |                     \
  |                      \
short overlapping windows  full video sequence
  |                                |
motion/temporal encoder            |        src/mres_fer/models.py :: MicroEncoder
(optional magnification)           |          - frame-to-frame differences × magnification
  |                                |          - 2-layer transformer over the window
latent micro-clues ----------------|        MicroEncoder output (one clue per window)
  |        \                       |
contrastive  temporal consistency  |        src/mres_fer/losses.py :: info_nce, temporal_consistency
  \          /                     |
   ----> Macro encoder <-----------/        src/mres_fer/models.py :: MacroEncoder
              |                               frame tokens + micro-clue tokens + CLS
          Classifier                        nn.Linear on the CLS embedding
```

| Diagram box | File / symbol | Notes |
|---|---|---|
| FRAMES → ViT features | `feature_extractor.py` | frozen `torchvision` `vit_b_16` (ImageNet weights), CLS embedding per frame, stored as one `[T, 768]` `.npy` per clip. Runs once; training never touches JPEGs. |
| Optional face alignment | `extractor.face_crop` in the config | simple centre crop; a proper aligner can be dropped in offline. |
| Short overlapping windows | `src/mres_fer/datasets.py :: make_windows` | `window_size=8`, `window_stride=4` → 50 % overlap, `max_windows=8` spread across the clip. |
| Motion/temporal encoder (optional magnification) | `MicroEncoder` | input = `[frame_feature ‖ magnified temporal difference]`, then a small transformer; `model.motion_magnification` scales the differences. |
| Latent micro-clues | `MicroEncoder` output `clues [B, N, 256]` | one latent clue per short window. |
| Contrastive loss | `losses.info_nce` | NT-Xent between two augmented views of each window. Macro windows **and** real micro-expression clips go through the *same* micro encoder in the same batch, so the clue space is shaped by genuine micro dynamics — no micro labels, no alignment to annotated micro-expressions required. |
| Temporal consistency | `losses.temporal_consistency` | penalises jumps between the clues of consecutive overlapping windows (they describe one continuous motion). |
| Macro encoder | `MacroEncoder` | transformer over 32 uniformly sampled frame tokens + the micro-clue tokens (+ a learned type embedding to distinguish the two) + a CLS token. |
| Classifier | `MicroGuidedFER.classifier` | linear layer on the CLS embedding, 6 macro classes. |

Total trainable parameters: **≈5.6 M** (the ViT is frozen and not part of this count).

### Training objective

```
L = w_cls · CE(logits, macro_label)
  + w_contrastive · InfoNCE(window_view_a, window_view_b)      # macro windows + micro clips
  + w_temporal   · mean ‖clue_{t+1} − clue_t‖²
```
Defaults: `w_cls=1.0`, `w_contrastive=0.5`, `w_temporal=0.1`, `temperature=0.07`.

Ablations are one-liners: `--set loss.w_contrastive=0` (no micro guidance),
`--set model.use_micro_tokens=false` (clues not fed to the macro encoder),
`--set model.motion_magnification=1.0` (no magnification).

---

## 3. Dataset assumptions (MMEW_Final)

Derived from the folder screenshots:

```
MMEW_Final/
  Macro_Expression/S01/anger/S01-07-001.jpg ...     # subject / emotion / frames
  Micro_Expression/anger/S13-07-001/1.jpg ...       # emotion / clip / frames
  MMEW_Micro_Exp.xlsx                               # Subject, Filename, OnsetFrame, ApexFrame, OffsetFrame, AUs, Estimated Emotion
```

Handling in `src/mres_fer/data_index.py`:

- **Macro**: label = emotion folder name, subject = subject folder name. A
  macro emotion folder can hold frames of several recordings, so frames are
  grouped by their filename prefix (everything before the trailing `-NNN`);
  each group becomes one clip (`S01__anger__S01-07`).
- **Micro**: label = emotion folder, clip = sub-folder, subject = prefix before
  the first `-` (`S13-07-001` → `S13`). Frames are sorted numerically
  (`1, 2, …, 10`, not `1, 10, 2`). The `others` class is ignored by default
  (`labels.micro_ignore`).
- **Annotation sheet** (optional, `paths.micro_annotations`): `.csv` or `.xlsx`.
  Column names are matched case/space-insensitively. When present, micro clips
  are sampled **apex-centred**; otherwise uniformly. Nothing else depends on it,
  so the pipeline runs even if the sheet is missing.
- **Classes**: 6 macro classes (`anger, disgust, fear, happiness, sadness,
  surprise`) — editable in `labels.macro_classes`.

### Split

**Subject-independent** by default (`split.mode: subject`): subjects are shuffled
with the training seed and partitioned into test (20 %) / val (15 %) / train, so
no subject appears in two sets. The split is written to
`workdir/manifests/splits.json` and reused by `evaluation.py` and `visualize.py`.

With only 30 subjects a single 6-subject test split has a standard error of roughly
±8 accuracy points, so `cross_validate.py` runs subject-independent k-fold
(`split.num_folds`, default 5): every subject is tested exactly once and the report
is a mean ± std. Those are the numbers to quote.

### Regularisation (why the first real run overfitted)

The first 30-epoch run on MMEW reached train accuracy 1.000 by epoch 17 while
validation accuracy plateaued at 0.458 — 120 training clips are simply memorised.
The defaults now include, applied to the training split only:

| knob | default | effect |
| --- | --- | --- |
| `augment.temporal_crop` | 0.3 | samples the 32 frames from a random 70–100 % sub-segment |
| `augment.feature_noise` | 0.05 | gaussian noise on the ViT features |
| `augment.frame_dropout` | 0.1 | replaces random frames with the clip mean |
| `augment.mixup_alpha` | 0.4 | mixes two clips and their labels in feature space |
| `model.dropout` | 0.3 | up from 0.1 |
| `model.macro_layers` | 2 | down from 4 (5.6M → fewer parameters) |
| `train.weight_decay` | 0.05 | up from 0.01 |
| `loss.label_smoothing` | 0.1 | up from 0.05 |
| `train.early_stopping_patience` | 15 | stops once validation stops improving |

Note that with mixup the reported *train* accuracy is measured on mixed inputs
against the original labels, so it is pessimistic by design and should no longer
reach 1.000.

---

## 4. Repository layout

```
configs/default.yaml           all knobs (paths, sampling, model, losses, training)
setup_dirs.py                  step 0 — create the output directories
feature_extractor.py           step 1 — frozen ViT features + manifests
train.py                       step 2 — training, saves last.pt and best.pt
cross_validate.py              optional — subject-independent k-fold, mean ± std
finetune.py                    optional — end-to-end training, last ViT blocks unfrozen
evaluation.py                  step 3 — metrics, predictions, confusion matrix
visualize.py                   step 4 — figures
src/mres_fer/config.py         YAML config + workspace paths
src/mres_fer/data_index.py     MMEW folder scanner + annotation reader
src/mres_fer/datasets.py       feature datasets, window builder, subject split
src/mres_fer/models.py         MicroEncoder, MacroEncoder, MicroGuidedFER
src/mres_fer/losses.py         InfoNCE + temporal consistency
src/mres_fer/utils.py          seeding, device, logging, JSON IO
scripts/make_dummy_dataset.py  tiny synthetic MMEW tree for smoke tests
```

Outputs (under `paths.work_root`, default `./workdir`):

```
workdir/features/{macro,micro}/<clip_id>.npy
workdir/manifests/{macro_index.csv, micro_index.csv, splits.json}
workdir/checkpoints/{last.pt, best.pt}
workdir/results/{metrics_test.json, predictions_test.csv, confusion_matrix_test.csv, training_history.json}
workdir/figures/{training_curves.png, confusion_matrix.png, embeddings_tsne.png, micro_clue_timeline.png}
workdir/logs/*.log
```

---

## 5. How to run (command by command)

```bash
pip install -r requirements.txt

# edit configs/default.yaml -> paths.data_root, paths.micro_annotations, work_root

# 0) directories (also reports whether the dataset folders were found)
python setup_dirs.py --config configs/default.yaml

# 1) features — frozen ViT over every macro and micro clip (run once)
python feature_extractor.py --config configs/default.yaml
#    only one domain / re-extract / quick debug:
python feature_extractor.py --config configs/default.yaml --domain micro
python feature_extractor.py --config configs/default.yaml --overwrite --limit 20

# 2) training — checkpoints saved every epoch (last.pt) plus best val (best.pt)
python train.py --config configs/default.yaml

# 3) evaluation on the held-out subjects
python evaluation.py --config configs/default.yaml --checkpoint workdir/checkpoints/last.pt --save-embeddings

# 4) figures
python visualize.py --config configs/default.yaml --checkpoint workdir/checkpoints/last.pt

# 5) optional — end-to-end fine-tuning of the last 4 ViT blocks (reads JPEG frames,
#    uses the split in manifests/splits.json, or --fold i)
python finetune.py --config configs/default.yaml

# 6) (recommended) subject-independent 5-fold cross-validation -> results/cross_validation.json
python cross_validate.py --config configs/default.yaml
#    one fold only (checkpoints are suffixed _fold<i>):
python train.py --config configs/default.yaml --fold 0
```

Any config value can be overridden without editing the YAML:

```bash
python train.py --set train.epochs=60 train.batch_size=16 loss.w_contrastive=0.0
python train.py --set train.device=cpu train.amp=false     # CPU debug run
```

---

## 6. Verification done so far

The dataset itself is not on the development machine, so the whole pipeline was
smoke-tested on a synthetic MMEW-shaped tree
(`python scripts/make_dummy_dataset.py --root /tmp/MMEW_Dummy`, 6 subjects ×
6 classes, macro + micro):

- `feature_extractor.py` → 72 macro and 36 micro clips, `[T, 768]` features + manifests
- `train.py` (2 epochs, CPU) → losses decrease, `last.pt` / `best.pt` / history written
- `evaluation.py` → metrics, predictions, confusion matrix, embeddings
- `visualize.py` → all four figures

Accuracy on random noise is meaningless (≈chance) — the run only proves that
shapes, splits, checkpointing and IO are correct end to end.

---

## 7. Things to decide / next steps

1. **Face alignment** — currently an optional centre crop. If the professor
   wants proper alignment, do it offline once and point `data_root` at the
   aligned frames.
2. **Macro clip grouping** — confirm whether one macro emotion folder per
   subject is a single recording or several; the prefix grouping handles both,
   but the real number of clips should be sanity-checked against the macro CSV.
3. **Evaluation protocol** — subject-independent k-fold is implemented
   (`cross_validate.py`); full LOSO is the same loop with
   `--folds <number of subjects>`, just ~30× the compute.
4. **Micro supervision** — currently unsupervised (dynamics only). A supervised
   contrastive variant using micro labels is an easy ablation.
5. **Frozen features are the ceiling** — measured on MMEW (5-fold,
   subject-independent, chance 0.167): no micro branch 0.294 ± 0.038, micro tokens
   without contrastive 0.300 ± 0.041, full model 0.317 ± 0.028. The micro branch
   helps in the predicted direction but every variant is capped around 0.30;
   regularisation changes nothing (0.328 ± 0.057 unregularised) and face-cropped
   features do not help either (0.328 ± 0.057), which points at the frozen
   ImageNet ViT features rather than the head.
6. **Fine-tuning lifts the ceiling** — `finetune.py` with the last 6 ViT blocks
   unfrozen (backbone lr 2e-5, 45 epochs) reaches 0.444 accuracy / 0.402 macro-F1
   on the held-out subjects, well above every frozen-feature variant. These are
   now the defaults in `configs/default.yaml`.
7. **Backbone** — any `timm` ViT name also works (`extractor.backbone`), e.g. a
   face-pretrained ViT would likely beat ImageNet weights.
