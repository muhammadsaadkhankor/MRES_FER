# MRES_FER

Micro-expression guided macro facial expression recognition.

**Hypothesis.** Subtle local facial movements (micro-expressions) provide fine-grained
evidence that helps infer the overall macro-expression.

## Pipeline

```
video
  └─ frames                      (face alignment done offline, dataset dependent)
       ├─ [optional] Eulerian motion magnification
       │     └─ optical-flow encoder ─┐
       └─ [optional] ViT branch ──────┤
                                      ├─ gated feature fusion   T x N x (Dv + Dm)
                                      │
                                 temporal transformer
                                      ├─ micro-expression head
                                      └─ macro-expression head  (conditioned on micro)
```

Every optional block is a config switch, so ablations are configuration changes rather
than code changes:

| Switch | Effect |
| --- | --- |
| `model.use_appearance` | enable the per-frame ViT branch (`Dv` tokens) |
| `model.use_motion` | enable the optical-flow encoder (`Dm` tokens) |
| `model.use_motion_magnification` | amplify subtle motion before the branches |
| `model.use_gate` | learn a per-token gate on the motion stream instead of plain concat |
| `model.macro_from_micro` | feed the micro posterior into the macro head |

Appearance and motion tokens share the ViT patch grid, so fusion is token-aligned:
motion token `n` describes the same image region as appearance token `n`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # or a CUDA index
pip install -e ".[dev]"
pre-commit install
```

## Data

A dataset is a JSON manifest of clips plus a directory of extracted (ideally
face-aligned) frames:

```json
[
  {
    "clip_id": "sub01_EP02_01f",
    "frames_dir": "casme2/sub01/EP02_01f",
    "macro_label": 2,
    "micro_label": 1,
    "onset": 0,
    "apex": 27,
    "offset": 55,
    "subject": "sub01"
  }
]
```

`frames_dir` is relative to `data.root` and holds frames with sortable names
(`img_00001.jpg` ...). `micro_label` may be omitted for macro-only datasets: those clips
are ignored by the micro loss, so mixed corpora can be trained jointly. `subject` is
carried through for leave-one-subject-out splits.

### MMEW

`prepare-mmew` scans an MMEW release into a manifest. Both published layouts work
(`<subset>/<emotion>/<clip>/1.jpg` and `<subset>/<subject>/<emotion>/<clip>/1.jpg`), and
the subject is taken from the `PersonIndex-EmotionIndex-SampleIndex` clip name:

```bash
mres-fer prepare-mmew --root /path/to/MMEW_Final --out data/mmew
```

The label table at the dataset root (`MMEW_Micro_Exp.xlsx`) is picked up automatically;
`--micro-annotations`/`--macro-annotations` override the choice.

MMEW records both expression types from the same subjects, which is what makes the
micro-to-macro hypothesis testable on it:

- micro clips get a micro label **and** the macro label of the same emotion, so they
  supervise both heads;
- macro clips leave `micro_label` at the ignore index and only supervise the macro head;
- `repression` exists only in the micro taxonomy, so those clips ignore the macro head.

Class indices are alphabetical and derived from the emotion directories present in *your*
copy, since releases differ (some ship the micro-only `repression`, some only the six
emotions shared with the macro side). They are written to `data/mmew/labels.json`, and
the command prints the `model.num_micro_classes` / `model.num_macro_classes` your config
needs.

The onset/apex/offset columns of the spreadsheets are absolute recording frame numbers,
while the clip directories are already trimmed, so indices are rebased onto the clip and
clamped and `data.sampling: apex_centered` works directly.

Optical flow is the slowest part of dataloading. Cache it once:

```bash
python scripts/precompute_flow.py --config configs/base.yaml --split train
python scripts/precompute_flow.py --config configs/base.yaml --split val
```

## Usage

```bash
mres-fer summary  --config configs/base.yaml --override run.device=cpu
mres-fer train    --config configs/base.yaml
mres-fer evaluate --config configs/base.yaml --checkpoint runs/base/best.pt
```

`--override section.key=value` is repeatable and parsed as YAML, e.g.
`--override optim.batch_size=4 --override model.use_gate=false`.

Leave-one-subject-out is the standard protocol for micro-expression benchmarks; it takes
a single manifest and trains one model per held-out subject:

```bash
mres-fer loso --config configs/mmew.yaml --manifest data/mmew/manifest.json
```

Each fold writes to `run.output_dir/fold_<subject>/`, and per-fold plus mean metrics land
in `run.output_dir/loso_summary.json`. `--subject S03` (repeatable) restricts the run to
a few folds, which is the cheap way to sanity-check a config first.

Runs write `config.json`, `train.log`, `metrics.jsonl`, `best.pt` and `last.pt` into
`run.output_dir`. Validation reports accuracy, UF1 and UAR for both heads; UF1/UAR are
the MEGC-style metrics that ignore the heavy class imbalance of micro-expression data.

## Configs

- `configs/base.yaml` — full pipeline, magnification off
- `configs/magnified.yaml` — full pipeline with Eulerian magnification
- `configs/flow_only.yaml` — ablation without the ViT branch
- `configs/mmew.yaml` — MMEW class counts, apex-centred sampling, LOSO output layout

## Development

```bash
ruff check . && ruff format --check .
mypy
pytest -q
```

## Status

The training and evaluation machinery is complete and covered by tests on synthetic
clips; MMEW manifest building and LOSO are covered on a synthetic copy of the MMEW
directory layout, not on the real release. Face detection/alignment is intentionally out
of scope — run it offline, it depends on the corpus and on the crop convention you want.
Other corpora (CASME II, SAMM, DFEW) need their own manifest builder; the dataset and
split code itself is corpus-agnostic.
