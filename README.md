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

Runs write `config.json`, `train.log`, `metrics.jsonl`, `best.pt` and `last.pt` into
`run.output_dir`. Validation reports accuracy, UF1 and UAR for both heads; UF1/UAR are
the MEGC-style metrics that ignore the heavy class imbalance of micro-expression data.

## Configs

- `configs/base.yaml` — full pipeline, magnification off
- `configs/magnified.yaml` — full pipeline with Eulerian magnification
- `configs/flow_only.yaml` — ablation without the ViT branch

## Development

```bash
ruff check . && ruff format --check .
mypy
pytest -q
```

## Status

The training and evaluation machinery is complete and covered by tests on synthetic
clips. Dataset-specific pieces intentionally left out: face detection/alignment (do it
offline, it depends on the corpus), manifest builders for CASME II / SAMM / DFEW, and
leave-one-subject-out cross-validation driving.
