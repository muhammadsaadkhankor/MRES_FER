# MRES_FER

Micro-expression guided macro facial expression recognition on MMEW.

Frozen ViT frame features feed two branches: short overlapping windows go
through a motion/temporal encoder producing latent micro-clues (trained with a
contrastive loss over macro windows *and* real micro-expression clips, plus a
temporal-consistency term), and the full clip goes through a macro transformer
that consumes those clues before the classifier.

See [REPORT.md](REPORT.md) for the design, dataset assumptions and the
framework-to-code mapping.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
# edit configs/default.yaml -> paths.data_root (MMEW_Final), paths.micro_annotations, paths.work_root
python setup_dirs.py --config configs/default.yaml
python feature_extractor.py --config configs/default.yaml
python train.py --config configs/default.yaml
python evaluation.py --config configs/default.yaml --checkpoint workdir/checkpoints/last.pt --save-embeddings
python visualize.py --config configs/default.yaml --checkpoint workdir/checkpoints/last.pt
```

Override any config value inline, e.g. `python train.py --set train.epochs=60 loss.w_contrastive=0`.

MMEW is small (180 macro clips), so a single 6-subject test split is very noisy. For the
numbers you report, use subject-independent 5-fold cross-validation instead:

```bash
python cross_validate.py --config configs/default.yaml   # -> workdir/results/cross_validation.json
```

Frozen ImageNet features cap accuracy around 0.30 on MMEW; `finetune.py` trains the same
model end to end with the last ViT blocks unfrozen (reads the JPEG frames, no extraction):

```bash
python finetune.py --config configs/default.yaml
```

Smoke test without the dataset:

```bash
python scripts/make_dummy_dataset.py --root /tmp/MMEW_Dummy
python feature_extractor.py --set paths.data_root=/tmp/MMEW_Dummy paths.work_root=/tmp/dummy_work extractor.device=cpu
```
