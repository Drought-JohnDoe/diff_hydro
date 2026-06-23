# Reproducibility

## Environment

```bash
conda env create -f environment.yml
conda activate diff-hydro-publication
```

## Raw data root

By default the wrappers look for retained raw data in:

```bash
export MODEL6_PUBLICATION_DATA_ROOT=/home/mircore/Desktop/diff_hydro
```

## Train

```bash
python model6/train_model6.py --subset 671 --epochs 1 --run-name smoke_test
```

Subset options:

- `32`
- `455`
- `671`
- `custom --custom-basin-list path/to/list.txt`

## Evaluate

```bash
python model6/evaluate_model6.py
python model6/evaluate_model6.py --recompute --with-figures
```

## Figures

```bash
python model6/make_figures.py
python model6/run_rohini_replication_figures.py
```

## Checkpoint

- main checkpoint: `model6/checkpoints/best_model6_checkpoint.pt`

The checkpoint is small enough for standard Git storage. Git LFS is optional only if larger future checkpoints are added.

