# Model 6 Public Release

This folder is a clean public-release package for the 30-epoch `Model 6` run on the `671` CAMELS-US basins. It keeps the trained checkpoint, epoch-30 metrics, selected plots, a runnable 5-basin demo, and the Model 6 code path required to retrain or evaluate the model.

## Recommended Python Environment

The original Model 6 code path in this release is tied to the `hydroDL` runtime used during training. On this machine, the working environment is:

```bash
conda activate mhpihydrodl
```

If you prefer one-off commands, replace `python` with:

```bash
/home/mircore/anaconda3/envs/mhpihydrodl/bin/python
```

## What Is Included

- `checkpoints/epoch30/`
  - `model_Ep30.pt`
  - `master.json`
  - `statDict.json`
  - training logs from the epoch-30 run
- `results/epoch30/`
  - epoch-30 evaluation arrays: `Eva30.npy`, `pred30.npy`, `obs.npy`
  - summary and per-basin metrics
  - CONUS parameter maps
  - basin-scale flux and diagnostic maps
- `results/representative_basins/`
  - one poor, one moderate, and one very good basin with metrics and streamflow plots
- `benchmarks/hbv_ep10/`
  - the saved HBV epoch-10 benchmark used for comparison
- `demo_data/`
  - one combined static-attributes CSV for 5 basins
  - one dynamic CSV per demo basin
- `notebooks/end_to_end_demo_5_basins.ipynb`
  - end-to-end demo training and evaluation on the included 5-basin sample
- `scripts/`
  - small public entry-point scripts
- `code/dPLHBVrelease/hydroDL-dev/`
  - the minimal `hydroDL` and `example/model_six` runtime required to run Model 6

## What Is Excluded

- the full raw CAMELS-US forcing/observation dataset for all `671` basins
- the full global Caravan cache and training outputs
- other model families such as HBV training, Model Five, Model Seven, or arid-only architecture experiments

The full raw data are intentionally excluded. Users should prepare or download the CAMELS-US data themselves before running the full `671`-basin pipeline.

## Folder Structure

```text
model6_public_release/
├── benchmarks/
├── checkpoints/
├── code/
├── configs/
├── demo_data/
├── notebooks/
├── results/
└── scripts/
```

## Representative Basins Included

- `poor`: `07142300`
- `moderate`: `01639500`
- `very good`: `14301000`

Their metrics are in:

- `results/representative_basins/representative_basin_metrics.csv`

Their streamflow plots are in:

- `results/representative_basins/`

## Demo Notebook

The notebook uses only the included 5-basin demo data.

Open:

- `notebooks/end_to_end_demo_5_basins.ipynb`

It will:

1. inspect the included demo data
2. train a small 5-basin Model 6 example
3. evaluate that demo checkpoint on the held-out test period
4. save demo metrics and demo hydrographs

## Run the Demo From the Command Line

Train the included 5-basin demo:

```bash
cd model6_public_release
python scripts/train.py demo --epochs 2 --run-name demo_5_basins
```

Evaluate the saved demo checkpoint:

```bash
python scripts/evaluate.py demo \
  --checkpoint outputs/demo_training/demo_5_basins/model_Ep2.pt \
  --run-name demo_5_basins
```

## Train or Evaluate the Full 671-Basin Model

Prepare the CAMELS-US raw data externally and point the release scripts to that data root.

Train Model 6 on all 671 basins:

```bash
python scripts/train.py full671 \
  --data-root /path/to/Camels \
  --epochs 30 \
  --batch-size 32 \
  --rho 365 \
  --max-iter-ep 200 \
  --gpu-id 0
```

Evaluate an existing full run:

```bash
python scripts/evaluate.py full671 test \
  --data-root /path/to/Camels \
  --epoch 30 \
  --gpu-id 0
```

Analyze the epoch outputs:

```bash
python scripts/evaluate.py full671 analyze \
  --data-root /path/to/Camels \
  --epoch 30 \
  --gpu-id 0
```

Run the sequential full pipeline:

```bash
python scripts/run_full_671_basins.py \
  --data-root /path/to/Camels \
  --epochs 30 \
  --epoch-to-evaluate 30 \
  --gpu-id 0
```

Resume training later:

```bash
python scripts/continue_training.py \
  --data-root /path/to/Camels \
  --start-epoch 30 \
  --end-epoch 35 \
  --gpu-id 0
```

## Notes on the Local and GitHub Copies

The local release copy keeps the large raw epoch-30 diagnostics archive:

- `results/epoch30/diagnostics/model_six_diagnostics_ep30.npz`

The GitHub copy keeps the same folder structure, but the oversized raw diagnostics archive can be omitted if needed for GitHub file-size limits. All summary metrics, maps, checkpoint files, demo data, scripts, and representative plots remain in the GitHub-ready copy.
