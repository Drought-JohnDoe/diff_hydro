# Code Differentiable Hydrology

This is a clean Model 6 focused codebase extracted from the larger `diff_hydro` workspace.

It keeps only the DynamicSimHyd Model 6 workflow:

- training
- resume training
- testing
- analysis
- HBV comparison reporting

It intentionally does **not** include:

- HBV training code
- older SIMHYD variants
- Model Five, Model Seven, or routing-only ablations
- raw CAMELS data

## Folder Layout

- `code/dPLHBVrelease/hydroDL-dev/hydroDL/`
  Core `hydroDL` package used by Model 6.
- `code/dPLHBVrelease/hydroDL-dev/example/model_six/`
  Model 6 train/test/analyze/resume scripts.
- `outputs/rnnStreamflow/CAMELSMODELSIX/`
  Saved Model 6 checkpoints and run metadata.
- `outputs/report_model_six_vs_hbv_ep20/`
  Saved Ep20 comparison figures and tables versus the HBV benchmark.
- `benchmarks/hbv_ep10/`
  Saved HBV10 benchmark evaluation and summary plots used for comparison.
- `report_model_six_vs_hbv.py`
  Report generator for Model 6 versus HBV.
- `run_model6.py`
  Small config-driven runner for train/resume/test/analyze/report/NSE checks.
- `configs/`
  Ready-to-edit JSON configs.
- `notebooks/model6_runner.ipynb`
  Minimal notebook workflow for resuming and reevaluating the model.

## Final Model

The best Model 6 run bundled here is the 20-epoch all-671 benchmark:

- train period: `1980-10-01` to `1995-10-01`
- test period: `1995-10-01` to `2010-10-01`
- warmup: `365` days
- `rho`: `365`
- batch size: `32`
- windows per epoch: `200`
- basins: all `671` CAMELS basins

Best checkpoint:

- `outputs/rnnStreamflow/CAMELSMODELSIX/DynamicSimHydModelSix/AllBasins/daymet/111111/`
  `T_19801001_19951001_BS_32_HS_64_RHO_365_Buff_365_Mul_4_Route_1_CmpW_1_LGDyn_1_DSQ_1_DETGAM_1_DPART_1_DCFMAX_1_DROUTE_0_CRoute_1_DryCh_1_ZGate_1_MaxIter200_All671_BS32_HS64_MaxIter200/model_Ep20.pt`

Saved evaluation outputs:

- `outputs/rnnStreamflow/CAMELSMODELSIX/DynamicSimHydModelSix/AllBasins/daymet/111111/Train19801001_19951001Test19951001_20101001_ModelSixAll671_BS32_HS64_MaxIter200/`

Saved analysis:

- `outputs/rnnStreamflow/CAMELSMODELSIX/DynamicSimHydModelSix/AllBasins/daymet/111111/analysis_ep20/`

Saved HBV comparison:

- `outputs/report_model_six_vs_hbv_ep20/`
- local HBV benchmark:
  - `benchmarks/hbv_ep10/Eva10.npy`
  - `benchmarks/hbv_ep10/summary_stats.csv`

## Reported Ep20 Result

- median NSE: `0.6907`
- median KGE: `0.7255`
- median logNSE: `0.5770`
- median low-flow NSE: `-11.5291`
- median high-flow NSE: `0.5900`

## Data

This package does not include CAMELS forcing or observation files.

Set the data root with:

```bash
export DYNAMIC_SIMHYD_ROOT_DB=/path/to/Camels
```

Optionally set a custom output root:

```bash
export DYNAMIC_SIMHYD_ROOT_OUT=/path/to/outputs/rnnStreamflow
```

## Train

```bash
cd code/dPLHBVrelease/hydroDL-dev/example/model_six
conda run -n pytorch python trainModelSix.py \
  --epochs 10 \
  --batch-size 32 \
  --rho 365 \
  --hidden-size 64 \
  --save-epoch 1 \
  --max-iter-ep 200 \
  --nmul 4 \
  --use-all-basins \
  --gpu-id 0
```

## Resume

```bash
cd code/dPLHBVrelease/hydroDL-dev/example/model_six
conda run -n pytorch python continueModelSix.py \
  --start-epoch 5 \
  --end-epoch 20 \
  --batch-size 32 \
  --rho 365 \
  --hidden-size 64 \
  --max-iter-ep 200 \
  --nmul 4 \
  --use-all-basins \
  --gpu-id 0
```

## Test

```bash
cd code/dPLHBVrelease/hydroDL-dev/example/model_six
conda run -n pytorch python testModelSix.py \
  --epoch 20 \
  --batch-size 32 \
  --rho 365 \
  --hidden-size 64 \
  --nmul 4 \
  --use-all-basins \
  --test-batch 64 \
  --max-iter-ep 200 \
  --gpu-id 0
```

## Analyze

```bash
cd code/dPLHBVrelease/hydroDL-dev/example/model_six
conda run -n pytorch python analyzeModelSix.py \
  --epoch 20 \
  --batch-size 32 \
  --rho 365 \
  --hidden-size 64 \
  --nmul 4 \
  --use-all-basins \
  --chunk-size 64 \
  --max-iter-ep 200 \
  --gpu-id 0
```

## Compare To HBV

```bash
conda run -n pytorch python report_model_six_vs_hbv.py \
  --epoch 20 \
  --result-suffix Train19801001_19951001Test19951001_20101001_ModelSixAll671_BS32_HS64_MaxIter200 \
  --analysis-dir outputs/rnnStreamflow/CAMELSMODELSIX/DynamicSimHydModelSix/AllBasins/daymet/111111/analysis_ep20 \
  --out-dir outputs/report_model_six_vs_hbv_ep20 \
  --subset-tag AllBasins \
  --hbv-eva-path benchmarks/hbv_ep10/Eva10.npy
```

## Notes

- The scripts now use `example/model_six/Sub531ID.txt` as the default subset file location.
- The report script now derives its root from its own file location, so it can run from this clean package directly.
- This package is intended to be the single Model 6 starting point for future development.

## Configurable Runner

The easiest way to rerun experiments is with:

```bash
python run_model6.py <action> --config configs/model6_resume_ep20_to25.json
```

Available actions:

- `train`
- `resume`
- `test`
- `analyze`
- `report`
- `nse`

### Resume Ep20 to Ep25

Default config:

- `configs/model6_resume_ep20_to25.json`

This is already set up for:

- `start_epoch = 20`
- `end_epoch = 25`
- `epoch = 25`

Example:

```bash
python run_model6.py resume --config configs/model6_resume_ep20_to25.json
python run_model6.py test --config configs/model6_resume_ep20_to25.json
python run_model6.py analyze --config configs/model6_resume_ep20_to25.json
python run_model6.py report --config configs/model6_resume_ep20_to25.json
python run_model6.py nse --config configs/model6_resume_ep20_to25.json
```

### Change Hyperparameters

Edit the JSON before rerunning. Good first knobs to explore:

- `max_iter_ep`
- `batch_size`
- `reg_amp_w`
- `reg_smooth_w`
- `reg_part_w`

For safe checkpoint resume, do not change architecture-defining fields like:

- `hidden_size`
- `nmul`
- dynamic/static model structure flags

unless you intend to start fresh training instead of continuing from `Ep20`.
