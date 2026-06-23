import json
import os
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(os.environ.get("MODEL6_PUBLICATION_ROOT", str(Path(__file__).resolve().parents[3]))).resolve()
DATA_ROOT = Path(os.environ.get("MODEL6_PUBLICATION_DATA_ROOT", "/home/mircore/Desktop/diff_hydro")).resolve()
SOURCE_ROOT = REPO_ROOT / "model6" / "source"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "hbv_module"))

from hydro_ml.diagnosis import calc_fhv, calc_flv, calc_kge, calc_nse, calc_r2, highflow_nse, lowflow_nse, safe_ratio  # noqa: E402
from hydroDL.model import crit, train  # noqa: E402

import ECO_HYBRID.run_model6_closed_snow_asrz_simhyd_simple_demo32 as simple_demo  # noqa: E402
from ECO_HYBRID.run_ecohbv_asrz_hbv11b_demo32_fixed import LAI_GAPFILLED_FILE, build_lai_daily_for_basins  # noqa: E402
from Model_six_physical.run_model6_physical_fix import prepare_data as prepare_full671_base, to_device  # noqa: E402

try:  # optional legacy comparison branch
    import ECO_HYBRID.run_model6_physical_asrz_minimal_demo32 as asrz_demo  # type: ignore # noqa: E402
except ModuleNotFoundError:  # pragma: no cover
    asrz_demo = None


SEED = 111111
GPU_ID = int(os.environ.get("MODEL6_LAIECO671_GPU_ID", "1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "0"))
EPOCHS = int(os.environ.get("MODEL6_LAIECO671_EPOCHS", "10"))
START_EPOCH = int(os.environ.get("MODEL6_LAIECO671_START_EPOCH", "1"))
BATCH_SIZE = int(os.environ.get("MODEL6_LAIECO671_BATCH_SIZE", "256"))
RHO = int(os.environ.get("MODEL6_LAIECO671_RHO", "365"))
MAX_ITER_EP = int(os.environ.get("MODEL6_LAIECO671_MAX_ITER", "20"))
LR = float(os.environ.get("MODEL6_LAIECO671_LR", "0.005"))
CHUNK_SIZE = int(os.environ.get("MODEL6_LAIECO671_CHUNK", "64"))
ALPHA = float(os.environ.get("MODEL6_LAIECO671_ALPHA", "0.25"))
TARGET_MEDIAN_NSE = float(os.environ.get("MODEL6_LAIECO671_TARGET_NSE", "0.732"))
EVAL_EVERY = int(os.environ.get("MODEL6_LAIECO671_EVAL_EVERY", "1"))
LOSS_MODE = os.environ.get("MODEL6_LAIECO671_LOSS", "rmsecomb").strip().lower()
LOSS_WEIGHT_NSE = float(os.environ.get("MODEL6_LAIECO671_LOSS_WEIGHT_NSE", "0.5"))
LOSS_NSE_EPS = float(os.environ.get("MODEL6_LAIECO671_LOSS_NSE_EPS", "0.1"))
MODEL_VARIANT = os.environ.get("MODEL6_LAIECO671_MODEL", "closed").strip().lower()
RUN_DIR_NAME = os.environ.get("MODEL6_LAIECO671_RUN_DIR", "Model6Closed_Snow_aSrz_SIMHYD_Simple_LAIEco_full671_target0732")
RESUME_CKPT = os.environ.get("MODEL6_LAIECO671_RESUME_CKPT", "").strip()
EVAL_ONLY = os.environ.get("MODEL6_LAIECO671_EVAL_ONLY", "0") == "1"
BASIN_SET = os.environ.get("MODEL6_LAIECO671_BASIN_SET", "all").strip().lower()
PROTOTYPE_PER_REGIME = int(os.environ.get("MODEL6_LAIECO671_PROTOTYPE_PER_REGIME", "8"))
REGIME_BALANCED = os.environ.get("MODEL6_LAIECO671_REGIME_BALANCED", "1") == "1"
VAL_YEARS = int(os.environ.get("MODEL6_LAIECO671_VAL_YEARS", "3"))
BASIN_LIST_FILE = os.environ.get("MODEL6_LAIECO671_BASIN_LIST", "").strip()

PROJECT_DIR = REPO_ROOT / "model6" / "results" / "train_runs"
RUN_DIR = PROJECT_DIR / RUN_DIR_NAME
PLOTS_DIR = RUN_DIR / "plots"
FULL671_BASELINE_DIR = DATA_ROOT / "ECO_HYBRID" / "Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep80_iter200"
FULL671_BASELINE_CKPT = FULL671_BASELINE_DIR / "model_Ep80_state.pt"
SOFT_BEST_CKPT = DATA_ROOT / "Model_six_physical" / "Model6PhysicalFix_B_soft_gate_continue_Ep50" / "model_Ep40_state.pt"
DYNK_BETA_DEMO32_CKPT = DATA_ROOT / "ECO_HYBRID" / "Model6C_dynamicK_staticBetaGW_demo32" / "model_Ep20_state.pt"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def build_new_model():
    if MODEL_VARIANT == "closed":
        return simple_demo.rnn.MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleLAIEco(
            ninv=4 + len(simple_demo.ATTR_LST),
            nmul=simple_demo.NMUL,
            nattr=len(simple_demo.ATTR_LST),
            hiddeninv=simple_demo.HIDDEN_SIZE,
            inittime=simple_demo.BUFFTIME,
            routOpt=True,
            comprout=False,
            compwts=True,
            lgdyn=True,
            lgdynweight=0.6,
            dynamic_sq=True,
            dynamic_etgam=True,
            dynamic_partition=True,
            dynamic_cfmax_snow=True,
            dynamic_routing_scale=False,
            dynamic_all=False,
            reg_amp_w=1e-3,
            reg_smooth_w=1e-3,
            reg_part_w=1e-3,
            component_routing=True,
        )
    if MODEL_VARIANT == "dynamick_staticbeta":
        return simple_demo.rnn.MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleDynamicKStaticBetaGW(
            ninv=4 + len(simple_demo.ATTR_LST),
            nmul=simple_demo.NMUL,
            nattr=len(simple_demo.ATTR_LST),
            hiddeninv=simple_demo.HIDDEN_SIZE,
            inittime=simple_demo.BUFFTIME,
            routOpt=True,
            comprout=False,
            compwts=True,
            lgdyn=True,
            lgdynweight=0.6,
            dynamic_sq=True,
            dynamic_etgam=True,
            dynamic_partition=True,
            dynamic_cfmax_snow=True,
            dynamic_routing_scale=False,
            dynamic_all=False,
            reg_amp_w=1e-3,
            reg_smooth_w=1e-3,
            reg_part_w=1e-3,
            reg_k_smooth_w=1e-4,
            reg_beta_w=1e-5,
            component_routing=True,
            dynamic_k=True,
        )
    if MODEL_VARIANT == "regime_moe":
        return simple_demo.rnn.MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleLAIRegimeMoE(
            ninv=4 + len(simple_demo.ATTR_LST),
            nmul=simple_demo.NMUL,
            nattr=len(simple_demo.ATTR_LST),
            hiddeninv=simple_demo.HIDDEN_SIZE,
            inittime=simple_demo.BUFFTIME,
            routOpt=True,
            comprout=False,
            compwts=True,
            lgdyn=True,
            lgdynweight=0.6,
            dynamic_sq=True,
            dynamic_etgam=True,
            dynamic_partition=True,
            dynamic_cfmax_snow=True,
            dynamic_routing_scale=False,
            dynamic_all=False,
            reg_amp_w=1e-3,
            reg_smooth_w=1e-3,
            reg_part_w=1e-3,
            gate_entropy_w=1e-3,
            gate_smooth_w=1e-3,
            component_routing=False,
        )
    raise ValueError(f"Unsupported MODEL6_LAIECO671_MODEL={MODEL_VARIANT}")


def model_label():
    if MODEL_VARIANT == "closed":
        return "Model6Closed_Snow_aSrz_SIMHYD_Simple_LAIEco_full671"
    if MODEL_VARIANT == "dynamick_staticbeta":
        return "Model6C_dynamicK_staticBetaGW_LAIEco_full671"
    if MODEL_VARIANT == "regime_moe":
        return "Model6Closed_Snow_aSrz_SIMHYD_Simple_LAIRegimeMoE_full671"
    return f"ModelVariant_{MODEL_VARIANT}"


def default_warm_checkpoint():
    if MODEL_VARIANT == "closed":
        return FULL671_BASELINE_CKPT
    if MODEL_VARIANT == "dynamick_staticbeta" and DYNK_BETA_DEMO32_CKPT.exists():
        return DYNK_BETA_DEMO32_CKPT
    return FULL671_BASELINE_CKPT


def build_old_model():
    return simple_demo.rnn.MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimple(
        ninv=4 + len(simple_demo.ATTR_LST),
        nmul=simple_demo.NMUL,
        nattr=len(simple_demo.ATTR_LST),
        hiddeninv=simple_demo.HIDDEN_SIZE,
        inittime=simple_demo.BUFFTIME,
        routOpt=True,
        comprout=False,
        compwts=True,
        lgdyn=True,
        lgdynweight=0.6,
        dynamic_sq=True,
        dynamic_etgam=True,
        dynamic_partition=True,
        dynamic_cfmax_snow=True,
        dynamic_routing_scale=False,
        dynamic_all=False,
        reg_amp_w=1e-3,
        reg_smooth_w=1e-3,
        reg_part_w=1e-3,
        component_routing=True,
    )


def build_soft_gate_model():
    if asrz_demo is None:
        raise RuntimeError("Soft-gate comparison model is unavailable in the clean publication package.")
    return asrz_demo.build_soft_gate_model()


def partial_load(source_state, target_model):
    tgt = target_model.state_dict()
    loaded = []
    skipped = []
    for key, value in source_state.items():
        if key in tgt and tuple(tgt[key].shape) == tuple(value.shape):
            tgt[key] = value.clone()
            loaded.append(key)
        else:
            skipped.append(key)
    target_model.load_state_dict(tgt, strict=False)
    return {"loaded": loaded, "skipped": skipped}


def prepare_data_with_lai():
    prepared = prepare_full671_base()
    attrs_raw = simple_demo.camels.DataframeCamels(
        tRange=simple_demo.T_INV,
        subset=prepared["basin_ids"],
        forType=simple_demo.FORCING,
    ).getDataConst(varLst=simple_demo.ATTR_LST, doNorm=False, rmNan=False).astype(np.float32)
    train_dates = pd.to_datetime(prepared["t_train_array"].astype(str))
    test_dates = pd.to_datetime(prepared["t_test_array"].astype(str))
    lai_train, lai_train_info = build_lai_daily_for_basins(prepared["basin_ids"], train_dates, attrs_raw, LAI_GAPFILLED_FILE)
    lai_test, lai_test_info = build_lai_daily_for_basins(prepared["basin_ids"], test_dates, attrs_raw, LAI_GAPFILLED_FILE)

    prepared["attrs_raw"] = attrs_raw
    prepared["attr_index"] = {name: i for i, name in enumerate(simple_demo.ATTR_LST)}
    prepared["x_train_old"] = prepared["x_train"].copy()
    prepared["x_eval_old"] = prepared["x_eval"].copy()
    prepared["x_train"] = np.concatenate([prepared["x_train"], lai_train], axis=2).astype(np.float32)
    x_hist_old = prepared["x_eval_old"][:, :len(prepared["t_train_array"]), :]
    x_test_old = prepared["x_eval_old"][:, len(prepared["t_train_array"]):, :]
    x_hist_new = np.concatenate([x_hist_old, lai_train], axis=2).astype(np.float32)
    x_test_new = np.concatenate([x_test_old, lai_test], axis=2).astype(np.float32)
    prepared["x_eval"] = np.concatenate([x_hist_new, x_test_new], axis=1).astype(np.float32)
    prepared["lai_train_info"] = lai_train_info
    prepared["lai_test_info"] = lai_test_info
    prepared["x_train"][np.isnan(prepared["x_train"])] = 0.0
    prepared["x_eval"][np.isnan(prepared["x_eval"])] = 0.0
    return prepared


def regime_labels_from_aridity(aridity):
    bins = [-np.inf, 0.5, 1.0, 2.0, np.inf]
    labels = np.array(["humid", "dry_subhumid", "semiarid", "arid"], dtype=object)
    idx = np.digitize(aridity, bins[1:-1], right=True)
    return labels[idx]


def subset_prepared(prepared, basin_indices):
    basin_indices = np.asarray(basin_indices, dtype=int)
    sub = {}
    for key, val in prepared.items():
        if isinstance(val, np.ndarray) and val.shape[0] == len(prepared["basin_ids"]):
            sub[key] = val[basin_indices].copy()
        elif isinstance(val, pd.DataFrame) and len(val) == len(prepared["basin_ids"]):
            sub[key] = val.iloc[basin_indices].reset_index(drop=True).copy()
        elif isinstance(val, list) and len(val) == len(prepared["basin_ids"]):
            sub[key] = [val[i] for i in basin_indices]
        else:
            sub[key] = val
    return sub


def select_regime_prototype_indices(prepared, per_regime):
    aridity = prepared["attrs_raw"][:, prepared["attr_index"]["aridity"]]
    labels = regime_labels_from_aridity(aridity)
    chosen = []
    for regime in ["humid", "dry_subhumid", "semiarid", "arid"]:
        idx = np.where(labels == regime)[0]
        if len(idx) == 0:
            continue
        idx = idx[np.argsort(aridity[idx])]
        take = min(per_regime, len(idx))
        if take == len(idx):
            picked = idx
        else:
            loc = np.linspace(0, len(idx) - 1, take, dtype=int)
            picked = idx[loc]
        chosen.extend(picked.tolist())
    return np.array(sorted(chosen), dtype=int)


def apply_basin_subset(prepared):
    if BASIN_LIST_FILE:
        basin_path = Path(BASIN_LIST_FILE)
        basin_df = pd.read_csv(basin_path) if basin_path.suffix.lower() == ".csv" else None
        if basin_df is not None:
            if "gauge_id" in basin_df.columns:
                basin_ids = [int(str(x).split("_")[-1]) for x in basin_df["gauge_id"].tolist()]
            elif "basin_id" in basin_df.columns:
                basin_ids = [int(x) for x in basin_df["basin_id"].tolist()]
            else:
                basin_ids = [int(x) for x in basin_df.iloc[:, 0].tolist()]
        else:
            basin_ids = [int(line.strip()) for line in basin_path.read_text().splitlines() if line.strip()]
        basin_lookup = {int(b): i for i, b in enumerate(prepared["basin_ids"])}
        basin_indices = np.array([basin_lookup[b] for b in basin_ids if b in basin_lookup], dtype=int)
        subset = subset_prepared(prepared, basin_indices)
        return subset, {
            "basin_set": "custom_file",
            "basin_list_file": str(basin_path),
            "prototype_basin_ids": [int(x) for x in subset["basin_ids"]],
        }
    if BASIN_SET != "prototype32":
        return prepared, {"basin_set": BASIN_SET, "prototype_basin_ids": []}
    basin_indices = select_regime_prototype_indices(prepared, PROTOTYPE_PER_REGIME)
    subset = subset_prepared(prepared, basin_indices)
    prototype_aridity = subset["attrs_raw"][:, subset["attr_index"]["aridity"]]
    regime_manifest = regime_labels_from_aridity(prototype_aridity).tolist()
    subset["prototype_regime"] = regime_manifest
    return subset, {
        "basin_set": BASIN_SET,
        "prototype_basin_ids": [int(subset["basin_ids"][i]) for i in range(len(subset["basin_ids"]))],
        "prototype_regimes": regime_manifest,
    }


def build_validation_bundle(prepared):
    val_days = max(365, VAL_YEARS * 365)
    n_train = prepared["x_train"].shape[1]
    if n_train <= simple_demo.BUFFTIME + val_days:
        return None
    val_start = n_train - val_days
    hist_start = max(0, val_start - simple_demo.BUFFTIME)
    x_eval = prepared["x_train"][:, hist_start:, :].astype(np.float32)
    z_base = prepared["z_train"][:, hist_start:, :].astype(np.float32)
    attr_rep = np.repeat(prepared["attr_norm"][:, None, :], z_base.shape[1], axis=1).astype(np.float32)
    z_eval = np.concatenate([z_base, attr_rep], axis=2).astype(np.float32)
    return {
        "x_eval": x_eval,
        "z_eval": z_eval,
        "obs": prepared["y_train"][:, val_start:, 0].astype(np.float32),
        "inittime": val_start - hist_start,
        "name": f"train_last_{VAL_YEARS}y",
    }


def regime_summary(metrics_df):
    df = metrics_df.copy()
    df["regime"] = regime_labels_from_aridity(df["aridity_index"].to_numpy(dtype=np.float32))
    rows = []
    for regime, sub in df.groupby("regime"):
        rows.append({
            "regime": regime,
            "count": int(len(sub)),
            "median_NSE": float(sub["NSE"].median()),
            "mean_NSE": float(sub["NSE"].mean()),
            "median_KGE": float(sub["KGE"].median()),
            "test_nse_gt_0_count": int((sub["NSE"] > 0).sum()),
            "test_nse_gt_0p5_count": int((sub["NSE"] > 0.5).sum()),
            "test_nse_gt_0p7_count": int((sub["NSE"] > 0.7).sum()),
        })
    return pd.DataFrame(rows)


def evaluate_model_components(model, prepared, use_old_x=False, x_eval_override=None, z_eval_override=None, obs_override=None, inittime_override=None):
    n_basin = len(prepared["basin_ids"])
    x_eval = x_eval_override if x_eval_override is not None else (prepared["x_eval_old"] if use_old_x else prepared["x_eval"])
    z_eval = z_eval_override if z_eval_override is not None else prepared["z_eval"]
    obs_eval = obs_override if obs_override is not None else prepared["obs_test"]
    test_len = obs_eval.shape[1]
    old_inittime = model.inittime
    old_training = model.training
    model.inittime = len(prepared["t_train_array"]) if inittime_override is None else int(inittime_override)
    model.train(mode=False)

    pred = np.zeros((n_basin, test_len), dtype=np.float32)
    mix_store = {}
    comp_store = {}
    weights = np.zeros((n_basin, simple_demo.NMUL), dtype=np.float32)
    for i0 in range(0, n_basin, CHUNK_SIZE):
        i1 = min(i0 + CHUNK_SIZE, n_basin)
        x_part = torch.from_numpy(np.swapaxes(x_eval[i0:i1], 1, 0)).float()
        z_part = torch.from_numpy(np.swapaxes(z_eval[i0:i1], 1, 0)).float()
        x_part, z_part = to_device(x_part, z_part)
        with torch.no_grad():
            q_part, diag_part = model(x_part, z_part, return_diagnostics=True, return_component_diagnostics=True)
        pred[i0:i1] = q_part.detach().cpu().numpy()[:, :, 0].T
        for key, val in diag_part.items():
            if not torch.is_tensor(val):
                continue
            arr = val.detach().cpu().numpy()
            if key == "component_weights":
                weights[i0:i1] = arr
            elif key.endswith("_components") and arr.ndim in (3, 4):
                if arr.ndim == 4 and arr.shape[-1] == 1:
                    arr = arr[..., 0]
                if key not in comp_store:
                    comp_store[key] = np.zeros((n_basin, test_len, arr.shape[2]), dtype=np.float32)
                comp_store[key][i0:i1] = np.swapaxes(arr, 0, 1)
            elif arr.ndim == 3 and arr.shape[0] == test_len:
                if key not in mix_store:
                    mix_store[key] = np.zeros((n_basin, test_len), dtype=np.float32)
                mix_store[key][i0:i1] = np.swapaxes(arr[:, :, 0], 0, 1)

    model.inittime = old_inittime
    model.train(mode=old_training)
    return {"pred": pred, "mix": mix_store, "comp": comp_store, "weights": weights, "obs": obs_eval}


def compute_tables(model_name, prepared, eval_res, is_closed_simple):
    obs = eval_res.get("obs", prepared["obs_test"])
    pred = eval_res["pred"]
    mix = eval_res["mix"]
    comp = eval_res["comp"]
    weights = eval_res["weights"]
    meta = prepared["meta"].copy()
    metrics_rows = []
    wb_rows = []

    for i, basin_id in enumerate(prepared["basin_ids"]):
        o = obs[i]
        s = pred[i]
        w = weights[i]
        p_comp = comp["precipitation_components"][i]
        if is_closed_simple and "process_local_residual_components" in comp:
            residual_comp = comp["process_local_residual_components"][i]
            weighted_residual = np.sum(residual_comp * w[None, :], axis=1)
        else:
            soil = mix["SMS"][i]
            q_for_resid = mix["q_after_gate"][i]
            losses = mix["groundwater_loss"][i] + mix["channel_loss"][i] + mix["gate_loss"][i]
            total_store = mix["SNOWPACK"][i] + mix["MELTWATER"][i] + soil + mix["GW"][i]
            prev_store = np.concatenate([total_store[:1], total_store[:-1]])
            weighted_residual = mix["precipitation"][i] - mix["interception_evaporation"][i] - mix["actual_ET"][i] - losses - q_for_resid - (total_store - prev_store)
        cum_p = float(np.nansum(p_comp[:, 0]))
        q_process = mix["Q_process"][i] if is_closed_simple else mix["q_after_gate"][i]
        intv = mix["interception_evaporation"][i]
        et = mix["actual_ET"][i]
        srun = mix["surface_runoff"][i]
        iflow = mix["interflow"][i]
        bas = mix["baseflow"][i]
        rec = mix["recharge_to_groundwater"][i]
        snow = mix["SNOWPACK"][i]
        melt = mix["MELTWATER"][i]
        gw = mix["GW"][i]
        if is_closed_simple:
            soil_store = mix["Sa"][i]
            alpha = mix["alpha"][i]
            theta_cap = mix["theta_cap"][i]
            asrz = soil_store - np.nanmin(soil_store)
            mean_asrz = float(np.nanmean(asrz))
            asrz_cap = float(np.nanmax(asrz))
            alpha_mean = float(np.nanmean(alpha))
            theta_cap_mean = float(np.nanmean(theta_cap))
            external_loss_over_p = 0.0
        else:
            soil_store = mix["SMS"][i]
            alpha_mean = np.nan
            theta_cap_mean = np.nan
            mean_asrz = np.nan
            asrz_cap = np.nan
            losses = mix["groundwater_loss"][i] + mix["channel_loss"][i] + mix["gate_loss"][i]
            external_loss_over_p = safe_ratio(float(np.nansum(losses)), cum_p)

        metrics_rows.append({
            "model": model_name,
            "basin_id": basin_id,
            "lat": float(meta.loc[i, "lat"]),
            "lon": float(meta.loc[i, "lon"]),
            "NSE": calc_nse(o, s),
            "KGE": calc_kge(o, s),
            "R2": calc_r2(o, s),
            "FLV": calc_flv(o, s),
            "FHV": calc_fhv(o, s),
            "low_flow_NSE": lowflow_nse(o, s),
            "high_flow_NSE": highflow_nse(o, s),
            "ET_over_P": safe_ratio(float(np.nansum(et + intv)), cum_p),
            "INT_over_P": safe_ratio(float(np.nansum(intv)), cum_p),
            "Q_over_P": safe_ratio(float(np.nansum(q_process)), cum_p),
            "SRUN_over_P": safe_ratio(float(np.nansum(srun)), cum_p),
            "IFLOW_over_P": safe_ratio(float(np.nansum(iflow)), cum_p),
            "BAS_over_P": safe_ratio(float(np.nansum(bas)), cum_p),
            "REC_over_P": safe_ratio(float(np.nansum(rec)), cum_p),
            "external_loss_over_P": external_loss_over_p,
            "mean_abs_daily_wb_residual_mm_day": float(np.nanmean(np.abs(weighted_residual))),
            "cumulative_relative_wb_error": safe_ratio(float(np.nansum(np.abs(weighted_residual))), cum_p),
            "SNOWPACK_drift_mm": float(snow[-1]),
            "MELTWATER_drift_mm": float(melt[-1]),
            "Sa_drift_mm": float(soil_store[-1]) if is_closed_simple else np.nan,
            "GW_drift_mm": float(gw[-1]),
            "alpha_mean": alpha_mean,
            "theta_cap_mean": theta_cap_mean,
            "aSrz_capacity_mm": asrz_cap,
            "mean_aSrz_mm": mean_asrz,
            "aridity_index": float(prepared["attrs_raw"][i, prepared["attr_index"]["aridity"]]),
        })
        wb_rows.append({
            "model": model_name,
            "basin_id": basin_id,
            "cumulative_precipitation_mm": cum_p,
            "mean_abs_daily_wb_residual_mm_day": float(np.nanmean(np.abs(weighted_residual))),
            "max_abs_daily_wb_residual_mm_day": float(np.nanmax(np.abs(weighted_residual))),
            "cumulative_relative_wb_error": safe_ratio(float(np.nansum(np.abs(weighted_residual))), cum_p),
            "external_loss_over_P": external_loss_over_p,
        })
    return pd.DataFrame(metrics_rows), pd.DataFrame(wb_rows)


def summarize_model(model_name, metrics_df):
    return {
        "model": model_name,
        "number_of_basins": int(len(metrics_df)),
        "median_NSE": float(metrics_df["NSE"].median()),
        "mean_NSE": float(metrics_df["NSE"].mean()),
        "median_KGE": float(metrics_df["KGE"].median()),
        "median_R2": float(metrics_df["R2"].median()),
        "median_FLV": float(metrics_df["FLV"].median()),
        "median_FHV": float(metrics_df["FHV"].median()),
        "median_low_flow_NSE": float(metrics_df["low_flow_NSE"].median()),
        "median_high_flow_NSE": float(metrics_df["high_flow_NSE"].median()),
        "median_ET_over_P": float(metrics_df["ET_over_P"].median()),
        "median_Q_over_P": float(metrics_df["Q_over_P"].median()),
        "median_alpha_mean": float(metrics_df["alpha_mean"].median()) if "alpha_mean" in metrics_df else np.nan,
        "median_aSrz_capacity_mm": float(metrics_df["aSrz_capacity_mm"].median()) if "aSrz_capacity_mm" in metrics_df else np.nan,
        "median_mean_aSrz_mm": float(metrics_df["mean_aSrz_mm"].median()) if "mean_aSrz_mm" in metrics_df else np.nan,
        "median_weighted_process_closure_residual_mm_day": float(metrics_df["mean_abs_daily_wb_residual_mm_day"].median()),
        "mean_weighted_process_closure_residual_mm_day": float(metrics_df["mean_abs_daily_wb_residual_mm_day"].mean()),
        "median_weighted_cumulative_relative_wb_error": float(metrics_df["cumulative_relative_wb_error"].median()),
        "mean_weighted_cumulative_relative_wb_error": float(metrics_df["cumulative_relative_wb_error"].mean()),
        "basins_gt_1pct_wb_error": int((metrics_df["cumulative_relative_wb_error"] > 0.01).sum()),
        "total_external_losses_frac_of_p": float(metrics_df["external_loss_over_P"].mean()),
        "test_nse_gt_0_count": int((metrics_df["NSE"] > 0).sum()),
        "test_nse_gt_0p5_count": int((metrics_df["NSE"] > 0.5).sum()),
        "test_nse_gt_0p7_count": int((metrics_df["NSE"] > 0.7).sum()),
        "test_nse_lt_0_count": int((metrics_df["NSE"] < 0).sum()),
    }


def compute_regime_sampling_probs(prepared):
    aridity = prepared["attrs_raw"][:, prepared["attr_index"]["aridity"]]
    labels = regime_labels_from_aridity(aridity)
    weight_map = {"humid": 1.0, "dry_subhumid": 1.25, "semiarid": 2.0, "arid": 2.5}
    weights = np.asarray([weight_map[str(label)] for label in labels], dtype=np.float64)
    weights /= weights.sum()
    return weights


def train_model(model, prepared):
    if torch.cuda.is_available():
        model = model.cuda(GPU_ID)
    rmse_loss = crit.RmseLossComb(alpha=ALPHA)
    if torch.cuda.is_available():
        rmse_loss = rmse_loss.cuda()
    basin_std = np.nanstd(prepared["y_train"][:, :, 0], axis=1).astype(np.float32)
    basin_std[~np.isfinite(basin_std)] = 1.0
    basin_std[basin_std < 1e-6] = 1.0
    nse_loss = crit.NSELossBatch(basin_std, eps=LOSS_NSE_EPS) if LOSS_MODE in {"nsebatch", "mix"} else None
    sampling_probs = compute_regime_sampling_probs(prepared) if REGIME_BALANCED else None
    val_bundle = build_validation_bundle(prepared)
    optim = torch.optim.Adadelta(model.parameters(), lr=LR)
    model.zero_grad()
    run_csv = RUN_DIR / "run.csv"
    mode = "a" if START_EPOCH > 1 and run_csv.exists() else "w"
    best_state_path = RUN_DIR / "model_best_state.pt"
    best_metrics_path = RUN_DIR / "best_metrics.json"
    best_median = -np.inf
    if best_metrics_path.exists():
        best_median = json.loads(best_metrics_path.read_text()).get("selection_metric", -np.inf)

    with open(run_csv, mode, encoding="utf-8") as rf:
        for epoch in range(START_EPOCH, EPOCHS + 1):
            model.train(True)
            loss_ep = 0.0
            t0 = time.time()
            for _ in range(MAX_ITER_EP):
                if sampling_probs is None:
                    i_grid, i_t = train.randomIndex(len(prepared["basin_ids"]), prepared["x_train"].shape[1], [BATCH_SIZE, RHO], bufftime=simple_demo.BUFFTIME)
                else:
                    i_grid = np.random.choice(len(prepared["basin_ids"]), size=BATCH_SIZE, replace=True, p=sampling_probs)
                    i_t = np.random.randint(simple_demo.BUFFTIME, prepared["x_train"].shape[1] - RHO, size=BATCH_SIZE)
                x_batch = train.selectSubset(prepared["x_train"], i_grid, i_t, RHO, bufftime=simple_demo.BUFFTIME)
                y_batch = train.selectSubset(prepared["y_train"], i_grid, i_t, RHO)
                z_batch = train.selectSubset(prepared["z_train"], i_grid, i_t, RHO, c=prepared["attr_norm"], bufftime=simple_demo.BUFFTIME)
                y_p = model(x_batch, z_batch)
                if LOSS_MODE == "rmsecomb":
                    loss = rmse_loss(y_p, y_batch)
                elif LOSS_MODE == "nsebatch":
                    loss = nse_loss(y_p, y_batch, i_grid)
                elif LOSS_MODE == "mix":
                    loss = (1.0 - LOSS_WEIGHT_NSE) * rmse_loss(y_p, y_batch) + LOSS_WEIGHT_NSE * nse_loss(y_p, y_batch, i_grid)
                else:
                    raise ValueError(f"Unsupported MODEL6_LAIECO671_LOSS={LOSS_MODE}")
                if hasattr(model, "get_auxiliary_loss"):
                    aux = model.get_auxiliary_loss()
                    if aux is not None:
                        loss = loss + aux
                loss.backward()
                optim.step()
                model.zero_grad()
                loss_ep += float(loss.item())
            loss_ep /= MAX_ITER_EP
            msg = f"Epoch {epoch} Loss {loss_ep:.4f} time {time.time() - t0:.2f}"
            print(msg, flush=True)
            rf.write(msg + "\n")
            rf.flush()
            ckpt = RUN_DIR / f"model_Ep{epoch}_state.pt"
            torch.save(model.state_dict(), ckpt)
            if epoch % EVAL_EVERY != 0:
                continue
            val_res = evaluate_model_components(
                model,
                prepared,
                use_old_x=False,
                x_eval_override=None if val_bundle is None else val_bundle["x_eval"],
                z_eval_override=None if val_bundle is None else val_bundle["z_eval"],
                obs_override=None if val_bundle is None else val_bundle["obs"],
                inittime_override=None if val_bundle is None else val_bundle["inittime"],
            ) if val_bundle is not None else None
            if val_res is not None:
                val_metrics_df, _ = compute_tables(model_label(), prepared, val_res, is_closed_simple=True)
                val_summary = summarize_model(f"{model_label()}_{val_bundle['name']}", val_metrics_df)
                val_regime = regime_summary(val_metrics_df)
                dry_mask = val_regime["regime"].isin(["semiarid", "arid"])
                dry_metric = float(val_regime.loc[dry_mask, "median_NSE"].mean()) if dry_mask.any() else float(val_summary["median_NSE"])
                selection_metric = float(val_summary["median_NSE"] + 0.15 * dry_metric)
            else:
                val_summary = None
                selection_metric = -np.inf

            eval_res = evaluate_model_components(model, prepared, use_old_x=False)
            metrics_df, wb_df = compute_tables(model_label(), prepared, eval_res, is_closed_simple=True)
            summary = summarize_model(model_label(), metrics_df)
            test_regime_df = regime_summary(metrics_df)
            if val_summary is None:
                selection_metric = float(summary["median_NSE"])
            epoch_metrics = {"epoch": epoch, "selection_metric": selection_metric, "validation": val_summary, "test": summary}
            (RUN_DIR / f"epoch_{epoch:03d}_metrics.json").write_text(json.dumps(epoch_metrics, indent=2) + "\n", encoding="utf-8")
            if selection_metric > best_median:
                best_median = selection_metric
                torch.save(model.state_dict(), best_state_path)
                best_metrics_path.write_text(json.dumps(epoch_metrics, indent=2) + "\n", encoding="utf-8")
                metrics_df.to_csv(RUN_DIR / "per_basin_metrics_best_so_far.csv", index=False)
                wb_df.to_csv(RUN_DIR / "water_balance_best_so_far.csv", index=False)
                test_regime_df.to_csv(RUN_DIR / "per_regime_metrics_best_so_far.csv", index=False)
            val_msg = ""
            if val_summary is not None:
                val_msg = f", val median NSE {val_summary['median_NSE']:.6f}, select {selection_metric:.6f}"
            status = f"Eval epoch {epoch}: test median NSE {summary['median_NSE']:.6f}, mean NSE {summary['mean_NSE']:.6f}{val_msg}, best {best_median:.6f}"
            print(status, flush=True)
            rf.write(status + "\n")
            rf.flush()
            if summary["median_NSE"] >= TARGET_MEDIAN_NSE:
                stop_msg = f"Target median NSE {TARGET_MEDIAN_NSE:.3f} reached at epoch {epoch}"
                print(stop_msg, flush=True)
                rf.write(stop_msg + "\n")
                rf.flush()
                break
    return model


def make_spatial_plots(df, model_name, out_prefix):
    order = ["<0", "0-0.2", "0.2-0.4", "0.4-0.5", "0.5-0.65", "0.65-0.8", ">=0.8"]
    palette = {
        "<0": "#b71c1c",
        "0-0.2": "#ef6c00",
        "0.2-0.4": "#fdd835",
        "0.4-0.5": "#9ccc65",
        "0.5-0.65": "#26a69a",
        "0.65-0.8": "#42a5f5",
        ">=0.8": "#283593",
        "NaN": "#9e9e9e",
    }
    plot_df = df.copy()
    plot_df["nse_bin"] = plot_df["NSE"].apply(
        lambda val: "NaN" if not np.isfinite(val) else "<0" if val < 0 else "0-0.2" if val < 0.2 else "0.2-0.4" if val < 0.4 else "0.4-0.5" if val < 0.5 else "0.5-0.65" if val < 0.65 else "0.65-0.8" if val < 0.8 else ">=0.8"
    )
    plt.figure(figsize=(10.8, 5.8))
    for lab in order:
        sub = plot_df[plot_df["nse_bin"] == lab]
        if len(sub) == 0:
            continue
        plt.scatter(sub["lon"], sub["lat"], s=18, c=palette[lab], label=lab, edgecolors="black", linewidths=0.20)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"{model_name} spatial NSE bins")
    plt.legend(ncol=4, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{out_prefix}_spatial_nse_bins.png", dpi=220)
    plt.close()


def save_final_outputs(prepared, baseline_metrics, baseline_wb, soft_metrics, soft_wb, new_metrics, new_wb, final_manifest):
    summary_rows = []
    metric_frames = []
    wb_frames = []
    if soft_metrics is not None and soft_wb is not None:
        summary_rows.append(summarize_model("Model6PhysicalFix_B_soft_gate", soft_metrics))
        metric_frames.append(soft_metrics)
        wb_frames.append(soft_wb)
    if baseline_metrics is not None and baseline_wb is not None:
        summary_rows.append(summarize_model("Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep80_baseline", baseline_metrics))
        metric_frames.append(baseline_metrics)
        wb_frames.append(baseline_wb)
    summary_rows.append(summarize_model(model_label(), new_metrics))
    metric_frames.append(new_metrics)
    wb_frames.append(new_wb)
    summary_df = pd.DataFrame(summary_rows)
    pd.concat(metric_frames, ignore_index=True).to_csv(RUN_DIR / "per_basin_metrics_compare.csv", index=False)
    pd.concat(wb_frames, ignore_index=True).to_csv(RUN_DIR / "water_balance_compare.csv", index=False)
    summary_df.to_csv(RUN_DIR / "summary_compare.csv", index=False)
    regime_summary(new_metrics).to_csv(RUN_DIR / "per_regime_metrics_compare.csv", index=False)
    make_spatial_plots(new_metrics, model_label(), "laieco_full671")
    (RUN_DIR / "manifest.json").write_text(json.dumps(final_manifest, indent=2) + "\n", encoding="utf-8")
    lines = [
        "Conservative LAI-aware closed Model 6 full671 run",
        f"Variant: {MODEL_VARIANT}",
        f"Warm checkpoint: {final_manifest['warm_checkpoint']}",
        f"Target median NSE: {TARGET_MEDIAN_NSE:.3f}",
        f"Train epochs requested: {START_EPOCH}-{EPOCHS}",
        f"Batch size: {BATCH_SIZE}",
        f"Rho: {RHO}",
        f"Max iter per epoch: {MAX_ITER_EP}",
        "",
    ]
    for _, row in summary_df.iterrows():
        lines.append(str(row["model"]))
        lines.append(f"- median NSE: {row['median_NSE']:.6f}")
        lines.append(f"- mean NSE: {row['mean_NSE']:.6f}")
        lines.append(f"- median KGE: {row['median_KGE']:.6f}")
        lines.append(f"- median R2: {row['median_R2']:.6f}")
        lines.append(f"- median ET/P: {row['median_ET_over_P']:.6f}")
        lines.append(f"- median Q/P: {row['median_Q_over_P']:.6f}")
        lines.append(f"- NSE > 0.7 basins: {int(row.get('test_nse_gt_0p7_count', 0))}")
        lines.append(f"- NSE < 0 basins: {int(row.get('test_nse_lt_0_count', 0))}")
        lines.append("")
    (RUN_DIR / "final_result.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ensure_dir(RUN_DIR)
    ensure_dir(PLOTS_DIR)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.set_device(GPU_ID)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(8)

    prepared = prepare_data_with_lai()
    prepared, subset_manifest = apply_basin_subset(prepared)
    prepared["meta"].to_csv(RUN_DIR / "basin_metadata.csv", index=False)
    if subset_manifest.get("prototype_basin_ids"):
        (RUN_DIR / "prototype_basin_ids.txt").write_text(
            "\n".join(str(x) for x in subset_manifest["prototype_basin_ids"]) + "\n",
            encoding="utf-8",
        )

    baseline_metrics = None
    baseline_wb = None
    if FULL671_BASELINE_CKPT.exists():
        baseline_model = build_old_model()
        baseline_state = torch.load(FULL671_BASELINE_CKPT, map_location="cpu")
        baseline_model.load_state_dict(baseline_state, strict=False)
        if torch.cuda.is_available():
            baseline_model = baseline_model.cuda(GPU_ID)
        baseline_eval = evaluate_model_components(baseline_model, prepared, use_old_x=True)
        baseline_metrics, baseline_wb = compute_tables("Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep80_baseline", prepared, baseline_eval, is_closed_simple=True)

    soft_metrics = None
    soft_wb = None
    if asrz_demo is not None and SOFT_BEST_CKPT.exists():
        soft_model = build_soft_gate_model()
        soft_state = torch.load(SOFT_BEST_CKPT, map_location="cpu")
        soft_model.load_state_dict(soft_state, strict=False)
        if torch.cuda.is_available():
            soft_model = soft_model.cuda(GPU_ID)
        soft_eval = evaluate_model_components(soft_model, prepared, use_old_x=True)
        soft_metrics, soft_wb = compute_tables("Model6PhysicalFix_B_soft_gate", prepared, soft_eval, is_closed_simple=False)

    new_model = build_new_model()
    warm_ckpt = Path(os.environ.get("MODEL6_LAIECO671_WARM_CKPT", str(default_warm_checkpoint())))
    if RESUME_CKPT:
        resume_state = torch.load(RESUME_CKPT, map_location="cpu")
        new_model.load_state_dict(resume_state, strict=False)
        load_report = {"mode": "resume", "source": RESUME_CKPT}
    else:
        warm_state = torch.load(warm_ckpt, map_location="cpu")
        if MODEL_VARIANT == "regime_moe":
            load_report = {
                "mode": "warm_start",
                "source": str(warm_ckpt),
                "humid_expert": partial_load(warm_state, new_model.humid_expert),
                "dry_pulse_expert": partial_load(warm_state, new_model.dry_pulse_expert),
                "slow_gw_expert": partial_load(warm_state, new_model.slow_gw_expert),
            }
        else:
            load_report = partial_load(warm_state, new_model)
            load_report.update({"mode": "warm_start", "source": str(warm_ckpt)})
    (RUN_DIR / "load_report.json").write_text(json.dumps(load_report, indent=2) + "\n", encoding="utf-8")

    if EVAL_ONLY:
        if torch.cuda.is_available():
            new_model = new_model.cuda(GPU_ID)
    else:
        new_model = train_model(new_model, prepared)

    best_state_path = RUN_DIR / "model_best_state.pt"
    if best_state_path.exists():
        best_state = torch.load(best_state_path, map_location="cpu")
        new_model.load_state_dict(best_state, strict=False)
    if torch.cuda.is_available():
        new_model = new_model.cuda(GPU_ID)
    new_eval = evaluate_model_components(new_model, prepared, use_old_x=False)
    new_metrics, new_wb = compute_tables(model_label(), prepared, new_eval, is_closed_simple=True)

    best_metrics = {}
    best_metrics_path = RUN_DIR / "best_metrics.json"
    if best_metrics_path.exists():
        best_metrics = json.loads(best_metrics_path.read_text())
    final_manifest = {
        "variant": MODEL_VARIANT,
        "warm_checkpoint": str(warm_ckpt),
        "soft_gate_checkpoint": str(SOFT_BEST_CKPT),
        "lai_gapfilled_file": str(LAI_GAPFILLED_FILE),
        "lai_train_info": prepared["lai_train_info"],
        "lai_test_info": prepared["lai_test_info"],
        "load_report": load_report,
        "target_median_nse": TARGET_MEDIAN_NSE,
        "basin_set": BASIN_SET,
        "prototype_per_regime": PROTOTYPE_PER_REGIME,
        "regime_balanced_sampling": REGIME_BALANCED,
        "validation_years": VAL_YEARS,
        "subset_manifest": subset_manifest,
        "best_metrics": best_metrics,
    }
    save_final_outputs(prepared, baseline_metrics, baseline_wb, soft_metrics, soft_wb, new_metrics, new_wb, final_manifest)
    print(pd.read_csv(RUN_DIR / "summary_compare.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
