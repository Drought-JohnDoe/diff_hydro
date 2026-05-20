import json
import random
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code" / "dPLHBVrelease" / "hydroDL-dev"))

from Diagnosis import calc_fhv, calc_flv, calc_kge, calc_nse, calc_r2, highflow_nse, lowflow_nse, safe_ratio  # noqa: E402
from hydroDL import utils  # noqa: E402
from hydroDL.data import camels  # noqa: E402
from hydroDL.model import crit, rnn, train  # noqa: E402


T_TRAIN = [19801001, 19951001]
T_INV = [19801001, 19951001]
T_TEST = [19951001, 20101001]
FORCING = "daymet"
SEED = 111111
BUFFTIME = 365
RHO = 365
EPOCHS = 10
BATCH_SIZE = 8
MAX_ITER_EP = 5
HIDDEN_SIZE = 64
NMUL = 4
GPU_ID = 1 if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 0
LR = 0.25
CHUNK_SIZE = 16

ATTR_LST = [
    "p_mean", "pet_mean", "p_seasonality", "frac_snow", "aridity", "high_prec_freq", "high_prec_dur",
    "low_prec_freq", "low_prec_dur", "elev_mean", "slope_mean", "area_gages2", "frac_forest", "lai_max",
    "lai_diff", "gvf_max", "gvf_diff", "dom_land_cover_frac", "dom_land_cover", "root_depth_50",
    "soil_depth_pelletier", "soil_depth_statsgo", "soil_porosity", "soil_conductivity",
    "max_water_content", "sand_frac", "silt_frac", "clay_frac", "geol_1st_class", "glim_1st_class_frac",
    "geol_2nd_class", "glim_2nd_class_frac", "carbonate_rocks_frac", "geol_porostiy", "geol_permeability",
]
SNOW_FRAC_IDX = ATTR_LST.index("frac_snow")
VAR_F = ["prcp", "tmean"]

PROJECT_DIR = ROOT / "Model_six_physical"
DEMO_DIR = PROJECT_DIR / "Model6_HBVDeepGWHybrid_demo32"
PLOTS_DIR = DEMO_DIR / "plots"
SOFT_GATE_DIR = PROJECT_DIR / "variants" / "Model6PhysicalFix_B_soft_gate"
SOFT_GATE_PER_BASIN = PROJECT_DIR / "model6_physicalfix_per_basin_metrics.csv"
SOFT_GATE_WB = PROJECT_DIR / "model6_physicalfix_water_balance_summary.csv"
HBV_PER_BASIN = PROJECT_DIR / "hbv_epoch10_vs_model6_soft_gate_by_basin.csv"
HBV_WB = PROJECT_DIR / "hbv_epoch10_per_basin_water_balance.csv"
BASELINE_RUN = ROOT / "outputs" / "rnnStreamflow" / "CAMELSMODELSIX" / "DynamicSimHydModelSix" / "AllBasins" / FORCING / str(SEED) / (
    "T_19801001_19951001_BS_32_HS_64_RHO_365_Buff_365_Mul_4_Route_1_CmpW_1_LGDyn_1_DSQ_1_DETGAM_1_DPART_1_DCFMAX_1_DROUTE_0_CRoute_1_DryCh_1_ZGate_1_MaxIter200_All671_BS32_HS64_MaxIter200"
)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def seasonal_features(t_range):
    t_arr = utils.time.tRange2Array(t_range)
    dates = pd.to_datetime(t_arr.astype(str))
    doy = dates.dayofyear.to_numpy(dtype=np.float32)
    ang = 2.0 * np.pi * (doy - 1.0) / 365.0
    return np.stack([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)


def legacy_state_to_current(state):
    mapped = {}
    key_map = {
        "lstmdyn.lstm.w_ih": "lstmdyn.lstm.weight_ih_l0",
        "lstmdyn.lstm.w_hh": "lstmdyn.lstm.weight_hh_l0",
        "lstmdyn.lstm.b_ih": "lstmdyn.lstm.bias_ih_l0",
        "lstmdyn.lstm.b_hh": "lstmdyn.lstm.bias_hh_l0",
    }
    for key, value in state.items():
        mapped[key_map.get(key, key)] = value
    return mapped


def logit_from_value(value, lo, hi):
    frac = (value - lo) / (hi - lo)
    frac = min(max(frac, 1e-4), 1.0 - 1e-4)
    return float(np.log(frac / (1.0 - frac)))


def partial_load_from_soft_gate(target_model, soft_state, nmul, old_nfea=13, new_nfea=19):
    tgt = target_model.state_dict()
    src = soft_state if isinstance(soft_state, dict) else soft_state.state_dict()
    loaded, skipped = [], []
    new_param_biases = {
        13: logit_from_value(0.02, 0.0005, 0.15),   # K_slow
        14: logit_from_value(2.0, 0.0, 10.0),       # PERC_cap
        15: logit_from_value(0.01, 0.0, 0.05),      # K_deep_return
        16: logit_from_value(0.001, 0.0, 0.01),     # K_deep_leak
        17: logit_from_value(0.20, 0.0, 2.0),       # CAP_t
        18: logit_from_value(0.10, 0.0, 0.60),      # K_channel
    }
    for k, v in src.items():
        if k == "staticOut.weight" and "staticOut.weight" in tgt:
            tw = tgt["staticOut.weight"].clone()
            old_rows = old_nfea * nmul
            tw[:old_rows, :] = v[:old_rows, :]
            tw[old_rows:new_nfea * nmul, :] = 0.0
            tgt["staticOut.weight"] = tw
            loaded.append("staticOut.weight[:old_rows]")
            continue
        if k == "staticOut.bias" and "staticOut.bias" in tgt:
            tb = tgt["staticOut.bias"].clone()
            old_rows = old_nfea * nmul
            tb[:old_rows] = v[:old_rows]
            for param_idx, bias_val in new_param_biases.items():
                start = param_idx * nmul
                end = start + nmul
                tb[start:end] = bias_val
            tgt["staticOut.bias"] = tb
            loaded.append("staticOut.bias[:old_rows]+hybrid_init")
            continue
        if k == "compStaticBias" and "compStaticBias" in tgt:
            tb = tgt["compStaticBias"].clone()
            tb[:, :old_nfea, :] = v[:, :old_nfea, :]
            tb[:, old_nfea:new_nfea, :] = 0.0
            tgt["compStaticBias"] = tb
            loaded.append("compStaticBias[:old_nfea]")
            continue
        if k in tgt and tgt[k].shape == v.shape:
            tgt[k] = v.clone()
            loaded.append(k)
        else:
            skipped.append(k)
    target_model.load_state_dict(tgt, strict=False)
    return loaded, skipped


def load_pet_full(root_db, gageid, forcing):
    var_lst_nl = ["PEVAP"]
    t_pet_range = [19800101, 20150101] if forcing != "maurer" else [19800101, 20090101]
    t_pet_lst = utils.time.tRange2Array(t_pet_range)
    pet_dir = str(root_db) + "/pet_harg/" + forcing + "/"
    ntime = len(t_pet_lst)
    pet_full = np.empty([len(gageid), ntime, len(var_lst_nl)], dtype=np.float32)
    for k, gid in enumerate(gageid):
        pet_full[k, :, :] = camels.readcsvGage(pet_dir, gid, var_lst_nl, ntime)
    return pet_full, t_pet_lst


def prepare_data():
    root_db = ROOT / "Camels"
    with open(BASELINE_RUN / "statDict.json", "r") as fp:
        stat_dict = json.load(fp)

    camels.initcamels(str(root_db))
    gageinfo = camels.gageDict
    basin_ids_all = gageinfo["id"].tolist()
    rng = np.random.RandomState(SEED)
    subset_ids = sorted([int(x) for x in rng.choice(basin_ids_all, size=32, replace=False)])
    subset_idx = [basin_ids_all.index(gid) for gid in subset_ids]
    areas = gageinfo["area"][subset_idx]

    df_train = camels.DataframeCamels(tRange=T_TRAIN, subset=subset_ids, forType=FORCING)
    forc_un = df_train.getDataTs(varLst=VAR_F, doNorm=False, rmNan=False).astype(np.float32)
    obs_un = df_train.getDataObs(doNorm=False, rmNan=False, basinnorm=False).astype(np.float32)
    temp_area = np.tile(areas[:, None, None], (1, obs_un.shape[1], 1))
    obs_un = (obs_un * 0.0283168 * 3600 * 24) / (temp_area * 1e6) * 1e3

    df_inv = camels.DataframeCamels(tRange=T_INV, subset=subset_ids, forType=FORCING)
    forc_inv_un = df_inv.getDataTs(varLst=VAR_F, doNorm=False, rmNan=False).astype(np.float32)
    attrs_un = df_inv.getDataConst(varLst=ATTR_LST, doNorm=False, rmNan=False).astype(np.float32)

    pet_full, t_pet_lst = load_pet_full(root_db, gageinfo["id"], FORCING)
    t_train_lst = utils.time.tRange2Array(T_TRAIN)
    t_inv_lst = utils.time.tRange2Array(T_INV)
    _, _, ind2 = np.intersect1d(t_train_lst, t_pet_lst, return_indices=True)
    _, _, ind2inv = np.intersect1d(t_inv_lst, t_pet_lst, return_indices=True)
    pet_un = pet_full[:, ind2, :][subset_idx, :, :]
    pet_inv_un = pet_full[:, ind2inv, :][subset_idx, :, :]

    season_train = np.tile(seasonal_features(T_TRAIN)[None, :, :], (len(subset_ids), 1, 1))
    series_inv = np.concatenate([forc_inv_un, pet_inv_un], axis=2)
    attr_norm = camels.transNormbyDic(attrs_un, ATTR_LST, stat_dict, toNorm=True).astype(np.float32)
    attr_norm[np.isnan(attr_norm)] = 0.0
    series_norm = camels.transNormbyDic(series_inv, VAR_F + ["pet"], stat_dict, toNorm=True).astype(np.float32)
    series_norm[np.isnan(series_norm)] = 0.0
    snow_frac_raw = attrs_un[:, SNOW_FRAC_IDX:SNOW_FRAC_IDX + 1].astype(np.float32)
    snow_frac_ts = np.repeat(snow_frac_raw[:, None, :], series_norm.shape[1], axis=1)
    z_train = np.concatenate([series_norm, snow_frac_ts], axis=2)
    x_train = np.concatenate([forc_un, pet_un, season_train], axis=2).astype(np.float32)
    x_train[np.isnan(x_train)] = 0.0

    df_test = camels.DataframeCamels(tRange=T_TEST, subset=subset_ids, forType=FORCING)
    forc_test = df_test.getDataTs(varLst=VAR_F, doNorm=False, rmNan=False).astype(np.float32)
    obs_test = df_test.getDataObs(doNorm=False, rmNan=False, basinnorm=False).astype(np.float32)
    temp_area_test = np.tile(areas[:, None, None], (1, obs_test.shape[1], 1))
    obs_test = (obs_test * 0.0283168 * 3600 * 24) / (temp_area_test * 1e6) * 1e3
    obs_test = obs_test[:, :, 0].astype(np.float32)

    _, _, ind2test = np.intersect1d(utils.time.tRange2Array(T_TEST), t_pet_lst, return_indices=True)
    pet_test = pet_full[:, ind2test, :][subset_idx, :, :]
    season_hist = np.tile(seasonal_features(T_TRAIN)[None, :, :], (len(subset_ids), 1, 1))
    season_test = np.tile(seasonal_features(T_TEST)[None, :, :], (len(subset_ids), 1, 1))
    x_hist = np.concatenate([forc_un, pet_un, season_hist], axis=2).astype(np.float32)
    x_hist[np.isnan(x_hist)] = 0.0
    x_test = np.concatenate([forc_test, pet_test, season_test], axis=2).astype(np.float32)
    x_test[np.isnan(x_test)] = 0.0
    x_eval = np.concatenate([x_hist, x_test], axis=1)

    series_test = np.concatenate([forc_test, pet_test], axis=2)
    series_eval = np.concatenate([series_inv, series_test], axis=1)
    series_norm_eval = camels.transNormbyDic(series_eval, VAR_F + ["pet"], stat_dict, toNorm=True).astype(np.float32)
    series_norm_eval[np.isnan(series_norm_eval)] = 0.0
    snow_frac_ts_eval = np.repeat(snow_frac_raw[:, None, :], series_norm_eval.shape[1], axis=1)
    c_temp = np.repeat(attr_norm[:, None, :], series_norm_eval.shape[1], axis=1)
    z_eval = np.concatenate([series_norm_eval, snow_frac_ts_eval, c_temp], axis=2).astype(np.float32)

    meta = pd.DataFrame({
        "basin_id": subset_ids,
        "lat": gageinfo["lat"][subset_idx],
        "lon": gageinfo["lon"][subset_idx],
        "area_km2": areas,
    })
    return {
        "subset_ids": subset_ids,
        "subset_idx": subset_idx,
        "meta": meta,
        "x_train": x_train,
        "y_train": obs_un.astype(np.float32),
        "z_train": z_train.astype(np.float32),
        "attr_norm": attr_norm.astype(np.float32),
        "x_eval": x_eval.astype(np.float32),
        "z_eval": z_eval.astype(np.float32),
        "obs_test": obs_test.astype(np.float32),
        "x_test_only": x_test.astype(np.float32),
    }


def build_model():
    ninv = 4 + len(ATTR_LST)
    model = rnn.MultiInv_DynamicSimHydModelSix_HBVDeepGWHybrid(
        ninv=ninv,
        nmul=NMUL,
        nattr=len(ATTR_LST),
        hiddeninv=HIDDEN_SIZE,
        inittime=BUFFTIME,
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
        dry_channel_loss=True,
        zero_flow_gate=True,
        channel_loss_max=0.60,
        gate_variant="soft",
        gate_strength_max=0.30,
    )
    return model


def build_soft_gate_model():
    ninv = 4 + len(ATTR_LST)
    return rnn.MultiInv_DynamicSimHydModelSix_Physical(
        ninv=ninv,
        nmul=NMUL,
        nattr=len(ATTR_LST),
        hiddeninv=HIDDEN_SIZE,
        inittime=BUFFTIME,
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
        dry_channel_loss=True,
        zero_flow_gate=True,
        channel_loss_max=0.60,
        gate_variant="soft",
        gate_strength_max=0.30,
    )


def to_device(*args):
    if torch.cuda.is_available():
        return [x.cuda(GPU_ID) for x in args]
    return list(args)


def evaluate_model(model, prepared):
    n_basin = len(prepared["subset_ids"])
    x_eval = prepared["x_eval"]
    z_eval = prepared["z_eval"]
    test_len = prepared["obs_test"].shape[1]
    old_inittime = model.inittime
    old_training = model.training
    model.inittime = len(utils.time.tRange2Array(T_TRAIN))
    model.train(mode=False)

    pred = np.zeros((n_basin, test_len), dtype=np.float32)
    diag_store = None
    comp_store = None
    for i0 in range(0, n_basin, CHUNK_SIZE):
        i1 = min(i0 + CHUNK_SIZE, n_basin)
        x_part = torch.from_numpy(np.swapaxes(x_eval[i0:i1], 1, 0)).float()
        z_part = torch.from_numpy(np.swapaxes(z_eval[i0:i1], 1, 0)).float()
        x_part, z_part = to_device(x_part, z_part)
        with torch.no_grad():
            q_part, diag_part = model(x_part, z_part, return_diagnostics=True, return_component_diagnostics=True)
        pred[i0:i1] = q_part.detach().cpu().numpy()[:, :, 0].T
        if diag_store is None:
            diag_store = {}
            for key, val in diag_part.items():
                if key in {"component_weights", "route_a_components", "route_b_components"}:
                    continue
                if val.ndim == 3:
                    diag_store[key] = np.zeros((n_basin, test_len), dtype=np.float32)
            comp_store = {
                "component_weights": np.zeros((n_basin, NMUL), dtype=np.float32),
                "route_a_components": np.zeros((n_basin, NMUL), dtype=np.float32),
                "route_b_components": np.zeros((n_basin, NMUL), dtype=np.float32),
            }
        for key in list(diag_store.keys()):
            diag_store[key][i0:i1] = diag_part[key].detach().cpu().numpy()[:, :, 0].T
        comp_store["component_weights"][i0:i1] = diag_part["component_weights"].detach().cpu().numpy()
        comp_store["route_a_components"][i0:i1] = diag_part["route_a_components"].detach().cpu().numpy()
        comp_store["route_b_components"][i0:i1] = diag_part["route_b_components"].detach().cpu().numpy()

    model.inittime = old_inittime
    model.train(mode=old_training)
    return {"pred": pred, "diag": diag_store, "comp": comp_store}


def evaluate_soft_gate_model(model, prepared):
    n_basin = len(prepared["subset_ids"])
    x_eval = prepared["x_eval"]
    z_eval = prepared["z_eval"]
    test_len = prepared["obs_test"].shape[1]
    old_inittime = model.inittime
    old_training = model.training
    model.inittime = len(utils.time.tRange2Array(T_TRAIN))
    model.train(mode=False)

    pred = np.zeros((n_basin, test_len), dtype=np.float32)
    diag_store = None
    for i0 in range(0, n_basin, CHUNK_SIZE):
        i1 = min(i0 + CHUNK_SIZE, n_basin)
        x_part = torch.from_numpy(np.swapaxes(x_eval[i0:i1], 1, 0)).float()
        z_part = torch.from_numpy(np.swapaxes(z_eval[i0:i1], 1, 0)).float()
        x_part, z_part = to_device(x_part, z_part)
        with torch.no_grad():
            q_part, diag_part = model(x_part, z_part, return_diagnostics=True)
        pred[i0:i1] = q_part.detach().cpu().numpy()[:, :, 0].T
        if diag_store is None:
            diag_store = {k: np.zeros((n_basin, test_len), dtype=np.float32) for k, v in diag_part.items() if v.ndim == 3}
        for key in diag_store:
            diag_store[key][i0:i1] = diag_part[key].detach().cpu().numpy()[:, :, 0].T
    model.inittime = old_inittime
    model.train(mode=old_training)
    return {"pred": pred, "diag": diag_store}


def compute_hybrid_tables(prepared, eval_res):
    obs = prepared["obs_test"]
    pred = eval_res["pred"]
    diag = eval_res["diag"]
    meta = prepared["meta"].copy()

    metrics_rows, wb_rows = [], []
    for i, basin_id in enumerate(prepared["subset_ids"]):
        o = obs[i]
        s = pred[i]
        p = diag["precipitation"][i]
        et_total = diag["actual_ET"][i] + diag["interception_evaporation"][i]
        d_storage = diag["SMS"][i] + diag["UZ"][i] + diag["LZ"][i] + diag["DEEP_GW"][i] + diag["CHANNEL_STORE"][i] + diag["SNOWPACK"][i] + diag["MELTWATER"][i]
        prev = np.concatenate([d_storage[:1], d_storage[:-1]])
        delta_storage = d_storage - prev
        residual = p - diag["Q_process"][i] - diag["interception_evaporation"][i] - diag["actual_ET"][i] - diag["true_deep_leak"][i] - diag["channel_true_loss"][i] - delta_storage
        cum_p = float(np.nansum(p))
        metrics_rows.append({
            "model": "Model6_HBVDeepGWHybrid",
            "basin_id": basin_id,
            "NSE": calc_nse(o, s),
            "KGE": calc_kge(o, s),
            "R2": calc_r2(o, s),
            "FLV": calc_flv(o, s),
            "FHV": calc_fhv(o, s),
            "low_flow_NSE": lowflow_nse(o, s),
            "high_flow_NSE": highflow_nse(o, s),
            "ET_P_ratio": safe_ratio(float(np.nansum(et_total)), cum_p),
            "sim_obs_q_ratio": safe_ratio(float(np.nanmean(s)), float(np.nanmean(o))),
            "mean_true_deep_leak": float(np.nanmean(diag["true_deep_leak"][i])),
            "mean_channel_true_loss": float(np.nanmean(diag["channel_true_loss"][i])),
            "mean_capillary_rise": float(np.nanmean(diag["capillary_rise"][i])),
            "mean_deep_return": float(np.nanmean(diag["deep_return"][i])),
        })
        wb_rows.append({
            "model": "Model6_HBVDeepGWHybrid",
            "basin_id": basin_id,
            "mean_abs_water_balance_error_mm_day": float(np.nanmean(np.abs(residual))),
            "cumulative_abs_water_balance_error_mm": float(np.nansum(np.abs(residual))),
            "cumulative_precipitation_mm": cum_p,
            "relative_error": safe_ratio(float(np.nansum(np.abs(residual))), cum_p),
        })
    return pd.DataFrame(metrics_rows), pd.DataFrame(wb_rows)


def compute_soft_gate_tables(prepared, eval_res):
    obs = prepared["obs_test"]
    pred = eval_res["pred"]
    diag = eval_res["diag"]
    metrics_rows, wb_rows = [], []
    for i, basin_id in enumerate(prepared["subset_ids"]):
        o = obs[i]
        s = pred[i]
        p = diag["precipitation"][i]
        et_total = diag["actual_ET"][i] + diag["interception_evaporation"][i]
        storage = diag["SMS"][i] + diag["GW"][i] + diag["SNOWPACK"][i] + diag["MELTWATER"][i]
        prev = np.concatenate([storage[:1], storage[:-1]])
        delta_storage = storage - prev
        residual = p - diag["interception_evaporation"][i] - diag["actual_ET"][i] - diag["groundwater_loss_capped"][i] - diag["channel_loss"][i] - diag["gate_loss"][i] - diag["q_after_gate"][i] - delta_storage
        cum_p = float(np.nansum(p))
        metrics_rows.append({
            "model": "Model6PhysicalFix_B_soft_gate",
            "basin_id": basin_id,
            "NSE": calc_nse(o, s),
            "KGE": calc_kge(o, s),
            "R2": calc_r2(o, s),
            "FLV": calc_flv(o, s),
            "FHV": calc_fhv(o, s),
            "low_flow_NSE": lowflow_nse(o, s),
            "high_flow_NSE": highflow_nse(o, s),
            "ET_P_ratio": safe_ratio(float(np.nansum(et_total)), cum_p),
            "sim_obs_q_ratio": safe_ratio(float(np.nanmean(s)), float(np.nanmean(o))),
        })
        wb_rows.append({
            "model": "Model6PhysicalFix_B_soft_gate",
            "basin_id": basin_id,
            "mean_abs_water_balance_error_mm_day": float(np.nanmean(np.abs(residual))),
            "cumulative_abs_water_balance_error_mm": float(np.nansum(np.abs(residual))),
            "cumulative_precipitation_mm": cum_p,
            "relative_error": safe_ratio(float(np.nansum(np.abs(residual))), cum_p),
        })
    return pd.DataFrame(metrics_rows), pd.DataFrame(wb_rows)


def summarize_model(model_name, metrics_df, wb_df):
    return {
        "model": model_name,
        "median_NSE": float(metrics_df["NSE"].median()),
        "median_KGE": float(metrics_df["KGE"].median()),
        "median_R2": float(metrics_df["R2"].median()),
        "median_FLV": float(metrics_df["FLV"].median()),
        "median_FHV": float(metrics_df["FHV"].median()),
        "median_low_flow_NSE": float(metrics_df["low_flow_NSE"].median()),
        "median_high_flow_NSE": float(metrics_df["high_flow_NSE"].median()),
        "median_mean_abs_residual_mm_day": float(wb_df["mean_abs_water_balance_error_mm_day"].median()),
        "median_relative_error": float(wb_df["relative_error"].median()),
        "basins_gt_1pct_relative_error": int((wb_df["relative_error"] > 0.01).sum()),
        "median_ET_P_ratio": float(metrics_df["ET_P_ratio"].median()),
        "median_sim_obs_q_ratio": float(metrics_df["sim_obs_q_ratio"].median()),
    }


def make_plots(summary_df, comparison_df):
    ensure_dir(PLOTS_DIR)
    plt.figure(figsize=(8, 4.5))
    plt.bar(summary_df["model"], summary_df["median_NSE"], color=["#777777", "#1e88e5", "#c62828"])
    plt.ylabel("Median NSE")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "median_nse_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.bar(summary_df["model"], summary_df["median_relative_error"], color=["#777777", "#1e88e5", "#c62828"])
    plt.ylabel("Median cumulative relative WB error")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "median_water_balance_error_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(comparison_df["soft_gate_NSE"], comparison_df["hybrid_NSE"], s=20, alpha=0.7)
    lo = min(float(comparison_df["soft_gate_NSE"].min()), float(comparison_df["hybrid_NSE"].min()))
    hi = max(float(comparison_df["soft_gate_NSE"].max()), float(comparison_df["hybrid_NSE"].max()))
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)
    plt.xlabel("Soft-gate NSE")
    plt.ylabel("Hybrid NSE")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "soft_gate_vs_hybrid_nse_scatter.png", dpi=200)
    plt.close()


def sync_to_github():
    gh_root = Path("/home/mircore/Desktop/diff_hydro_github")
    target = gh_root / "final_model" / "model6_physical"
    ensure_dir(target)
    (target / "scripts").mkdir(parents=True, exist_ok=True)
    (target / "results").mkdir(parents=True, exist_ok=True)
    (target / "checkpoints" / "soft_gate").mkdir(parents=True, exist_ok=True)
    (target / "checkpoints" / "hybrid_demo32").mkdir(parents=True, exist_ok=True)

    shutil.copy2(ROOT / "code" / "dPLHBVrelease" / "hydroDL-dev" / "hydroDL" / "model" / "rnn.py", target / "scripts" / "rnn.py")
    shutil.copy2(PROJECT_DIR / "run_model6_physical_fix.py", target / "scripts" / "run_model6_physical_fix.py")
    shutil.copy2(DEMO_DIR / "run_model6_hbvdeepgw_hybrid_demo32.py", target / "scripts" / "run_model6_hbvdeepgw_hybrid_demo32.py")

    for fn in [
        "final_result.txt",
        "model6_physicalfix_training_log.csv",
        "model6_physicalfix_metrics_by_epoch.csv",
        "model6_physicalfix_per_basin_metrics.csv",
        "model6_physicalfix_water_balance_summary.csv",
        "model6_physicalfix_flux_state_violations.csv",
        "model6_physicalfix_parameter_bounds.csv",
        "model6_physicalfix_component_diagnostics.csv",
        "model6_physicalfix_worst_basins.csv",
        "hbv_epoch10_physical_audit_summary.csv",
        "hbv_epoch10_vs_model6_soft_gate_summary.csv",
        "hbv_epoch10_vs_model6_soft_gate_final.txt",
    ]:
        src = PROJECT_DIR / fn
        if src.exists():
            shutil.copy2(src, target / "results" / fn)

    for fn in [
        "hybrid_demo32_basin_ids.txt",
        "hybrid_demo32_summary.csv",
        "hybrid_demo32_per_basin_metrics.csv",
        "hybrid_demo32_water_balance_summary.csv",
        "hybrid_demo32_compare_three_models.csv",
        "final_result.txt",
        "run.csv",
        "load_report.json",
    ]:
        src = DEMO_DIR / fn
        if src.exists():
            shutil.copy2(src, target / "results" / f"hybrid_demo32_{fn}" if fn == "final_result.txt" else fn)

    if (PROJECT_DIR / "plots").exists():
        dst = target / "results" / "plots_model6_physical"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(PROJECT_DIR / "plots", dst)
    if PLOTS_DIR.exists():
        dst = target / "results" / "plots_hybrid_demo32"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(PLOTS_DIR, dst)

    for ckpt in sorted((SOFT_GATE_DIR).glob("model_Ep*_state.pt")):
        shutil.copy2(ckpt, target / "checkpoints" / "soft_gate" / ckpt.name)
    for ckpt in sorted(DEMO_DIR.glob("model_Ep*.pt")):
        shutil.copy2(ckpt, target / "checkpoints" / "hybrid_demo32" / ckpt.name)

    readme = target / "README.md"
    readme.write_text(
        "# Model 6 physical branch\n\n"
        "This folder contains the latest Model 6 soft-gate physical-fix code/results and the 32-basin "
        "`Model6_HBVDeepGWHybrid` demo run.\n\n"
        "Included:\n"
        "- latest `rnn.py` with `MultiInv_DynamicSimHydModelSix_Physical` and `MultiInv_DynamicSimHydModelSix_HBVDeepGWHybrid`\n"
        "- soft-gate checkpoints and CSV metrics\n"
        "- 32-basin hybrid demo checkpoints, metrics, plots, and comparison summaries\n\n"
        "Excluded:\n"
        "- oversized raw diagnostic `.npz` files (>100 MB each), to keep the normal GitHub push valid without Git LFS\n"
    )

    from subprocess import run
    run(["git", "-C", str(gh_root), "add", "final_model/model6_physical"], check=True)
    status = run(["git", "-C", str(gh_root), "status", "--short"], check=True, capture_output=True, text=True).stdout.strip()
    if status:
        run(["git", "-C", str(gh_root), "commit", "-m", "Add Model 6 HBVDeepGW hybrid demo and latest soft-gate artifacts"], check=True)
        run(["git", "-C", str(gh_root), "push", "origin", "HEAD"], check=True)


def main():
    ensure_dir(DEMO_DIR)
    ensure_dir(PLOTS_DIR)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.set_device(GPU_ID)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    prepared = prepare_data()
    (DEMO_DIR / "hybrid_demo32_basin_ids.txt").write_text("\n".join(str(x) for x in prepared["subset_ids"]) + "\n")

    soft_gate_state = torch.load(SOFT_GATE_DIR / "model_Ep10_state.pt", map_location="cpu")
    model = build_model()
    load_report = {}
    final_ckpt = DEMO_DIR / f"model_Ep{EPOCHS}.pt"
    if final_ckpt.exists():
        state = torch.load(final_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        load_report["resumed_from_existing_checkpoint"] = True
    else:
        load_report["resumed_from_existing_checkpoint"] = False
        soft_gate_state = legacy_state_to_current(soft_gate_state)
        loaded, skipped = partial_load_from_soft_gate(model, soft_gate_state, nmul=NMUL)
        load_report["loaded"] = loaded
        load_report["skipped"] = skipped
        if torch.cuda.is_available():
            model = model.cuda(GPU_ID)
        loss_fun = crit.RmseLossComb(alpha=0.25)
        if torch.cuda.is_available():
            loss_fun = loss_fun.cuda()
        optim = torch.optim.Adadelta(model.parameters(), lr=LR)
        model.zero_grad()
        with open(DEMO_DIR / "run.csv", "w") as rf:
            for i_epoch in range(1, EPOCHS + 1):
                model.train(True)
                loss_ep = 0.0
                t0 = time.time()
                for _ in range(MAX_ITER_EP):
                    i_grid, i_t = train.randomIndex(len(prepared["subset_ids"]), prepared["x_train"].shape[1], [BATCH_SIZE, RHO], bufftime=BUFFTIME)
                    x_batch = train.selectSubset(prepared["x_train"], i_grid, i_t, RHO, bufftime=BUFFTIME)
                    y_batch = train.selectSubset(prepared["y_train"], i_grid, i_t, RHO)
                    z_batch = train.selectSubset(prepared["z_train"], i_grid, i_t, RHO, c=prepared["attr_norm"], bufftime=BUFFTIME)
                    y_p = model(x_batch, z_batch)
                    loss = loss_fun(y_p, y_batch)
                    if hasattr(model, "get_auxiliary_loss"):
                        aux = model.get_auxiliary_loss()
                        if aux is not None:
                            loss = loss + aux
                    loss.backward()
                    optim.step()
                    model.zero_grad()
                    loss_ep += float(loss.item())
                loss_ep /= MAX_ITER_EP
                log_str = f"Epoch {i_epoch} Loss {loss_ep:.4f} time {time.time() - t0:.2f}"
                print(log_str, flush=True)
                rf.write(log_str + "\n")
                rf.flush()
                torch.save(model.state_dict(), DEMO_DIR / f"model_Ep{i_epoch}.pt")
    (DEMO_DIR / "load_report.json").write_text(json.dumps(load_report, indent=2))

    if torch.cuda.is_available():
        model = model.cuda(GPU_ID)
    eval_res = evaluate_model(model, prepared)
    hybrid_metrics, hybrid_wb = compute_hybrid_tables(prepared, eval_res)
    hybrid_metrics.to_csv(DEMO_DIR / "hybrid_demo32_per_basin_metrics.csv", index=False)
    hybrid_wb.to_csv(DEMO_DIR / "hybrid_demo32_water_balance_summary.csv", index=False)

    subset_ids = prepared["subset_ids"]
    soft_gate_model = build_soft_gate_model()
    soft_gate_model.load_state_dict(soft_gate_state, strict=False)
    if torch.cuda.is_available():
        soft_gate_model = soft_gate_model.cuda(GPU_ID)
    soft_eval = evaluate_soft_gate_model(soft_gate_model, prepared)
    soft_metrics, soft_wb_subset = compute_soft_gate_tables(prepared, soft_eval)
    soft_metrics.to_csv(DEMO_DIR / "soft_gate_demo32_per_basin_metrics.csv", index=False)
    soft_wb_subset.to_csv(DEMO_DIR / "soft_gate_demo32_water_balance_summary.csv", index=False)

    hbv_by_basin = pd.read_csv(HBV_PER_BASIN)
    hbv_metrics = hbv_by_basin[hbv_by_basin["basin_id"].isin(subset_ids)].copy()
    hbv_wb = pd.read_csv(HBV_WB)
    hbv_wb = hbv_wb[hbv_wb["basin_id"].isin(subset_ids)].copy()

    summary_rows = []
    summary_rows.append(summarize_model("HBV_Epoch10", hbv_metrics, hbv_wb))
    summary_rows.append(summarize_model("Model6PhysicalFix_B_soft_gate", soft_metrics, soft_wb_subset))
    summary_rows.append(summarize_model("Model6_HBVDeepGWHybrid", hybrid_metrics, hybrid_wb))
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(DEMO_DIR / "hybrid_demo32_summary.csv", index=False)
    summary_df.to_csv(DEMO_DIR / "hybrid_demo32_compare_three_models.csv", index=False)

    comparison_df = soft_metrics[["basin_id", "NSE"]].rename(columns={"NSE": "soft_gate_NSE"}).merge(
        hybrid_metrics[["basin_id", "NSE"]].rename(columns={"NSE": "hybrid_NSE"}),
        on="basin_id",
        how="inner",
    )
    make_plots(summary_df, comparison_df)

    lines = [
        "Model6_HBVDeepGWHybrid demo-32 final result",
        f"Subset size: {len(subset_ids)}",
        f"Basins: {subset_ids}",
        "",
    ]
    for _, row in summary_df.iterrows():
        lines.append(str(row["model"]))
        lines.append(f"- median NSE: {row['median_NSE']:.4f}")
        lines.append(f"- median KGE: {row['median_KGE']:.4f}")
        lines.append(f"- median R2: {row['median_R2']:.4f}")
        lines.append(f"- median FLV: {row['median_FLV']:.4f}")
        lines.append(f"- median FHV: {row['median_FHV']:.4f}")
        lines.append(f"- median low-flow NSE: {row['median_low_flow_NSE']:.4f}")
        lines.append(f"- median high-flow NSE: {row['median_high_flow_NSE']:.4f}")
        lines.append(f"- median WB residual: {row['median_mean_abs_residual_mm_day']:.6f} mm/day")
        lines.append(f"- median relative WB error: {row['median_relative_error']:.6f}")
        lines.append(f"- basins >1% relative WB error: {int(row['basins_gt_1pct_relative_error'])}")
        lines.append("")
    best_mass = summary_df.sort_values(["median_relative_error", "basins_gt_1pct_relative_error", "median_NSE"], ascending=[True, True, False]).iloc[0]["model"]
    best_nse = summary_df.sort_values("median_NSE", ascending=False).iloc[0]["model"]
    lines.append(f"Best by physical validity first: {best_mass}")
    lines.append(f"Best by median NSE: {best_nse}")
    (DEMO_DIR / "final_result.txt").write_text("\n".join(lines) + "\n")

    shutil.copy2(Path(__file__), DEMO_DIR / "run_model6_hbvdeepgw_hybrid_demo32.py")
    sync_to_github()

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
