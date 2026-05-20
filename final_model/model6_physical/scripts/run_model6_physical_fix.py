import json
import math
import os
import sys
import time
from collections import OrderedDict
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

from Diagnosis import calc_fhv, calc_flv, calc_kge, calc_nse, calc_r2, highflow_nse, lowflow_nse  # noqa: E402
from hydroDL import utils  # noqa: E402
from hydroDL.data import camels  # noqa: E402
from hydroDL.model import crit, rnn, train  # noqa: E402


T_TRAIN = [19801001, 19951001]
T_INV = [19801001, 19951001]
T_TEST = [19951001, 20101001]
BUFFTIME = 365
FORCING = "daymet"
SEED = 111111
GPU_ID = int(os.environ.get("MODEL6_PHYSICAL_GPU_ID", "1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "0"))
EPOCHS = int(os.environ.get("MODEL6_PHYSICAL_EPOCHS", "10"))
BATCH_SIZE = int(os.environ.get("MODEL6_PHYSICAL_BATCH_SIZE", "32"))
RHO = int(os.environ.get("MODEL6_PHYSICAL_RHO", "365"))
MAX_ITER_EP = int(os.environ.get("MODEL6_PHYSICAL_MAX_ITER", "100"))
HIDDEN_SIZE = int(os.environ.get("MODEL6_PHYSICAL_HIDDEN_SIZE", "64"))
NMUL = int(os.environ.get("MODEL6_PHYSICAL_NMUL", "4"))
LR = float(os.environ.get("MODEL6_PHYSICAL_LR", "0.25"))
ALPHA = float(os.environ.get("MODEL6_PHYSICAL_ALPHA", "0.25"))
CHUNK_SIZE = int(os.environ.get("MODEL6_PHYSICAL_CHUNK", "32"))
QUICK_EVAL_EPOCHS = {
    int(x) for x in os.environ.get("MODEL6_PHYSICAL_EVAL_EPOCHS", "5,10").split(",") if x.strip()
}
NEG_TOL = -1e-6
PARTITION_TOL = 1e-5
COMP_WT_TOL = 1e-5
UH_TOL = 1e-4

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
VAR_F_INV = ["prcp", "tmean"]
ESSENTIAL_DIAG_KEYS = [
    "precipitation", "rainfall", "snowfall", "snowmelt", "refreezing", "snow_release_to_soil",
    "interception_evaporation", "actual_ET", "infiltration", "surface_runoff", "interflow",
    "recharge_to_groundwater", "soil_overflow", "baseflow_raw", "baseflow_capped",
    "groundwater_loss_raw", "groundwater_loss_capped", "channel_loss", "gate_loss",
    "q_raw_process", "q_after_channel_loss", "q_after_gate", "total_discharge",
    "SMS", "GW", "SNOWPACK", "MELTWATER",
    "INSC", "COEF_t", "SQ_t", "SMSC", "SUB_t", "INTER_t", "RECH_t", "CRAK_t", "K_t", "LG_t",
    "TT", "SG_CRIT", "CFMAX_t", "CFR", "CWH", "partition_sum_error",
]
COMPONENT_DIAG_KEYS = [
    "q_raw_process_components", "q_after_channel_loss_components", "q_after_gate_components",
    "q_routed_components", "channel_loss_components", "gate_loss_components",
    "channel_loss_fraction_components", "zero_flow_probability_components",
    "zero_flow_keep_fraction_components", "component_weights", "route_a_components", "route_b_components",
]
PARAM_BOUNDS = OrderedDict([
    ("INSC", (0.5, 5.0)),
    ("COEF_t", (50.0, 400.0)),
    ("SQ_t", (0.0, 6.0)),
    ("SMSC", (50.0, 500.0)),
    ("SUB_t", (0.0, 1.0)),
    ("INTER_t", (0.0, 1.0)),
    ("RECH_t", (0.0, 1.0)),
    ("CRAK_t", (0.0, 1.0)),
    ("K_t", (0.003, 0.3)),
    ("LG_t", (0.0, 0.2)),
    ("TT", (-2.5, 2.5)),
    ("CFMAX_t", (0.0, 15.0)),
    ("CFR", (0.0, 0.1)),
    ("CWH", (0.0, 0.2)),
    ("SG_CRIT", (0.0, 300.0)),
])
FLUX_KEYS = [
    "rainfall", "snowfall", "snowmelt", "interception_evaporation", "actual_ET", "infiltration",
    "recharge_to_groundwater", "surface_runoff", "interflow", "baseflow_capped",
    "groundwater_loss_capped", "channel_loss", "gate_loss", "q_after_gate",
]
STATE_KEYS = ["SMS", "GW", "SNOWPACK", "MELTWATER"]


def seasonal_features(t_range):
    t_arr = utils.time.tRange2Array(t_range)
    dates = pd.to_datetime(t_arr.astype(str))
    doy = dates.dayofyear.to_numpy(dtype=np.float32)
    ang = 2.0 * np.pi * (doy - 1.0) / 365.0
    return np.stack([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)


def safe_ratio(num, den):
    if not np.isfinite(den) or abs(den) <= 1e-8:
        return np.nan
    return num / den


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def legacy_state_to_current(state):
    mapped = OrderedDict()
    key_map = {
        "lstmdyn.lstm.w_ih": "lstmdyn.lstm.weight_ih_l0",
        "lstmdyn.lstm.w_hh": "lstmdyn.lstm.weight_hh_l0",
        "lstmdyn.lstm.b_ih": "lstmdyn.lstm.bias_ih_l0",
        "lstmdyn.lstm.b_hh": "lstmdyn.lstm.bias_hh_l0",
    }
    for key, value in state.items():
        mapped[key_map.get(key, key)] = value
    return mapped


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
    baseline_run = ROOT / "outputs" / "rnnStreamflow" / "CAMELSMODELSIX" / "DynamicSimHydModelSix" / "AllBasins" / FORCING / str(SEED) / (
        "T_19801001_19951001_BS_32_HS_64_RHO_365_Buff_365_Mul_4_Route_1_CmpW_1_LGDyn_1_DSQ_1_DETGAM_1_DPART_1_DCFMAX_1_DROUTE_0_CRoute_1_DryCh_1_ZGate_1_MaxIter200_All671_BS32_HS64_MaxIter200"
    )
    with open(baseline_run / "statDict.json", "r") as fp:
        stat_dict = json.load(fp)

    camels.initcamels(str(root_db))
    gageinfo = camels.gageDict
    basin_ids = gageinfo["id"].tolist()
    basin_index = {gid: i for i, gid in enumerate(basin_ids)}
    train_ind = [basin_index[j] for j in basin_ids]
    areas = gageinfo["area"][train_ind]

    df_train = camels.DataframeCamels(tRange=T_TRAIN, subset=basin_ids, forType=FORCING)
    forc_un = df_train.getDataTs(varLst=VAR_F, doNorm=False, rmNan=False).astype(np.float32)
    obs_un = df_train.getDataObs(doNorm=False, rmNan=False, basinnorm=False).astype(np.float32)
    temp_area = np.tile(areas[:, None, None], (1, obs_un.shape[1], 1))
    obs_un = (obs_un * 0.0283168 * 3600 * 24) / (temp_area * (10 ** 6)) * 10 ** 3

    df_inv = camels.DataframeCamels(tRange=T_INV, subset=basin_ids, forType=FORCING)
    forc_inv_un = df_inv.getDataTs(varLst=VAR_F_INV, doNorm=False, rmNan=False).astype(np.float32)
    attrs_un = df_inv.getDataConst(varLst=ATTR_LST, doNorm=False, rmNan=False).astype(np.float32)

    pet_full, t_pet_lst = load_pet_full(root_db, gageinfo["id"], FORCING)
    t_train_lst = utils.time.tRange2Array(T_TRAIN)
    t_inv_lst = utils.time.tRange2Array(T_INV)
    _, _, ind2 = np.intersect1d(t_train_lst, t_pet_lst, return_indices=True)
    _, _, ind2inv = np.intersect1d(t_inv_lst, t_pet_lst, return_indices=True)
    pet_un = pet_full[:, ind2, :][train_ind, :, :]
    pet_inv_un = pet_full[:, ind2inv, :][train_ind, :, :]

    season_train = np.tile(seasonal_features(T_TRAIN)[None, :, :], (len(basin_ids), 1, 1))

    series_inv = np.concatenate([forc_inv_un, pet_inv_un], axis=2)
    attr_norm = camels.transNormbyDic(attrs_un, ATTR_LST, stat_dict, toNorm=True)
    attr_norm[np.isnan(attr_norm)] = 0.0
    series_norm = camels.transNormbyDic(series_inv, VAR_F_INV + ["pet"], stat_dict, toNorm=True)
    series_norm[np.isnan(series_norm)] = 0.0
    snow_frac_raw = attrs_un[:, SNOW_FRAC_IDX:SNOW_FRAC_IDX + 1].astype(np.float32)
    snow_frac_ts = np.repeat(snow_frac_raw[:, None, :], series_norm.shape[1], axis=1)
    z_train = np.concatenate([series_norm, snow_frac_ts], axis=2).astype(np.float32)
    x_train = np.concatenate([forc_un, pet_un, season_train], axis=2).astype(np.float32)
    x_train[np.isnan(x_train)] = 0.0
    y_train = obs_un.astype(np.float32)

    df_test = camels.DataframeCamels(tRange=T_TEST, subset=basin_ids, forType=FORCING)
    forc_test = df_test.getDataTs(varLst=VAR_F, doNorm=False, rmNan=False).astype(np.float32)
    obs_test = df_test.getDataObs(doNorm=False, rmNan=False, basinnorm=False).astype(np.float32)
    temp_area_test = np.tile(areas[:, None, None], (1, obs_test.shape[1], 1))
    obs_test = (obs_test * 0.0283168 * 3600 * 24) / (temp_area_test * (10 ** 6)) * 10 ** 3
    obs_test = obs_test[:, :, 0].astype(np.float32)

    _, _, ind2test = np.intersect1d(utils.time.tRange2Array(T_TEST), t_pet_lst, return_indices=True)
    pet_test = pet_full[:, ind2test, :][train_ind, :, :]
    season_hist = np.tile(seasonal_features(T_TRAIN)[None, :, :], (len(basin_ids), 1, 1))
    season_test = np.tile(seasonal_features(T_TEST)[None, :, :], (len(basin_ids), 1, 1))
    x_hist = np.concatenate([forc_un, pet_un, season_hist], axis=2).astype(np.float32)
    x_hist[np.isnan(x_hist)] = 0.0
    x_test = np.concatenate([forc_test, pet_test, season_test], axis=2).astype(np.float32)
    x_test[np.isnan(x_test)] = 0.0
    x_eval = np.concatenate([x_hist, x_test], axis=1)

    series_test = np.concatenate([forc_test, pet_test], axis=2)
    series_eval = np.concatenate([series_inv, series_test], axis=1)
    series_norm_eval = camels.transNormbyDic(series_eval, VAR_F_INV + ["pet"], stat_dict, toNorm=True)
    series_norm_eval[np.isnan(series_norm_eval)] = 0.0
    snow_frac_ts_eval = np.repeat(snow_frac_raw[:, None, :], series_norm_eval.shape[1], axis=1)
    c_temp = np.repeat(attr_norm[:, None, :], series_norm_eval.shape[1], axis=1)
    z_eval = np.concatenate([series_norm_eval, snow_frac_ts_eval, c_temp], axis=2).astype(np.float32)

    meta = pd.DataFrame({
        "basin_id": basin_ids,
        "lat": gageinfo["lat"],
        "lon": gageinfo["lon"],
        "area_km2": areas,
    })
    return {
        "root_db": root_db,
        "baseline_run": baseline_run,
        "stat_dict": stat_dict,
        "basin_ids": basin_ids,
        "areas": areas,
        "meta": meta,
        "attr_norm": attr_norm.astype(np.float32),
        "x_train": x_train,
        "y_train": y_train,
        "z_train": z_train,
        "x_eval": x_eval,
        "z_eval": z_eval,
        "x_test_only": x_test,
        "obs_test": obs_test,
        "t_train_array": utils.time.tRange2Array(T_TRAIN),
        "t_test_array": utils.time.tRange2Array(T_TEST),
    }


def build_model(gate_variant):
    ninv = 4 + len(ATTR_LST)
    model = rnn.MultiInv_DynamicSimHydModelSix_Physical(
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
        gate_variant=gate_variant,
        gate_strength_max=0.30,
    )
    return model


def load_baseline_weights(model, baseline_state_path):
    state = torch.load(str(baseline_state_path), map_location="cpu")
    state = legacy_state_to_current(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected


def to_device(*args):
    if torch.cuda.is_available():
        return [x.cuda(GPU_ID) for x in args]
    return list(args)


def evaluate_predictions(model, prepared, return_diagnostics=False, return_component_diagnostics=False):
    basin_ids = prepared["basin_ids"]
    n_basin = len(basin_ids)
    obs_test = prepared["obs_test"]
    x_eval = prepared["x_eval"]
    z_eval = prepared["z_eval"]
    x_hist_len = len(prepared["t_train_array"])
    test_len = len(prepared["t_test_array"])
    x_test_only = prepared["x_test_only"]

    old_inittime = model.inittime
    old_training = model.training
    model.inittime = x_hist_len
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
            outputs = model(
                x_part,
                z_part,
                return_diagnostics=return_diagnostics,
                return_component_diagnostics=return_component_diagnostics,
            )
        if return_diagnostics:
            q_part, diag_part = outputs
        else:
            q_part = outputs
            diag_part = None
        pred[i0:i1] = q_part.detach().cpu().numpy()[:, :, 0].T
        if return_diagnostics and diag_store is None:
            diag_store = {}
            for key in ESSENTIAL_DIAG_KEYS:
                if key in diag_part:
                    diag_store[key] = np.zeros((n_basin, test_len), dtype=np.float32)
            if return_component_diagnostics:
                comp_store = {
                    "component_weights": np.zeros((n_basin, NMUL), dtype=np.float32),
                    "route_a_components": np.zeros((n_basin, NMUL), dtype=np.float32),
                    "route_b_components": np.zeros((n_basin, NMUL), dtype=np.float32),
                    "q_after_gate_components": np.zeros((n_basin, test_len, NMUL), dtype=np.float32),
                    "q_raw_process_components": np.zeros((n_basin, test_len, NMUL), dtype=np.float32),
                    "channel_loss_components": np.zeros((n_basin, test_len, NMUL), dtype=np.float32),
                    "gate_loss_components": np.zeros((n_basin, test_len, NMUL), dtype=np.float32),
                    "channel_loss_fraction_components": np.zeros((n_basin, test_len, NMUL), dtype=np.float32),
                    "zero_flow_probability_components": np.zeros((n_basin, test_len, NMUL), dtype=np.float32),
                    "zero_flow_keep_fraction_components": np.zeros((n_basin, test_len, NMUL), dtype=np.float32),
                }

        if return_diagnostics:
            for key in diag_store.keys():
                arr = diag_part[key].detach().cpu().numpy()
                if arr.ndim == 3:
                    diag_store[key][i0:i1] = arr[:, :, 0].T
                else:
                    raise ValueError(f"Unexpected diag ndim for {key}: {arr.ndim}")

            if return_component_diagnostics:
                for key in ["component_weights", "route_a_components", "route_b_components"]:
                    comp_store[key][i0:i1] = diag_part[key].detach().cpu().numpy()
                for key in [
                    "q_after_gate_components", "q_raw_process_components", "channel_loss_components", "gate_loss_components",
                    "channel_loss_fraction_components", "zero_flow_probability_components", "zero_flow_keep_fraction_components",
                ]:
                    comp_store[key][i0:i1] = np.transpose(diag_part[key].detach().cpu().numpy(), (1, 0, 2))

    model.inittime = old_inittime
    model.train(mode=old_training)

    return {
        "pred": pred,
        "obs": obs_test,
        "x_test_only": x_test_only,
        "diag": diag_store,
        "comp": comp_store,
    }


def summarize_core_metrics(obs, pred):
    n = obs.shape[0]
    rows = []
    for i in range(n):
        o = obs[i]
        s = pred[i]
        rows.append({
            "NSE": calc_nse(o, s),
            "KGE": calc_kge(o, s),
            "R2": calc_r2(o, s),
            "FLV": calc_flv(o, s),
            "FHV": calc_fhv(o, s),
            "low_flow_NSE": lowflow_nse(o, s),
            "high_flow_NSE": highflow_nse(o, s),
            "mean_obs_q": float(np.nanmean(o)),
            "mean_sim_q": float(np.nanmean(s)),
            "runoff_ratio_sim_obs": safe_ratio(float(np.nanmean(s)), float(np.nanmean(o))),
        })
    df = pd.DataFrame(rows)
    return {
        "median_NSE": float(np.nanmedian(df["NSE"])),
        "mean_NSE": float(np.nanmean(df["NSE"])),
        "median_KGE": float(np.nanmedian(df["KGE"])),
        "median_R2": float(np.nanmedian(df["R2"])),
        "median_FLV": float(np.nanmedian(df["FLV"])),
        "median_FHV": float(np.nanmedian(df["FHV"])),
        "median_low_flow_NSE": float(np.nanmedian(df["low_flow_NSE"])),
        "median_high_flow_NSE": float(np.nanmedian(df["high_flow_NSE"])),
        "nse_lt_0_count": int(np.sum(df["NSE"] < 0)),
    }


def delta_storage(diag):
    total_storage = diag["SMS"] + diag["GW"] + diag["SNOWPACK"] + diag["MELTWATER"]
    prev = np.concatenate([total_storage[:, :1], total_storage[:, :-1]], axis=1)
    return total_storage - prev


def compute_per_basin_tables(variant_name, prepared, eval_res):
    meta = prepared["meta"].copy()
    obs = eval_res["obs"]
    pred = eval_res["pred"]
    diag = eval_res["diag"]
    comp = eval_res["comp"]
    x_test = eval_res["x_test_only"]
    basin_ids = prepared["basin_ids"]

    d_storage = delta_storage(diag)
    et_total = diag["actual_ET"] + diag["interception_evaporation"]
    residual = (
        diag["precipitation"]
        - diag["interception_evaporation"]
        - diag["actual_ET"]
        - diag["groundwater_loss_capped"]
        - diag["channel_loss"]
        - diag["gate_loss"]
        - diag["q_after_gate"]
        - d_storage
    )

    wb_rows = []
    metric_rows = []
    viol_rows = []
    param_rows = []
    et_snow_rows = []
    comp_rows = []
    for i, basin_id in enumerate(basin_ids):
        o = obs[i]
        s = pred[i]
        p = diag["precipitation"][i]
        rr = diag["rainfall"][i]
        sn = diag["snowfall"][i]
        sm = diag["snowmelt"][i]
        rf = diag["refreezing"][i]
        rel = diag["snow_release_to_soil"][i]
        temp = x_test[i, :, 1]
        tt = diag["TT"][i]
        snpk = diag["SNOWPACK"][i]
        mw = diag["MELTWATER"][i]
        cwh = diag["CWH"][i]
        pet = x_test[i, :, 2]
        available_soil = diag["SMS"][i] + diag["soil_overflow"][i]
        res_i = residual[i]
        qproc = diag["q_after_gate"][i]
        qmean_obs = float(np.nanmean(o))
        qmean_sim = float(np.nanmean(s))
        runoff_bias = safe_ratio(qmean_sim - qmean_obs, qmean_obs)
        etp_ratio = safe_ratio(float(np.nansum(et_total[i])), float(np.nansum(p)))
        wet_basin_flag = float(np.nanmean(p)) > 2.0
        extreme_channel_loss = float(np.nanmean(diag["channel_loss"][i])) > max(0.5, 0.5 * qmean_sim)
        extreme_gw_loss = float(np.nanmean(diag["groundwater_loss_capped"][i])) > max(0.5, 0.5 * qmean_sim)
        near_zero_for_wet = wet_basin_flag and qmean_sim < 0.1 and qmean_obs > 1.0

        wb_rows.append({
            "variant": variant_name,
            "basin_id": basin_id,
            "mean_abs_water_balance_error_mm_day": float(np.nanmean(np.abs(res_i))),
            "max_abs_daily_water_balance_error_mm_day": float(np.nanmax(np.abs(res_i))),
            "cumulative_water_balance_error_mm": float(np.nansum(res_i)),
            "cumulative_abs_daily_error_mm": float(np.nansum(np.abs(res_i))),
            "cumulative_precipitation_mm": float(np.nansum(p)),
            "relative_error": safe_ratio(float(np.nansum(np.abs(res_i))), float(np.nansum(p))),
            "n_daily_error_gt_1e_4": int(np.sum(np.abs(res_i) > 1e-4)),
        })

        metric_rows.append({
            "variant": variant_name,
            "basin_id": basin_id,
            "NSE": calc_nse(o, s),
            "KGE": calc_kge(o, s),
            "R2": calc_r2(o, s),
            "FLV": calc_flv(o, s),
            "FHV": calc_fhv(o, s),
            "low_flow_NSE": lowflow_nse(o, s),
            "high_flow_NSE": highflow_nse(o, s),
            "bias": float(np.nanmean(s - o)),
            "runoff_ratio_bias": runoff_bias,
            "observed_mean_q": qmean_obs,
            "simulated_mean_q": qmean_sim,
            "sim_obs_q_ratio": safe_ratio(qmean_sim, qmean_obs),
            "precip_runoff_ratio_obs": safe_ratio(float(np.nansum(p)), float(np.nansum(o))),
            "precip_runoff_ratio_sim": safe_ratio(float(np.nansum(p)), float(np.nansum(s))),
            "ET_P_ratio": etp_ratio,
            "extreme_channel_loss_flag": bool(extreme_channel_loss),
            "extreme_groundwater_loss_flag": bool(extreme_gw_loss),
            "near_zero_sim_discharge_wet_flag": bool(near_zero_for_wet),
        })

        for key in FLUX_KEYS:
            arr = diag[key][i]
            viol_rows.append({
                "variant": variant_name,
                "basin_id": basin_id,
                "kind": "flux",
                "variable": key,
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "mean": float(np.nanmean(arr)),
                "violation_count": int(np.sum(arr < NEG_TOL)),
            })
        for key in STATE_KEYS:
            arr = diag[key][i]
            if key == "SMS":
                exceed = arr - diag["SMSC"][i]
                violation_count = int(np.sum(exceed > 1e-6))
                extra_max = float(np.nanmax(np.maximum(exceed, 0.0)))
            else:
                violation_count = int(np.sum(arr < NEG_TOL))
                extra_max = np.nan
            viol_rows.append({
                "variant": variant_name,
                "basin_id": basin_id,
                "kind": "state",
                "variable": key,
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "mean": float(np.nanmean(arr)),
                "violation_count": violation_count,
                "extra_max": extra_max,
            })

        for key, (lo, hi) in PARAM_BOUNDS.items():
            arr = diag[key][i]
            param_rows.append({
                "variant": variant_name,
                "basin_id": basin_id,
                "parameter": key,
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "mean": float(np.nanmean(arr)),
                "bound_low": lo,
                "bound_high": hi,
                "violation_count": int(np.sum((arr < lo - 1e-6) | (arr > hi + 1e-6))),
            })
        param_rows.append({
            "variant": variant_name,
            "basin_id": basin_id,
            "parameter": "partition_sum_error",
            "min": float(np.nanmin(diag["partition_sum_error"][i])),
            "max": float(np.nanmax(diag["partition_sum_error"][i])),
            "mean": float(np.nanmean(diag["partition_sum_error"][i])),
            "bound_low": 0.0,
            "bound_high": PARTITION_TOL,
            "violation_count": int(np.sum(diag["partition_sum_error"][i] > PARTITION_TOL)),
        })
        comp_sum_err = abs(float(np.sum(comp["component_weights"][i])) - 1.0)
        route_sums = []
        for j in range(NMUL):
            a = torch.full((15, 1, 1), float(comp["route_a_components"][i, j]), dtype=torch.float32)
            b = torch.full((15, 1, 1), float(comp["route_b_components"][i, j]), dtype=torch.float32)
            route_sums.append(float(rnn.UH_gamma(a, b, lenF=15).sum().item()))
        param_rows.append({
            "variant": variant_name,
            "basin_id": basin_id,
            "parameter": "component_weight_sum_error",
            "min": comp_sum_err,
            "max": comp_sum_err,
            "mean": comp_sum_err,
            "bound_low": 0.0,
            "bound_high": COMP_WT_TOL,
            "violation_count": int(comp_sum_err > COMP_WT_TOL),
        })
        param_rows.append({
            "variant": variant_name,
            "basin_id": basin_id,
            "parameter": "unit_hydrograph_sum_error_max",
            "min": float(np.min(np.abs(np.array(route_sums) - 1.0))),
            "max": float(np.max(np.abs(np.array(route_sums) - 1.0))),
            "mean": float(np.mean(np.abs(np.array(route_sums) - 1.0))),
            "bound_low": 0.0,
            "bound_high": UH_TOL,
            "violation_count": int(np.sum(np.abs(np.array(route_sums) - 1.0) > UH_TOL)),
        })

        snowfall_cold_frac = safe_ratio(float(np.nansum(sn[temp < tt])), float(np.nansum(sn)))
        snowmelt_warm_frac = safe_ratio(float(np.nansum(sm[temp > tt])), float(np.nansum(sm)))
        meltwater_excess = mw - cwh * np.maximum(snpk, 0.0)
        et_snow_rows.append({
            "variant": variant_name,
            "basin_id": basin_id,
            "et_exceeds_pet_count": int(np.sum(diag["actual_ET"][i] > pet + 1e-6)),
            "interception_exceeds_pet_count": int(np.sum(diag["interception_evaporation"][i] > pet + 1e-6)),
            "soil_et_exceeds_available_count": int(np.sum(diag["actual_ET"][i] > available_soil + 1e-6)),
            "ET_P_ratio": etp_ratio,
            "snowfall_when_cold_fraction": snowfall_cold_frac,
            "snowmelt_when_warm_fraction": snowmelt_warm_frac,
            "meltwater_exceeds_cwh_count": int(np.sum(meltwater_excess > 1e-6)),
            "meltwater_exceeds_cwh_max_mm": float(np.nanmax(np.maximum(meltwater_excess, 0.0))),
            "annual_max_snowpack_mm": float(np.nanmax(snpk)),
        })

        q_comp = comp["q_after_gate_components"][i]
        q_sum = np.sum(q_comp, axis=1)
        contrib = np.divide(np.nansum(q_comp, axis=0), np.nansum(q_sum) + 1e-8)
        near_zero_all = np.nansum(q_comp, axis=0) < 1e-3
        dominant = contrib > 0.8
        for j in range(NMUL):
            comp_rows.append({
                "variant": variant_name,
                "basin_id": basin_id,
                "component": j + 1,
                "mean_component_discharge": float(np.nanmean(q_comp[:, j])),
                "component_weight": float(comp["component_weights"][i, j]),
                "component_contribution_to_final_q": float(contrib[j]),
                "collapsed_near_zero": bool(near_zero_all[j]),
                "dominates_basin": bool(dominant[j]),
                "mean_channel_loss_component": float(np.nanmean(comp["channel_loss_components"][i, :, j])),
                "mean_gate_loss_component": float(np.nanmean(comp["gate_loss_components"][i, :, j])),
            })

    wb_df = pd.DataFrame(wb_rows)
    metrics_df = pd.DataFrame(metric_rows).merge(meta, on="basin_id", how="left")
    viol_df = pd.DataFrame(viol_rows)
    param_df = pd.DataFrame(param_rows)
    et_snow_df = pd.DataFrame(et_snow_rows)
    comp_df = pd.DataFrame(comp_rows)
    return {
        "water_balance": wb_df,
        "metrics": metrics_df,
        "violations": viol_df,
        "parameter_bounds": param_df,
        "et_snow": et_snow_df,
        "component": comp_df,
        "residual": residual,
        "diag": diag,
        "comp_raw": comp,
    }


def save_npz_selected(path, payload):
    arrs = {}
    for key, val in payload.items():
        if isinstance(val, np.ndarray):
            arrs[key] = val
    np.savez_compressed(path, **arrs)


def plot_outputs(out_root, baseline_metrics, final_tables, original_wb, baseline_diag_npz):
    ensure_dir(out_root / "plots" / "worst_nse_hydrographs")
    ensure_dir(out_root / "plots" / "worst_water_balance")
    plt.figure(figsize=(8, 5))
    plt.hist(original_wb["mean_abs_water_balance_error_mm_day"], bins=40, alpha=0.5, label="original")
    for variant, tables in final_tables.items():
        plt.hist(tables["water_balance"]["mean_abs_water_balance_error_mm_day"], bins=40, alpha=0.5, label=variant)
    plt.xlabel("Mean abs water-balance residual (mm/day)")
    plt.ylabel("Basins")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_root / "plots" / "water_balance_residual_histogram.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.scatter(baseline_metrics["NSE"], final_tables["Model6PhysicalFix_A_explicit_gate"]["metrics"]["NSE"], s=8, alpha=0.5, label="A explicit")
    plt.scatter(baseline_metrics["NSE"], final_tables["Model6PhysicalFix_B_soft_gate"]["metrics"]["NSE"], s=8, alpha=0.5, label="B soft")
    vals = np.array([np.nanmin(baseline_metrics["NSE"]), np.nanmax(baseline_metrics["NSE"])])
    plt.plot(vals, vals, "k--", linewidth=1)
    plt.xlabel("Original NSE")
    plt.ylabel("Physical fix NSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_root / "plots" / "nse_comparison_original_vs_physical.png", dpi=200)
    plt.close()

    baseline_gw = np.load(baseline_diag_npz)["groundwater_loss"]
    baseline_gw_mean = np.nanmean(baseline_gw, axis=1)
    plt.figure(figsize=(8, 5))
    plt.boxplot(
        [baseline_gw_mean,
         final_tables["Model6PhysicalFix_A_explicit_gate"]["diag"]["groundwater_loss_capped"].mean(axis=1),
         final_tables["Model6PhysicalFix_B_soft_gate"]["diag"]["groundwater_loss_capped"].mean(axis=1)],
        labels=["original", "A explicit", "B soft"],
        showfliers=False,
    )
    plt.ylabel("Mean groundwater loss (mm/day)")
    plt.tight_layout()
    plt.savefig(out_root / "plots" / "groundwater_loss_before_vs_after.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for variant, tables in final_tables.items():
        weights = tables["comp_raw"]["component_weights"]
        for j in range(NMUL):
            plt.hist(weights[:, j], bins=30, alpha=0.25, label=f"{variant} C{j+1}" if j == 0 else None)
    plt.xlabel("Component weight")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_root / "plots" / "component_weight_distributions.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 7))
    for variant, tables in final_tables.items():
        plt.scatter(
            tables["metrics"]["precip_runoff_ratio_obs"],
            tables["metrics"]["precip_runoff_ratio_sim"],
            s=8,
            alpha=0.5,
            label=variant,
        )
    plt.xlabel("Observed P/Q ratio")
    plt.ylabel("Simulated P/Q ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_root / "plots" / "runoff_ratio_observed_vs_simulated.png", dpi=200)
    plt.close()


def generate_worst_basin_plots(out_root, final_tables, eval_payloads, best_variant):
    metrics = final_tables[best_variant]["metrics"].sort_values("NSE").head(25)
    worst_ids = metrics["basin_id"].tolist()
    basin_to_idx = {b: i for i, b in enumerate(eval_payloads[best_variant]["basin_ids"])}
    diag = final_tables[best_variant]["diag"]
    obs = eval_payloads[best_variant]["obs"]
    pred = eval_payloads[best_variant]["pred"]
    for basin_id in worst_ids:
        i = basin_to_idx[basin_id]
        plt.figure(figsize=(12, 4))
        plt.plot(obs[i], label="obs", linewidth=1.0)
        plt.plot(pred[i], label=best_variant, linewidth=1.0)
        plt.title(f"{best_variant} worst NSE basin {basin_id}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_root / "plots" / "worst_nse_hydrographs" / f"{basin_id}.png", dpi=160)
        plt.close()

        p = diag["precipitation"][i]
        q = diag["q_after_gate"][i]
        et = diag["actual_ET"][i] + diag["interception_evaporation"][i]
        gwloss = diag["groundwater_loss_capped"][i]
        chloss = diag["channel_loss"][i]
        gateloss = diag["gate_loss"][i]
        dstore = delta_storage(diag)[i]
        plt.figure(figsize=(12, 5))
        plt.plot(np.cumsum(p), label="cum P")
        plt.plot(np.cumsum(q), label="cum Q_process")
        plt.plot(np.cumsum(et), label="cum ET")
        plt.plot(np.cumsum(gwloss + chloss + gateloss), label="cum losses")
        plt.plot(np.cumsum(dstore), label="cum dS")
        plt.title(f"{best_variant} cumulative balance basin {basin_id}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_root / "plots" / "worst_water_balance" / f"{basin_id}.png", dpi=160)
        plt.close()


def quick_eval_row(variant, epoch, obs, pred):
    s = summarize_core_metrics(obs, pred)
    s["variant"] = variant
    s["epoch"] = epoch
    return s


def select_best_variant(final_summary):
    ranked = []
    for name, row in final_summary.items():
        ranked.append((
            row["median_relative_error"],
            row["mass_flag_count"],
            -row["median_NSE"],
            -row["median_low_flow_NSE"],
            name,
        ))
    ranked.sort()
    return ranked[0][-1]


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.set_device(GPU_ID)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(8)

    out_root = ROOT / "Model_six_physical"
    ensure_dir(out_root)
    ensure_dir(out_root / "plots")
    ensure_dir(out_root / "variants")

    prepared = prepare_data()
    baseline_run = prepared["baseline_run"]
    baseline_state_path = baseline_run / "model_Ep30_state_legacy.pt"
    baseline_pred = np.load(baseline_run.parent / "Train19801001_19951001Test19951001_20101001_ModelSix_Ep30Resume" / "pred30.npy")
    baseline_obs = np.load(baseline_run.parent / "Train19801001_19951001Test19951001_20101001_ModelSix_Ep30Resume" / "obs.npy")
    if baseline_pred.ndim == 3:
        baseline_pred = baseline_pred[:, :, 0]
    if baseline_obs.ndim == 3:
        baseline_obs = baseline_obs[:, :, 0]
    baseline_metrics = []
    for i, basin_id in enumerate(prepared["basin_ids"]):
        baseline_metrics.append({
            "basin_id": basin_id,
            "NSE": calc_nse(baseline_obs[i], baseline_pred[i]),
            "KGE": calc_kge(baseline_obs[i], baseline_pred[i]),
            "R2": calc_r2(baseline_obs[i], baseline_pred[i]),
            "FLV": calc_flv(baseline_obs[i], baseline_pred[i]),
            "FHV": calc_fhv(baseline_obs[i], baseline_pred[i]),
            "low_flow_NSE": lowflow_nse(baseline_obs[i], baseline_pred[i]),
            "high_flow_NSE": highflow_nse(baseline_obs[i], baseline_pred[i]),
        })
    baseline_metrics = pd.DataFrame(baseline_metrics)
    original_wb = pd.read_csv(ROOT / "outputs" / "Model6_Epoch30_PhysicalAudit" / "model6_epoch30_per_basin_water_balance.csv")
    baseline_diag_npz = ROOT / "outputs" / "rnnStreamflow" / "CAMELSMODELSIX" / "DynamicSimHydModelSix" / "AllBasins" / FORCING / str(SEED) / "analysis_ep30" / "model_six_diagnostics_ep30.npz"

    training_log_rows = []
    metrics_by_epoch_rows = []
    final_tables = {}
    eval_payloads = {}
    variant_specs = OrderedDict([
        ("Model6PhysicalFix_A_explicit_gate", "explicit"),
        ("Model6PhysicalFix_B_soft_gate", "soft"),
    ])

    for variant_name, gate_variant in variant_specs.items():
        variant_dir = out_root / "variants" / variant_name
        ensure_dir(variant_dir)
        print(f"Starting variant {variant_name} ({gate_variant})", flush=True)
        model = build_model(gate_variant)
        missing, unexpected = load_baseline_weights(model, baseline_state_path)
        (variant_dir / "load_report.json").write_text(json.dumps({
            "missing": missing,
            "unexpected": unexpected,
            "gate_variant": gate_variant,
        }, indent=2))
        if torch.cuda.is_available():
            model = model.cuda(GPU_ID)
        loss_fun = crit.RmseLossComb(alpha=ALPHA)
        if torch.cuda.is_available():
            loss_fun = loss_fun.cuda()
        optim = torch.optim.Adadelta(model.parameters(), lr=LR)
        model.zero_grad()

        run_csv = variant_dir / "run.csv"
        with open(run_csv, "w") as rf:
            for epoch in range(1, EPOCHS + 1):
                model.train(True)
                loss_ep = 0.0
                t0 = time.time()
                for _ in range(MAX_ITER_EP):
                    i_grid, i_t = train.randomIndex(len(prepared["basin_ids"]), prepared["x_train"].shape[1], [BATCH_SIZE, RHO], bufftime=BUFFTIME)
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
                epoch_time = time.time() - t0
                training_log_rows.append({
                    "variant": variant_name,
                    "epoch": epoch,
                    "loss": loss_ep,
                    "epoch_time_sec": epoch_time,
                })
                msg = f"Epoch {epoch} Loss {loss_ep:.4f} time {epoch_time:.2f}"
                print(f"{variant_name}: {msg}", flush=True)
                rf.write(msg + "\n")
                rf.flush()
                torch.save(model.state_dict(), variant_dir / f"model_Ep{epoch}_state.pt")
        print(f"{variant_name}: running final diagnostic evaluation", flush=True)
        final_eval = evaluate_predictions(model, prepared, return_diagnostics=True, return_component_diagnostics=True)
        eval_payloads[variant_name] = {
            "basin_ids": prepared["basin_ids"],
            "obs": final_eval["obs"],
            "pred": final_eval["pred"],
        }
        metrics_by_epoch_rows.append(quick_eval_row(variant_name, EPOCHS, final_eval["obs"], final_eval["pred"]))
        tables = compute_per_basin_tables(variant_name, prepared, final_eval)
        final_tables[variant_name] = tables
        save_npz_selected(variant_dir / "final_eval_selected.npz", {
            "pred": final_eval["pred"],
            "obs": final_eval["obs"],
            **{f"diag_{k}": v for k, v in tables["diag"].items()},
        })

    training_log_df = pd.DataFrame(training_log_rows)
    training_log_df.to_csv(out_root / "model6_physicalfix_training_log.csv", index=False)
    metrics_by_epoch_df = pd.DataFrame(metrics_by_epoch_rows)
    metrics_by_epoch_df.to_csv(out_root / "model6_physicalfix_metrics_by_epoch.csv", index=False)

    per_basin_frames = [baseline_metrics.assign(variant="original")]
    water_summary_rows = [{
        "variant": "original",
        "median_mean_abs_residual_mm_day": float(original_wb["mean_abs_water_balance_error_mm_day"].median()),
        "mean_mean_abs_residual_mm_day": float(original_wb["mean_abs_water_balance_error_mm_day"].mean()),
        "median_relative_error": float(original_wb["relative_error"].median()),
        "mean_relative_error": float(original_wb["relative_error"].mean()),
        "basins_gt_1pct_relative_error": int(np.sum(original_wb["relative_error"] > 0.01)),
    }]
    worst_frames = []
    flux_state_frames = []
    param_frames = []
    component_frames = []
    et_snow_frames = []
    final_summary = {}

    for variant_name, tables in final_tables.items():
        metrics_df = tables["metrics"].copy()
        per_basin_frames.append(metrics_df)
        flux_state_frames.append(tables["violations"])
        param_frames.append(tables["parameter_bounds"])
        component_frames.append(tables["component"])
        et_snow_frames.append(tables["et_snow"])
        wb_df = tables["water_balance"]
        water_summary_rows.append({
            "variant": variant_name,
            "median_mean_abs_residual_mm_day": float(wb_df["mean_abs_water_balance_error_mm_day"].median()),
            "mean_mean_abs_residual_mm_day": float(wb_df["mean_abs_water_balance_error_mm_day"].mean()),
            "median_relative_error": float(wb_df["relative_error"].median()),
            "mean_relative_error": float(wb_df["relative_error"].mean()),
            "basins_gt_1pct_relative_error": int(np.sum(wb_df["relative_error"] > 0.01)),
        })
        combined = metrics_df.merge(wb_df[["basin_id", "mean_abs_water_balance_error_mm_day", "relative_error"]], on="basin_id", how="left")
        worst_frames.append(combined.sort_values(["NSE", "relative_error"]).head(25).assign(variant=variant_name))
        final_summary[variant_name] = {
            "median_NSE": float(np.nanmedian(metrics_df["NSE"])),
            "median_KGE": float(np.nanmedian(metrics_df["KGE"])),
            "median_R2": float(np.nanmedian(metrics_df["R2"])),
            "median_FLV": float(np.nanmedian(metrics_df["FLV"])),
            "median_FHV": float(np.nanmedian(metrics_df["FHV"])),
            "nse_lt_0_count": int(np.sum(metrics_df["NSE"] < 0)),
            "median_low_flow_NSE": float(np.nanmedian(metrics_df["low_flow_NSE"])),
            "median_high_flow_NSE": float(np.nanmedian(metrics_df["high_flow_NSE"])),
            "median_relative_error": float(np.nanmedian(wb_df["relative_error"])),
            "mass_flag_count": int(np.sum(wb_df["relative_error"] > 0.01)),
            "median_et_p_ratio": float(np.nanmedian(metrics_df["ET_P_ratio"])),
            "median_runoff_ratio_sim_obs": float(np.nanmedian(metrics_df["sim_obs_q_ratio"])),
        }

    per_basin_df = pd.concat(per_basin_frames, ignore_index=True, sort=False)
    per_basin_df.to_csv(out_root / "model6_physicalfix_per_basin_metrics.csv", index=False)
    wb_summary_df = pd.DataFrame(water_summary_rows)
    wb_summary_df.to_csv(out_root / "model6_physicalfix_water_balance_summary.csv", index=False)
    pd.concat(flux_state_frames, ignore_index=True).to_csv(out_root / "model6_physicalfix_flux_state_violations.csv", index=False)
    pd.concat(param_frames, ignore_index=True).to_csv(out_root / "model6_physicalfix_parameter_bounds.csv", index=False)
    pd.concat(component_frames, ignore_index=True).to_csv(out_root / "model6_physicalfix_component_diagnostics.csv", index=False)
    pd.concat(et_snow_frames, ignore_index=True).to_csv(out_root / "model6_physicalfix_et_snow_sanity_checks.csv", index=False)
    pd.concat(worst_frames, ignore_index=True).to_csv(out_root / "model6_physicalfix_worst_basins.csv", index=False)

    best_variant = select_best_variant(final_summary)
    plot_outputs(out_root, baseline_metrics, final_tables, original_wb, baseline_diag_npz)
    generate_worst_basin_plots(out_root, final_tables, eval_payloads, best_variant)

    summary_lines = [
        "Model 6 physical-fix summary",
        f"best_variant: {best_variant}",
        "",
        "Original Model 6:",
        f"- median NSE: {np.nanmedian(baseline_metrics['NSE']):.4f}",
        f"- median KGE: {np.nanmedian(baseline_metrics['KGE']):.4f}",
        f"- median R2: {np.nanmedian(baseline_metrics['R2']):.4f}",
        f"- median FLV: {np.nanmedian(baseline_metrics['FLV']):.4f}",
        f"- median FHV: {np.nanmedian(baseline_metrics['FHV']):.4f}",
        f"- median relative water-balance error: {original_wb['relative_error'].median():.4f}",
        "",
    ]
    for variant_name in variant_specs.keys():
        row = final_summary[variant_name]
        summary_lines.extend([
            f"{variant_name}:",
            f"- median NSE: {row['median_NSE']:.4f}",
            f"- median KGE: {row['median_KGE']:.4f}",
            f"- median R2: {row['median_R2']:.4f}",
            f"- median FLV: {row['median_FLV']:.4f}",
            f"- median FHV: {row['median_FHV']:.4f}",
            f"- NSE<0 count: {row['nse_lt_0_count']}",
            f"- median low-flow NSE: {row['median_low_flow_NSE']:.4f}",
            f"- median high-flow NSE: {row['median_high_flow_NSE']:.4f}",
            f"- median relative water-balance error: {row['median_relative_error']:.6f}",
            f"- basins with relative error >1%: {row['mass_flag_count']}",
            f"- median ET/P ratio: {row['median_et_p_ratio']:.4f}",
            f"- median sim/obs mean Q ratio: {row['median_runoff_ratio_sim_obs']:.4f}",
            "",
        ])
    summary_lines.extend([
        "Selection logic: physical validity first, then median NSE, then low/high-flow behavior.",
        f"Chosen best model: {best_variant}",
    ])
    (out_root / "final_result.txt").write_text("\n".join(summary_lines) + "\n")

    print(pd.DataFrame([{"variant": "original",
                         "median_NSE": float(np.nanmedian(baseline_metrics['NSE'])),
                         "median_KGE": float(np.nanmedian(baseline_metrics['KGE'])),
                         "median_R2": float(np.nanmedian(baseline_metrics['R2'])),
                         "median_FLV": float(np.nanmedian(baseline_metrics['FLV'])),
                         "median_FHV": float(np.nanmedian(baseline_metrics['FHV'])),
                         "median_relative_error": float(original_wb['relative_error'].median()),
                         "nse_lt_0_count": int(np.sum(baseline_metrics['NSE'] < 0)),
                         "median_low_flow_NSE": float(np.nanmedian(baseline_metrics['low_flow_NSE'])),
                         "median_high_flow_NSE": float(np.nanmedian(baseline_metrics['high_flow_NSE']))}] +
                       [dict(variant=k, **v) for k, v in final_summary.items()]).to_string(index=False))


if __name__ == "__main__":
    main()
