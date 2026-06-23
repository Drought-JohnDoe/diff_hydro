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

from hydro_ml.diagnosis import (  # noqa: E402
    calc_fhv,
    calc_flv,
    calc_kge,
    calc_nse,
    highflow_nse,
    lowflow_nse,
    safe_ratio,
)
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
EPOCHS_INITIAL = 10
EPOCHS_TOTAL = 20
BATCH_SIZE = 16
MAX_ITER_EP = 8
HIDDEN_SIZE = 64
NMUL = 4
GPU_ID = 1 if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 0
LR = 0.10
CHUNK_SIZE = 16
NEG_TOL = -1e-6

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

PROJECT_DIR = REPO_ROOT / "model6" / "results" / "train_runs"
RUN_DIR = PROJECT_DIR / "Model6Closed_Snow_aSrz_SIMHYD_Simple_demo32"
PLOTS_DIR = RUN_DIR / "plots"
BASE_DEMO_DIR = DATA_ROOT / "ECO_HYBRID" / "Model6Physical_aSrz_Minimal_demo32"
BASE_DEMO_CKPT = BASE_DEMO_DIR / "model_Ep10_state.pt"
BASE_SUMMARY = BASE_DEMO_DIR / "asrz_before_after_summary.csv"
HBV_DEMO32_SUMMARY = DATA_ROOT / "hybrid_demo32_compare_three_models.csv"
DEMO32_LIST = BASE_DEMO_DIR / "demo32_basin_ids.txt"
BASELINE_RUN = DATA_ROOT / "outputs" / "rnnStreamflow" / "CAMELSMODELSIX" / "DynamicSimHydModelSix" / "AllBasins" / FORCING / str(SEED) / (
    "T_19801001_19951001_BS_32_HS_64_RHO_365_Buff_365_Mul_4_Route_1_CmpW_1_LGDyn_1_DSQ_1_DETGAM_1_DPART_1_DCFMAX_1_DROUTE_0_CRoute_1_DryCh_1_ZGate_1_MaxIter200_All671_BS32_HS64_MaxIter200"
)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def calc_r2(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 2:
        return np.nan
    o = obs[mask]
    s = sim[mask]
    if np.std(o) == 0 or np.std(s) == 0:
        return np.nan
    return float(np.corrcoef(o, s)[0, 1] ** 2)


def seasonal_features(t_range):
    t_arr = utils.time.tRange2Array(t_range)
    dates = pd.to_datetime(t_arr.astype(str))
    doy = dates.dayofyear.to_numpy(dtype=np.float32)
    ang = 2.0 * np.pi * (doy - 1.0) / 365.0
    return np.stack([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)


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
    root_db = DATA_ROOT / "Camels"
    with open(BASELINE_RUN / "statDict.json", "r") as fp:
        stat_dict = json.load(fp)

    subset_ids = [int(x.strip()) for x in DEMO32_LIST.read_text().splitlines() if x.strip()]
    camels.initcamels(str(root_db))
    gageinfo = camels.gageDict
    basin_ids_all = gageinfo["id"].tolist()
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

    return {
        "subset_ids": subset_ids,
        "x_train": x_train,
        "y_train": obs_un.astype(np.float32),
        "z_train": z_train.astype(np.float32),
        "attr_norm": attr_norm.astype(np.float32),
        "x_eval": x_eval.astype(np.float32),
        "z_eval": z_eval.astype(np.float32),
        "obs_test": obs_test.astype(np.float32),
    }


def to_device(*args):
    if torch.cuda.is_available():
        return [x.cuda(GPU_ID) for x in args]
    return list(args)


def diag_numpy(diag, key):
    arr = diag[key]
    if torch.is_tensor(arr):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def build_model():
    ninv = 4 + len(ATTR_LST)
    return rnn.MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimple(
        ninv=ninv, nmul=NMUL, nattr=len(ATTR_LST), hiddeninv=HIDDEN_SIZE, inittime=BUFFTIME,
        routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
        dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
        dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
        reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3, component_routing=True,
    )


def partial_load_from_asrz(source_state, target_model):
    tgt = target_model.state_dict()
    loaded, skipped = [], []
    for k, v in source_state.items():
        if k in tgt and tgt[k].shape == v.shape:
            tgt[k] = v.clone()
            loaded.append(k)
        else:
            skipped.append(k)
    target_model.load_state_dict(tgt, strict=False)
    return loaded, skipped


def evaluate_model_components(model, prepared):
    n_basin = len(prepared["subset_ids"])
    x_eval = prepared["x_eval"]
    z_eval = prepared["z_eval"]
    test_len = prepared["obs_test"].shape[1]
    old_inittime = model.inittime
    old_training = model.training
    model.inittime = len(utils.time.tRange2Array(T_TRAIN))
    model.train(mode=False)

    pred = np.zeros((n_basin, test_len), dtype=np.float32)
    mix_store = {}
    comp_store = {}
    weights = np.zeros((n_basin, NMUL), dtype=np.float32)
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
    return {"pred": pred, "mix": mix_store, "comp": comp_store, "weights": weights}


def one_basin_closure_test(model, prepared):
    model_inittime = model.inittime
    model.inittime = len(utils.time.tRange2Array(T_TRAIN))
    model.train(False)
    x_eval = prepared["x_eval"][:1]
    z_eval = prepared["z_eval"][:1]
    x_part = torch.from_numpy(np.swapaxes(x_eval, 1, 0)).float()
    z_part = torch.from_numpy(np.swapaxes(z_eval, 1, 0)).float()
    x_part, z_part = to_device(x_part, z_part)
    with torch.no_grad():
        _, diag = model(x_part, z_part, return_diagnostics=True, return_component_diagnostics=True)
    model.inittime = model_inittime

    w = diag_numpy(diag, "component_weights")[0]
    p = np.sum(diag_numpy(diag, "precipitation_components")[:, 0, :] * w[None, :], axis=1)
    intv = np.sum(diag_numpy(diag, "interception_evaporation_components")[:, 0, :] * w[None, :], axis=1)
    et = np.sum(diag_numpy(diag, "actual_ET_components")[:, 0, :] * w[None, :], axis=1)
    q = np.sum(diag_numpy(diag, "Q_process_components")[:, 0, :] * w[None, :], axis=1)
    residual_comp = diag_numpy(diag, "process_local_residual_components")[:, 0, :]
    residual = np.sum(residual_comp * w[None, :], axis=1)

    snow_prev = np.sum(diag_numpy(diag, "SNOWPACK_prev_components")[:, 0, :] * w[None, :], axis=1)
    melt_prev = np.sum(diag_numpy(diag, "MELTWATER_prev_components")[:, 0, :] * w[None, :], axis=1)
    sa_prev = np.sum(diag_numpy(diag, "Sa_prev_components")[:, 0, :] * w[None, :], axis=1)
    gw_prev = np.sum(diag_numpy(diag, "GW_prev_components")[:, 0, :] * w[None, :], axis=1)
    snow = np.sum(diag_numpy(diag, "SNOWPACK_components")[:, 0, :] * w[None, :], axis=1)
    melt = np.sum(diag_numpy(diag, "MELTWATER_components")[:, 0, :] * w[None, :], axis=1)
    sa = np.sum(diag_numpy(diag, "Sa_components")[:, 0, :] * w[None, :], axis=1)
    gw = np.sum(diag_numpy(diag, "GW_components")[:, 0, :] * w[None, :], axis=1)
    dstore = (snow - snow_prev) + (melt - melt_prev) + (sa - sa_prev) + (gw - gw_prev)

    closure_df = pd.DataFrame({
        "P": p,
        "INT": intv,
        "ET_a": et,
        "Q_process": q,
        "storage_change": dstore,
        "residual": residual,
        "SNOWPACK": snow,
        "MELTWATER": melt,
        "Sa": sa,
        "GW": gw,
    })
    closure_df.to_csv(RUN_DIR / "one_basin_closure_timeseries.csv", index=False)

    neg_flux_count = 0
    neg_state_count = 0
    for key in ["rainfall_components", "snowfall_components", "snowmelt_components", "interception_evaporation_components",
                "actual_ET_components", "surface_runoff_components", "interflow_components",
                "recharge_to_groundwater_components", "baseflow_components", "Q_process_components"]:
        arr = diag_numpy(diag, key)
        neg_flux_count += int(np.sum(arr < NEG_TOL))
    for key in ["SNOWPACK_components", "MELTWATER_components", "Sa_components", "GW_components"]:
        arr = diag_numpy(diag, key)
        neg_state_count += int(np.sum(arr < NEG_TOL))

    summary = {
        "mean_abs_daily_residual_mm_day": float(np.nanmean(np.abs(residual))),
        "cumulative_relative_wb_error": safe_ratio(float(np.nansum(np.abs(residual))), float(np.nansum(p))),
        "negative_flux_count": neg_flux_count,
        "negative_state_count": neg_state_count,
    }
    pd.DataFrame([summary]).to_csv(RUN_DIR / "one_basin_closure_summary.csv", index=False)
    return summary


def compute_tables(model_name, prepared, eval_res):
    obs = prepared["obs_test"]
    pred = eval_res["pred"]
    mix = eval_res["mix"]
    comp = eval_res["comp"]
    weights = eval_res["weights"]
    metrics_rows, wb_rows = [], []

    for i, basin_id in enumerate(prepared["subset_ids"]):
        o = obs[i]
        s = pred[i]
        w = weights[i]
        p_comp = comp["precipitation_components"][i]
        residual_comp = comp["process_local_residual_components"][i]
        weighted_residual = np.sum(residual_comp * w[None, :], axis=1)
        cum_p = float(np.nansum(p_comp[:, 0]))
        q_process = mix["Q_process"][i]
        intv = mix["interception_evaporation"][i]
        et = mix["actual_ET"][i]
        srun = mix["surface_runoff"][i]
        iflow = mix["interflow"][i]
        bas = mix["baseflow"][i]
        rec = mix["recharge_to_groundwater"][i]
        sa = mix["Sa"][i]
        snow = mix["SNOWPACK"][i]
        melt = mix["MELTWATER"][i]
        gw = mix["GW"][i]
        alpha = mix["alpha"][i]
        a_srz = sa - np.nanmin(sa)

        metrics_rows.append({
            "model": model_name,
            "basin_id": basin_id,
            "NSE": calc_nse(o, s),
            "KGE": calc_kge(o, s),
            "R2": calc_r2(o, s),
            "FLV": calc_flv(o, s),
            "FHV": calc_fhv(o, s),
            "low_flow_NSE": lowflow_nse(o, s),
            "high_flow_NSE": highflow_nse(o, s),
            "ET_over_P": safe_ratio(float(np.nansum(et)), cum_p),
            "INT_over_P": safe_ratio(float(np.nansum(intv)), cum_p),
            "Q_over_P": safe_ratio(float(np.nansum(q_process)), cum_p),
            "SRUN_over_P": safe_ratio(float(np.nansum(srun)), cum_p),
            "IFLOW_over_P": safe_ratio(float(np.nansum(iflow)), cum_p),
            "BAS_over_P": safe_ratio(float(np.nansum(bas)), cum_p),
            "REC_over_P": safe_ratio(float(np.nansum(rec)), cum_p),
            "external_loss_over_P": 0.0,
            "mean_abs_daily_wb_residual_mm_day": float(np.nanmean(np.abs(weighted_residual))),
            "cumulative_relative_wb_error": safe_ratio(float(np.nansum(np.abs(weighted_residual))), cum_p),
            "SNOWPACK_drift_mm": float(snow[-1]),
            "MELTWATER_drift_mm": float(melt[-1]),
            "Sa_drift_mm": float(sa[-1]),
            "GW_drift_mm": float(gw[-1]),
            "alpha_mean": float(np.nanmean(alpha)),
            "theta_cap_mean": float(np.nanmean(mix["theta_cap"][i])),
            "aSrz_capacity_mm": float(np.nanmax(a_srz)),
        })
        wb_rows.append({
            "model": model_name,
            "basin_id": basin_id,
            "cumulative_precipitation_mm": cum_p,
            "mean_abs_daily_wb_residual_mm_day": float(np.nanmean(np.abs(weighted_residual))),
            "max_abs_daily_wb_residual_mm_day": float(np.nanmax(np.abs(weighted_residual))),
            "cumulative_relative_wb_error": safe_ratio(float(np.nansum(np.abs(weighted_residual))), cum_p),
            "external_loss_over_P": 0.0,
        })
    return pd.DataFrame(metrics_rows), pd.DataFrame(wb_rows)


def train_model(model, prepared, run_dir, start_epoch, end_epoch):
    loss_fun = crit.RmseLossComb(alpha=0.25)
    if torch.cuda.is_available():
        loss_fun = loss_fun.cuda()
    optim = torch.optim.Adadelta(model.parameters(), lr=LR)
    if start_epoch > 1 and (run_dir / f"optim_state_ep{start_epoch-1}.pt").exists():
        optim.load_state_dict(torch.load(run_dir / f"optim_state_ep{start_epoch-1}.pt", map_location="cpu"))
    model.zero_grad()
    with open(run_dir / "run.csv", "a" if start_epoch > 1 else "w") as rf:
        for i_epoch in range(start_epoch, end_epoch + 1):
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
            torch.save(model.state_dict(), run_dir / f"model_Ep{i_epoch}_state.pt")
            torch.save(optim.state_dict(), run_dir / f"optim_state_ep{i_epoch}.pt")


def make_worst10_plots(metrics_df, prepared, pred):
    ensure_dir(PLOTS_DIR / "worst10_nse_hydrographs")
    obs = prepared["obs_test"]
    worst = metrics_df.sort_values("NSE").head(10)
    for _, row in worst.iterrows():
        idx = prepared["subset_ids"].index(int(row["basin_id"]))
        plt.figure(figsize=(8, 3))
        plt.plot(obs[idx], label="Obs", linewidth=1.2)
        plt.plot(pred[idx], label="Sim", linewidth=1.2)
        plt.title(f"Basin {int(row['basin_id'])} NSE={row['NSE']:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "worst10_nse_hydrographs" / f"{int(row['basin_id'])}.png", dpi=150)
        plt.close()


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
    torch.backends.cudnn.benchmark = False

    prepared = prepare_data()
    (RUN_DIR / "demo32_basin_ids.txt").write_text("\n".join(str(x) for x in prepared["subset_ids"]) + "\n")

    base_state = torch.load(BASE_DEMO_CKPT, map_location="cpu")
    model = build_model()
    loaded, skipped = partial_load_from_asrz(base_state, model)
    (RUN_DIR / "load_report.json").write_text(json.dumps({"loaded": loaded, "skipped": skipped}, indent=2))
    if torch.cuda.is_available():
        model = model.cuda(GPU_ID)

    closure_summary = one_basin_closure_test(model, prepared)

    train_model(model, prepared, RUN_DIR, 1, EPOCHS_INITIAL)
    eval10 = evaluate_model_components(model, prepared)
    metrics10, wb10 = compute_tables("Model6Closed_Snow_aSrz_SIMHYD_Simple_ep10", prepared, eval10)
    promising = (metrics10["NSE"].median() > 0.40) and ((metrics10["cumulative_relative_wb_error"] > 0.01).sum() == 0)
    if promising:
        train_model(model, prepared, RUN_DIR, EPOCHS_INITIAL + 1, EPOCHS_TOTAL)

    final_epoch = EPOCHS_TOTAL if promising else EPOCHS_INITIAL
    eval_res = evaluate_model_components(model, prepared)
    metrics_df, wb_df = compute_tables("Model6Closed_Snow_aSrz_SIMHYD_Simple", prepared, eval_res)
    metrics_df.to_csv(RUN_DIR / "per_basin_metrics.csv", index=False)
    wb_df.to_csv(RUN_DIR / "water_balance.csv", index=False)
    make_worst10_plots(metrics_df, prepared, eval_res["pred"])

    summary_rows = []
    base_df = pd.read_csv(BASE_SUMMARY)
    for _, r in base_df.iterrows():
        summary_rows.append({
            "model": r["model"],
            "median_NSE": float(r["median_NSE"]),
            "median_KGE": float(r["median_KGE"]),
            "median_R2": np.nan,
            "median_FLV": float(r["median_FLV"]),
            "median_FHV": float(r["median_FHV"]),
            "median_low_flow_NSE": float(r["median_low_flow_NSE"]),
            "median_high_flow_NSE": float(r["median_high_flow_NSE"]),
            "NSE_lt_0_count": np.nan,
            "median_daily_wb_residual": float(r["median_weighted_process_closure_residual_mm_day"]),
            "median_cumulative_wb_error": float(r["median_weighted_cumulative_relative_wb_error"]),
            "basins_gt_1pct_wb_error": int(r["basins_gt_1pct_wb_error"]),
            "external_loss_over_P": float(r["total_external_losses_frac_of_p"]),
        })
    if HBV_DEMO32_SUMMARY.exists():
        hbv_df = pd.read_csv(HBV_DEMO32_SUMMARY)
        hbv_row = hbv_df[hbv_df["model"] == "HBV_Epoch10"].iloc[0]
        summary_rows.append({
            "model": "HBV_Epoch10",
            "median_NSE": float(hbv_row["median_NSE"]),
            "median_KGE": float(hbv_row["median_KGE"]),
            "median_R2": float(hbv_row["median_R2"]),
            "median_FLV": float(hbv_row["median_FLV"]),
            "median_FHV": float(hbv_row["median_FHV"]),
            "median_low_flow_NSE": float(hbv_row["median_low_flow_NSE"]),
            "median_high_flow_NSE": float(hbv_row["median_high_flow_NSE"]),
            "NSE_lt_0_count": np.nan,
            "median_daily_wb_residual": float(hbv_row["median_mean_abs_residual_mm_day"]),
            "median_cumulative_wb_error": float(hbv_row["median_relative_error"]),
            "basins_gt_1pct_wb_error": int(hbv_row["basins_gt_1pct_relative_error"]),
            "external_loss_over_P": 0.0,
        })

    summary_rows.append({
        "model": "Model6Closed_Snow_aSrz_SIMHYD_Simple",
        "median_NSE": float(metrics_df["NSE"].median()),
        "median_KGE": float(metrics_df["KGE"].median()),
        "median_R2": float(metrics_df["R2"].median()),
        "median_FLV": float(metrics_df["FLV"].median()),
        "median_FHV": float(metrics_df["FHV"].median()),
        "median_low_flow_NSE": float(metrics_df["low_flow_NSE"].median()),
        "median_high_flow_NSE": float(metrics_df["high_flow_NSE"].median()),
        "NSE_lt_0_count": int((metrics_df["NSE"] < 0).sum()),
        "median_daily_wb_residual": float(metrics_df["mean_abs_daily_wb_residual_mm_day"].median()),
        "median_cumulative_wb_error": float(metrics_df["cumulative_relative_wb_error"].median()),
        "basins_gt_1pct_wb_error": int((metrics_df["cumulative_relative_wb_error"] > 0.01).sum()),
        "external_loss_over_P": 0.0,
    })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RUN_DIR / "summary_compare.csv", index=False)

    lines = [
        "Model6Closed_Snow_aSrz_SIMHYD_Simple demo32",
        f"Final trained epochs: {final_epoch}",
        f"Promising after ep10: {promising}",
        "",
        "One-basin pretraining closure test:",
        f"- mean abs daily residual: {closure_summary['mean_abs_daily_residual_mm_day']:.8f} mm/day",
        f"- cumulative relative WB error: {closure_summary['cumulative_relative_wb_error']:.8e}",
        f"- negative flux count: {closure_summary['negative_flux_count']}",
        f"- negative state count: {closure_summary['negative_state_count']}",
        "",
    ]
    final_row = summary_df[summary_df["model"] == "Model6Closed_Snow_aSrz_SIMHYD_Simple"].iloc[0]
    lines.extend([
        "Final model:",
        f"- median NSE: {final_row['median_NSE']:.4f}",
        f"- median KGE: {final_row['median_KGE']:.4f}",
        f"- median R2: {final_row['median_R2']:.4f}",
        f"- median FLV: {final_row['median_FLV']:.4f}",
        f"- median FHV: {final_row['median_FHV']:.4f}",
        f"- median low-flow NSE: {final_row['median_low_flow_NSE']:.4f}",
        f"- median high-flow NSE: {final_row['median_high_flow_NSE']:.4f}",
        f"- NSE < 0 count: {int(final_row['NSE_lt_0_count'])}",
        f"- median daily WB residual: {final_row['median_daily_wb_residual']:.8f} mm/day",
        f"- median cumulative WB error: {final_row['median_cumulative_wb_error']:.8f}",
        f"- basins >1% WB error: {int(final_row['basins_gt_1pct_wb_error'])}",
        f"- external_loss/P: {final_row['external_loss_over_P']:.6f}",
        f"- median Q/P: {metrics_df['Q_over_P'].median():.4f}",
        f"- median INT/P: {metrics_df['INT_over_P'].median():.4f}",
        f"- median ET_a/P: {metrics_df['ET_over_P'].median():.4f}",
        f"- median SRUN/P: {metrics_df['SRUN_over_P'].median():.4f}",
        f"- median IFLOW/P: {metrics_df['IFLOW_over_P'].median():.4f}",
        f"- median BAS/P: {metrics_df['BAS_over_P'].median():.4f}",
        f"- median REC/P: {metrics_df['REC_over_P'].median():.4f}",
        f"- median SNOWPACK drift: {metrics_df['SNOWPACK_drift_mm'].median():.3f} mm",
        f"- median MELTWATER drift: {metrics_df['MELTWATER_drift_mm'].median():.3f} mm",
        f"- median Sa drift: {metrics_df['Sa_drift_mm'].median():.3f} mm",
        f"- median GW drift: {metrics_df['GW_drift_mm'].median():.3f} mm",
        f"- median alpha mean: {metrics_df['alpha_mean'].median():.4f}",
        f"- median theta_cap: {metrics_df['theta_cap_mean'].median():.3f} mm",
        f"- median aSrz capacity: {metrics_df['aSrz_capacity_mm'].median():.3f} mm",
    ])
    (RUN_DIR / "final_result.txt").write_text("\n".join(lines) + "\n")
    (RUN_DIR / "run_model6_closed_snow_asrz_simhyd_simple_demo32.py").write_text(Path(__file__).read_text())
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
