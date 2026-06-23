from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(os.environ.get("MODEL6_PUBLICATION_ROOT", str(Path(__file__).resolve().parents[3]))).resolve()
DATA_ROOT = Path(os.environ.get("MODEL6_PUBLICATION_DATA_ROOT", "/home/mircore/Desktop/diff_hydro")).resolve()
SOURCE_ROOT = REPO_ROOT / "model6" / "source"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "hbv_module"))

from hydro_ml.diagnosis import calc_fhv, calc_flv, calc_kge, calc_nse, highflow_nse, lowflow_nse  # noqa: E402
from hydroDL import utils  # noqa: E402
from hydroDL.data import camels  # noqa: E402
from hydroDL.model import crit, rnn, train  # noqa: E402

ATTR_LST = [
    "p_mean", "pet_mean", "p_seasonality", "frac_snow", "aridity", "high_prec_freq", "high_prec_dur",
    "low_prec_freq", "low_prec_dur", "elev_mean", "slope_mean", "area_gages2", "frac_forest", "lai_max",
    "lai_diff", "gvf_max", "gvf_diff", "dom_land_cover_frac", "dom_land_cover", "root_depth_50",
    "soil_depth_pelletier", "soil_depth_statsgo", "soil_porosity", "soil_conductivity",
    "max_water_content", "sand_frac", "silt_frac", "clay_frac", "geol_1st_class", "glim_1st_class_frac",
    "geol_2nd_class", "glim_2nd_class_frac", "carbonate_rocks_frac", "geol_porostiy", "geol_permeability",
]


T_TRAIN = [19801001, 19951001]
T_TEST = [19951001, 20101001]
T_INV = [19801001, 19951001]
FORCING = "daymet"
VAR_F = ["prcp", "tmean"]
SEED = 111111
BUFFTIME = 365
RHO = 365
GPU_ID = 1 if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 0
HIDDEN_SIZE = 64
NMUL = 4
NEG_TOL = -1e-6

PROJECT_DIR = REPO_ROOT / "model6" / "results" / "train_runs"
BASELINE_RUN = DATA_ROOT / "outputs" / "rnnStreamflow" / "CAMELSMODELSIX" / "DynamicSimHydModelSix" / "AllBasins" / FORCING / str(SEED) / (
    "T_19801001_19951001_BS_32_HS_64_RHO_365_Buff_365_Mul_4_Route_1_CmpW_1_LGDyn_1_DSQ_1_DETGAM_1_DPART_1_DCFMAX_1_DROUTE_0_CRoute_1_DryCh_1_ZGate_1_MaxIter200_All671_BS32_HS64_MaxIter200"
)
DEMO32_LIST = DATA_ROOT / "ECO_HYBRID" / "Model6Physical_aSrz_Minimal_demo32" / "demo32_basin_ids.txt"
LAI_GAPFILLED_FILE = DATA_ROOT / "ECO_HYBRID" / "NOAA_LAI_CDR_671_DAILY" / "lai_cdr_daily_basin_671_gapfilled_rf.csv"
HBV_WARM_CKPT = DATA_ROOT / "ECO_HYBRID" / "Model6_aSrz_HBVOnly_demo32" / "model_Ep10_state.pt"
CARAVAN_TS_ROOT = DATA_ROOT / "global_data" / "Caravan_v1_5" / "extracted" / "usr" / "local" / "google" / "home" / "kratzert" / "Data" / "Caravan-Jan25-csv" / "timeseries" / "csv" / "camels"

PET_CANDIDATES = [
    "potential_evaporation_sum_ERA5_LAND",
    "potential_evaporation_sum_FAO_PENMAN_MONTEITH",
    "potential_evaporation_sum",
]
SNOW_FRAC_IDX = ATTR_LST.index("frac_snow")
LAI_MAX_IDX = ATTR_LST.index("lai_max")
LAI_DIFF_IDX = ATTR_LST.index("lai_diff")
SLOPE_IDX = ATTR_LST.index("slope_mean")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def t_range_dates(t_range: list[int]) -> pd.DatetimeIndex:
    return pd.to_datetime(utils.time.tRange2Array(t_range).astype(str))


def seasonal_features(t_range: list[int]) -> np.ndarray:
    dates = t_range_dates(t_range)
    doy = dates.dayofyear.to_numpy(dtype=np.float32)
    ang = 2.0 * np.pi * (doy - 1.0) / 365.0
    return np.stack([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)


def choose_pet_column(columns: list[str]) -> str:
    for col in PET_CANDIDATES:
        if col in columns:
            return col
    raise KeyError(f"No PET column found among {PET_CANDIDATES}")


def logit_from_frac(frac: float) -> float:
    frac = min(max(float(frac), 1e-4), 1.0 - 1e-4)
    return float(np.log(frac / (1.0 - frac)))


def logit_from_value(value: float, low: float, high: float) -> float:
    return logit_from_frac((value - low) / (high - low))


def load_pet_full(root_db: Path, gage_ids: list[int], forcing: str) -> tuple[np.ndarray, np.ndarray]:
    var_lst_nl = ["PEVAP"]
    t_pet_range = [19800101, 20150101] if forcing != "maurer" else [19800101, 20090101]
    t_pet_lst = utils.time.tRange2Array(t_pet_range)
    pet_dir = str(root_db) + "/pet_harg/" + forcing + "/"
    ntime = len(t_pet_lst)
    pet_full = np.empty([len(gage_ids), ntime, len(var_lst_nl)], dtype=np.float32)
    for k, gid in enumerate(gage_ids):
        pet_full[k, :, :] = camels.readcsvGage(pet_dir, gid, var_lst_nl, ntime)
    return pet_full, t_pet_lst


def load_caravan_rn(gauge_id: str, dates: pd.DatetimeIndex) -> np.ndarray:
    df = pd.read_csv(
        CARAVAN_TS_ROOT / f"{gauge_id}.csv",
        usecols=["date", "surface_net_solar_radiation_mean", "surface_net_thermal_radiation_mean"],
        parse_dates=["date"],
    )
    # Caravan stores these radiation variables as daily mean fluxes in W m-2.
    # Convert net radiation into MJ m-2 d-1 for the ecohydrology core.
    df["rn_mj_m2_d"] = (
        df["surface_net_solar_radiation_mean"].astype(np.float32)
        + df["surface_net_thermal_radiation_mean"].astype(np.float32)
    ) * 0.0864
    ser = df.set_index("date")["rn_mj_m2_d"].reindex(dates)
    if ser.isna().any():
        ser = ser.interpolate(method="time", limit_direction="both")
    return ser.to_numpy(dtype=np.float32)[:, None]


def build_lai_daily_for_basins(
    basin_ids: list[int],
    dates: pd.DatetimeIndex,
    attrs_raw: np.ndarray,
    daily_file: Path,
) -> tuple[np.ndarray, dict[str, float]]:
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.to_datetime(np.asarray(dates).astype(str))
    fallback = build_static_lai_fallback(attrs_raw, dates)
    try:
        daily = pd.read_csv(daily_file, usecols=["basin_id", "date", "lai_mean"])
    except pd.errors.ParserError:
        daily = pd.read_csv(daily_file, usecols=["basin_id", "date", "lai_mean"], engine="python")
    daily["basin_id"] = daily["basin_id"].astype(int)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily[daily["basin_id"].isin(basin_ids)].copy()
    if daily.empty:
        return fallback, {
            "daily_file": str(daily_file),
            "exact_fraction": 0.0,
            "climatology_fraction": 0.0,
            "static_fraction": 1.0,
        }

    exact = daily.pivot_table(index="date", columns="basin_id", values="lai_mean", aggfunc="mean")
    clim = daily.assign(doy=daily["date"].dt.dayofyear).groupby(["doy", "basin_id"])["lai_mean"].mean().unstack("basin_id")

    out = fallback.copy()
    exact_count = 0
    clim_count = 0
    total = len(basin_ids) * len(dates)
    for i, basin_id in enumerate(basin_ids):
        exact_series = exact[basin_id].reindex(dates) if basin_id in exact.columns else pd.Series(index=dates, dtype=float)
        doy_idx = dates.dayofyear
        clim_vals = clim[basin_id].reindex(doy_idx).to_numpy(dtype=np.float32) if basin_id in clim.columns else np.full(len(dates), np.nan, dtype=np.float32)
        arr = fallback[i, :, 0].copy()
        exact_vals = exact_series.to_numpy(dtype=np.float32)
        exact_mask = np.isfinite(exact_vals)
        arr[exact_mask] = exact_vals[exact_mask]
        exact_count += int(exact_mask.sum())
        clim_mask = (~exact_mask) & np.isfinite(clim_vals)
        arr[clim_mask] = clim_vals[clim_mask]
        clim_count += int(clim_mask.sum())
        out[i, :, 0] = arr
    return out.astype(np.float32), {
        "daily_file": str(daily_file),
        "exact_fraction": float(exact_count / max(total, 1)),
        "climatology_fraction": float(clim_count / max(total, 1)),
        "static_fraction": float(1.0 - (exact_count + clim_count) / max(total, 1)),
    }


def partial_load(source_state: dict[str, torch.Tensor], target_model: torch.nn.Module) -> dict[str, list[str]]:
    tgt = target_model.state_dict()
    loaded: list[str] = []
    skipped: list[str] = []
    for key, value in source_state.items():
        if key in tgt and tuple(tgt[key].shape) == tuple(value.shape):
            tgt[key] = value.clone()
            loaded.append(key)
        else:
            skipped.append(key)
    target_model.load_state_dict(tgt, strict=False)
    return {"loaded": loaded, "skipped": skipped}


def apply_ecohbv_prior_init(model: torch.nn.Module, theta_cap_upper: float) -> dict[str, object]:
    priors = [
        ("TT", 0.0, -2.5, 2.5),
        ("CFMAX", 3.0, 0.5, 10.0),
        ("CWH", 0.05, 0.0, 0.2),
        ("CFR", 0.03, 0.0, 0.1),
        ("theta_ab", 0.75, 0.5, 1.0),
        ("theta_ak", 3.0, 1.0, 10.0),
        ("theta_capb", 250.0, 0.0, theta_cap_upper),
        ("theta_efmax", 0.80, 0.5, 1.0),
        ("theta_wetpoint", 0.60, 0.3, 0.9),
        ("theta_veg", 0.35, 0.05, 0.95),
        ("K0", 0.10, 0.01, 0.80),
        ("K1", 0.02, 0.001, 0.30),
        ("K2", 0.003, 0.0001, 0.15),
        ("UZL", 40.0, 0.0, 150.0),
        ("PERC_MAX", 1.5, 0.0, 10.0),
        ("cap_max", 1.0, 0.0, 8.0),
        ("cap_shape", 1.2, 0.1, 5.0),
    ]

    component_offsets = np.array([-0.35, -0.10, 0.10, 0.35], dtype=np.float32)
    if model.nmul != len(component_offsets):
        component_offsets = np.linspace(-0.3, 0.3, model.nmul, dtype=np.float32)
    route_a_bias = logit_from_frac(0.30)
    route_b_bias = logit_from_frac(0.25)

    with torch.no_grad():
        model.staticOut.weight.zero_()
        model.staticOut.bias.zero_()
        model.compWeightBias.zero_()
        model.compStaticBias.zero_()

        for feat_idx, (_, value, low, high) in enumerate(priors):
            base_bias = logit_from_value(value, low, high)
            row = slice(feat_idx * model.nmul, (feat_idx + 1) * model.nmul)
            model.staticOut.bias[row] = torch.as_tensor(
                base_bias + component_offsets,
                dtype=model.staticOut.bias.dtype,
                device=model.staticOut.bias.device,
            )

        route_start = model.nstaticpm
        for comp_idx in range(model.nmul):
            model.staticOut.bias[route_start + 2 * comp_idx] = route_a_bias
            model.staticOut.bias[route_start + 2 * comp_idx + 1] = route_b_bias

    return {
        "hydro_priors": {name: value for name, value, _, _ in priors},
        "component_offsets": component_offsets.tolist(),
        "route_a_frac": 0.30,
        "route_b_frac": 0.25,
    }


def to_device(*args):
    if torch.cuda.is_available():
        return [x.cuda(GPU_ID) for x in args]
    return list(args)


def build_model(args) -> torch.nn.Module:
    ninv = 3 + len(ATTR_LST)
    model = rnn.MultiInv_DynamicECOHBVaSrz_HBV11b(
        ninv=ninv,
        nmul=NMUL,
        nattr=len(ATTR_LST),
        hiddeninv=HIDDEN_SIZE,
        inittime=BUFFTIME,
        routOpt=True,
        comprout=False,
        compwts=True,
        lgdyn=False,
        component_routing=True,
        dry_channel_loss=False,
        zero_flow_gate=False,
        theta_cap_mode=args.theta_cap_mode,
        veg_function=args.veg_function,
        theta_cap_upper=float(args.theta_cap_upper),
        drift_reg_weight=float(args.drift_reg_weight),
    )
    if torch.cuda.is_available():
        model = model.cuda(GPU_ID)
    return model


def prepare_data(args) -> dict[str, np.ndarray | list[int] | dict[str, float]]:
    root_db = DATA_ROOT / "Camels"
    with open(BASELINE_RUN / "statDict.json", "r", encoding="utf-8") as fp:
        stat_dict = json.load(fp)

    subset_ids = [int(x.strip()) for x in DEMO32_LIST.read_text().splitlines() if x.strip()]
    camels.initcamels(str(root_db))
    gageinfo = camels.gageDict
    basin_ids_all = gageinfo["id"].tolist()
    subset_idx = [basin_ids_all.index(gid) for gid in subset_ids]
    gauge_ids = [f"camels_{gid:08d}" for gid in subset_ids]
    areas = gageinfo["area"][subset_idx]

    df_train = camels.DataframeCamels(tRange=T_TRAIN, subset=subset_ids, forType=FORCING)
    forc_train = df_train.getDataTs(varLst=VAR_F, doNorm=False, rmNan=False).astype(np.float32)
    obs_train = df_train.getDataObs(doNorm=False, rmNan=False, basinnorm=False).astype(np.float32)
    temp_area = np.tile(areas[:, None, None], (1, obs_train.shape[1], 1))
    obs_train = (obs_train * 0.0283168 * 3600 * 24) / (temp_area * 1e6) * 1e3

    df_inv = camels.DataframeCamels(tRange=T_INV, subset=subset_ids, forType=FORCING)
    forc_inv = df_inv.getDataTs(varLst=VAR_F, doNorm=False, rmNan=False).astype(np.float32)
    attrs_raw = df_inv.getDataConst(varLst=ATTR_LST, doNorm=False, rmNan=False).astype(np.float32)

    pet_full, t_pet_lst = load_pet_full(root_db, gageinfo["id"], FORCING)
    train_dates = t_range_dates(T_TRAIN)
    test_dates = t_range_dates(T_TEST)
    t_train_arr = utils.time.tRange2Array(T_TRAIN)
    t_test_arr = utils.time.tRange2Array(T_TEST)
    t_inv_arr = utils.time.tRange2Array(T_INV)
    _, _, ind_pet_train = np.intersect1d(t_train_arr, t_pet_lst, return_indices=True)
    _, _, ind_pet_inv = np.intersect1d(t_inv_arr, t_pet_lst, return_indices=True)
    _, _, ind_pet_test = np.intersect1d(t_test_arr, t_pet_lst, return_indices=True)
    pet_train = pet_full[:, ind_pet_train, :][subset_idx, :, :]
    pet_inv = pet_full[:, ind_pet_inv, :][subset_idx, :, :]
    pet_test = pet_full[:, ind_pet_test, :][subset_idx, :, :]

    rn_train = np.stack([load_caravan_rn(gauge_id, train_dates)[:, 0] for gauge_id in gauge_ids], axis=0)[..., None].astype(np.float32)
    rn_test = np.stack([load_caravan_rn(gauge_id, test_dates)[:, 0] for gauge_id in gauge_ids], axis=0)[..., None].astype(np.float32)

    lai_train, lai_train_info = build_lai_daily_for_basins(subset_ids, train_dates, attrs_raw, LAI_GAPFILLED_FILE)
    lai_test, lai_test_info = build_lai_daily_for_basins(subset_ids, test_dates, attrs_raw, LAI_GAPFILLED_FILE)

    season_train = np.tile(seasonal_features(T_TRAIN)[None, :, :], (len(subset_ids), 1, 1))
    season_test = np.tile(seasonal_features(T_TEST)[None, :, :], (len(subset_ids), 1, 1))
    x_train = np.concatenate([forc_train, pet_train, season_train, rn_train, lai_train], axis=2).astype(np.float32)

    df_test = camels.DataframeCamels(tRange=T_TEST, subset=subset_ids, forType=FORCING)
    forc_test = df_test.getDataTs(varLst=VAR_F, doNorm=False, rmNan=False).astype(np.float32)
    obs_test = df_test.getDataObs(doNorm=False, rmNan=False, basinnorm=False).astype(np.float32)
    temp_area_test = np.tile(areas[:, None, None], (1, obs_test.shape[1], 1))
    obs_test = (obs_test * 0.0283168 * 3600 * 24) / (temp_area_test * 1e6) * 1e3
    obs_test = obs_test[:, :, 0].astype(np.float32)

    x_test = np.concatenate([forc_test, pet_test, season_test, rn_test, lai_test], axis=2).astype(np.float32)
    x_train[np.isnan(x_train)] = 0.0
    x_test[np.isnan(x_test)] = 0.0
    x_eval = np.concatenate([x_train, x_test], axis=1).astype(np.float32)

    series_inv = np.concatenate([forc_inv, pet_inv], axis=2)
    series_test = np.concatenate([forc_test, pet_test], axis=2)
    series_eval = np.concatenate([series_inv, series_test], axis=1)
    attr_norm = camels.transNormbyDic(attrs_raw, ATTR_LST, stat_dict, toNorm=True).astype(np.float32)
    attr_norm[np.isnan(attr_norm)] = 0.0
    series_norm_train = camels.transNormbyDic(series_inv, VAR_F + ["pet"], stat_dict, toNorm=True).astype(np.float32)
    series_norm_eval = camels.transNormbyDic(series_eval, VAR_F + ["pet"], stat_dict, toNorm=True).astype(np.float32)
    series_norm_train[np.isnan(series_norm_train)] = 0.0
    series_norm_eval[np.isnan(series_norm_eval)] = 0.0

    snow_frac_raw = attrs_raw[:, SNOW_FRAC_IDX:SNOW_FRAC_IDX + 1].astype(np.float32)
    mean_lai_raw = np.clip(attrs_raw[:, LAI_MAX_IDX:LAI_MAX_IDX + 1] - 0.5 * attrs_raw[:, LAI_DIFF_IDX:LAI_DIFF_IDX + 1], 0.0, None).astype(np.float32)
    slope_raw = np.nan_to_num(attrs_raw[:, SLOPE_IDX:SLOPE_IDX + 1], nan=0.0).astype(np.float32)

    snow_train = np.repeat(snow_frac_raw[:, None, :], series_norm_train.shape[1], axis=1)
    mean_lai_train = np.repeat(mean_lai_raw[:, None, :], series_norm_train.shape[1], axis=1)
    slope_train = np.repeat(slope_raw[:, None, :], series_norm_train.shape[1], axis=1)
    z_train = np.concatenate([series_norm_train, snow_train, mean_lai_train, slope_train], axis=2).astype(np.float32)

    snow_eval = np.repeat(snow_frac_raw[:, None, :], series_norm_eval.shape[1], axis=1)
    mean_lai_eval = np.repeat(mean_lai_raw[:, None, :], series_norm_eval.shape[1], axis=1)
    slope_eval = np.repeat(slope_raw[:, None, :], series_norm_eval.shape[1], axis=1)
    attr_ts_eval = np.repeat(attr_norm[:, None, :], series_norm_eval.shape[1], axis=1)
    z_eval = np.concatenate([series_norm_eval, snow_eval, mean_lai_eval, slope_eval, attr_ts_eval], axis=2).astype(np.float32)

    return {
        "subset_ids": subset_ids,
        "gauge_ids": gauge_ids,
        "x_train": x_train.astype(np.float32),
        "y_train": obs_train.astype(np.float32),
        "z_train": z_train.astype(np.float32),
        "attr_norm": attr_norm.astype(np.float32),
        "x_eval": x_eval.astype(np.float32),
        "z_eval": z_eval.astype(np.float32),
        "obs_test": obs_test.astype(np.float32),
        "lai_info": {"train": lai_train_info, "test": lai_test_info},
    }


def evaluate_period_metrics(obs: np.ndarray, sim: np.ndarray) -> pd.DataFrame:
    rows = []
    for i in range(obs.shape[0]):
        o = obs[i]
        s = sim[i]
        rows.append(
            {
                "NSE": calc_nse(o, s),
                "KGE": calc_kge(o, s),
                "FLV": calc_flv(o, s),
                "FHV": calc_fhv(o, s),
                "low_flow_NSE": lowflow_nse(o, s),
                "high_flow_NSE": highflow_nse(o, s),
            }
        )
    return pd.DataFrame(rows)


def evaluate_train(model: torch.nn.Module, prepared: dict) -> tuple[pd.DataFrame, np.ndarray]:
    old_inittime = model.inittime
    old_training = model.training
    model.inittime = BUFFTIME
    model.train(False)
    x = torch.from_numpy(np.swapaxes(prepared["x_train"], 1, 0)).float()
    z = torch.from_numpy(np.swapaxes(np.concatenate([prepared["z_train"], np.repeat(prepared["attr_norm"][:, None, :], prepared["z_train"].shape[1], axis=1)], axis=2), 1, 0)).float()
    x, z = to_device(x, z)
    with torch.no_grad():
        q = model(x, z)
    pred = q.detach().cpu().numpy()[:, :, 0].T
    obs = prepared["y_train"][:, BUFFTIME:, 0]
    metrics = evaluate_period_metrics(obs, pred)
    model.inittime = old_inittime
    model.train(old_training)
    return metrics, pred


def evaluate_test(model: torch.nn.Module, prepared: dict) -> tuple[pd.DataFrame, np.ndarray]:
    old_inittime = model.inittime
    old_training = model.training
    model.inittime = len(utils.time.tRange2Array(T_TRAIN))
    model.train(False)
    x = torch.from_numpy(np.swapaxes(prepared["x_eval"], 1, 0)).float()
    z = torch.from_numpy(np.swapaxes(prepared["z_eval"], 1, 0)).float()
    x, z = to_device(x, z)
    with torch.no_grad():
        q = model(x, z)
    pred = q.detach().cpu().numpy()[:, :, 0].T
    obs = prepared["obs_test"]
    metrics = evaluate_period_metrics(obs, pred)
    model.inittime = old_inittime
    model.train(old_training)
    return metrics, pred


def train_model(model: torch.nn.Module, prepared: dict, run_dir: Path, args) -> dict[str, float]:
    if args.loss_type == "rmse":
        loss_fun = crit.RmseLossComb(alpha=0.25)
    elif args.loss_type == "nse":
        loss_fun = crit.NSELoss()
    else:
        raise ValueError(f"Unsupported loss type: {args.loss_type}")
    if torch.cuda.is_available():
        loss_fun = loss_fun.cuda()
    optim = torch.optim.Adadelta(model.parameters(), lr=float(args.lr))
    epoch_rows = []
    best_train_median = -np.inf
    best_epoch = 0
    model.zero_grad()

    for epoch in range(1, int(args.epochs) + 1):
        model.train(True)
        loss_ep = 0.0
        t0 = time.time()
        for _ in range(int(args.max_iter_ep)):
            i_grid, i_t = train.randomIndex(
                len(prepared["subset_ids"]),
                prepared["x_train"].shape[1],
                [int(args.batch_size), int(args.rho)],
                bufftime=int(args.bufftime),
            )
            x_batch = train.selectSubset(prepared["x_train"], i_grid, i_t, int(args.rho), bufftime=int(args.bufftime))
            y_batch = train.selectSubset(prepared["y_train"], i_grid, i_t, int(args.rho))
            z_batch = train.selectSubset(
                prepared["z_train"],
                i_grid,
                i_t,
                int(args.rho),
                c=prepared["attr_norm"],
                bufftime=int(args.bufftime),
            )
            y_pred = model(x_batch, z_batch)
            loss = loss_fun(y_pred, y_batch)
            if hasattr(model, "get_auxiliary_loss"):
                aux = model.get_auxiliary_loss()
                if aux is not None:
                    loss = loss + aux
            loss.backward()
            optim.step()
            model.zero_grad()
            loss_ep += float(loss.item())
        loss_ep /= max(int(args.max_iter_ep), 1)

        train_metrics, _ = evaluate_train(model, prepared)
        test_metrics, _ = evaluate_test(model, prepared)
        train_median_nse = float(train_metrics["NSE"].median())
        test_median_nse = float(test_metrics["NSE"].median())
        train_mean_nse = float(train_metrics["NSE"].mean())
        test_mean_nse = float(test_metrics["NSE"].mean())
        row = {
            "epoch": epoch,
            "loss": loss_ep,
            "train_median_nse": train_median_nse,
            "train_mean_nse": train_mean_nse,
            "test_median_nse": test_median_nse,
            "test_mean_nse": test_mean_nse,
            "seconds": time.time() - t0,
        }
        epoch_rows.append(row)
        pd.DataFrame(epoch_rows).to_csv(run_dir / "epoch_metrics.csv", index=False)
        print(json.dumps(row), flush=True)
        torch.save(model.state_dict(), run_dir / f"model_ep{epoch}.pt")

        if train_median_nse > best_train_median:
            best_train_median = train_median_nse
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "best_model_train_nse.pt")

        if train_median_nse >= float(args.target_train_nse):
            break

    return {"best_epoch": best_epoch, "best_train_median_nse": best_train_median}


def summarize_run(model: torch.nn.Module, prepared: dict, run_dir: Path, args, train_state: dict[str, float]) -> None:
    model.load_state_dict(torch.load(run_dir / "best_model_train_nse.pt", map_location="cpu"))
    if torch.cuda.is_available():
        model = model.cuda(GPU_ID)
    train_metrics, train_pred = evaluate_train(model, prepared)
    test_metrics, test_pred = evaluate_test(model, prepared)

    train_df = pd.DataFrame(
        {
            "basin_id": prepared["subset_ids"],
            "gauge_id": prepared["gauge_ids"],
            "train_NSE": train_metrics["NSE"],
            "train_KGE": train_metrics["KGE"],
            "train_FLV": train_metrics["FLV"],
            "train_FHV": train_metrics["FHV"],
            "train_low_flow_NSE": train_metrics["low_flow_NSE"],
            "train_high_flow_NSE": train_metrics["high_flow_NSE"],
        }
    )
    test_df = pd.DataFrame(
        {
            "basin_id": prepared["subset_ids"],
            "gauge_id": prepared["gauge_ids"],
            "test_NSE": test_metrics["NSE"],
            "test_KGE": test_metrics["KGE"],
            "test_FLV": test_metrics["FLV"],
            "test_FHV": test_metrics["FHV"],
            "test_low_flow_NSE": test_metrics["low_flow_NSE"],
            "test_high_flow_NSE": test_metrics["high_flow_NSE"],
        }
    )
    merged = train_df.merge(test_df, on=["basin_id", "gauge_id"], how="inner")
    merged.to_csv(run_dir / "per_basin_metrics.csv", index=False)

    train_dates = t_range_dates(T_TRAIN)[BUFFTIME:]
    test_dates = t_range_dates(T_TEST)
    basin0 = 0
    pd.DataFrame(
        {
            "date": train_dates,
            "basin_id": prepared["subset_ids"][basin0],
            "gauge_id": prepared["gauge_ids"][basin0],
            "Q_obs_train": prepared["y_train"][basin0, BUFFTIME:, 0],
            "Q_sim_train": train_pred[basin0],
        }
    ).to_parquet(run_dir / "basin0_train_daily.parquet", index=False)
    pd.DataFrame(
        {
            "date": test_dates,
            "basin_id": prepared["subset_ids"][basin0],
            "gauge_id": prepared["gauge_ids"][basin0],
            "Q_obs_test": prepared["obs_test"][basin0],
            "Q_sim_test": test_pred[basin0],
        }
    ).to_parquet(run_dir / "basin0_test_daily.parquet", index=False)

    summary = {
        "run_name": args.run_name,
        "theta_cap_mode": args.theta_cap_mode,
        "veg_function": args.veg_function,
        "theta_cap_upper": float(args.theta_cap_upper),
        "drift_reg_weight": float(args.drift_reg_weight),
        "warm_start": str(args.warm_ckpt) if args.warm_ckpt else None,
        "best_epoch": int(train_state["best_epoch"]),
        "best_train_median_nse": float(train_state["best_train_median_nse"]),
        "train_median_nse": float(train_metrics["NSE"].median()),
        "train_mean_nse": float(train_metrics["NSE"].mean()),
        "train_median_kge": float(train_metrics["KGE"].median()),
        "test_median_nse": float(test_metrics["NSE"].median()),
        "test_mean_nse": float(test_metrics["NSE"].mean()),
        "test_median_kge": float(test_metrics["KGE"].median()),
        "train_nse_gt_0_7_count": int((train_metrics["NSE"] >= 0.7).sum()),
        "test_nse_gt_0_count": int((test_metrics["NSE"] > 0.0).sum()),
        "lai_info": prepared["lai_info"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed ECOHBV demo32 classical trainer with gap-filled LAI.")
    parser.add_argument("--run-name", type=str, default="ECOHBV_aSrz_HBV11b_demo32_gapfill_fixed")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-iter-ep", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rho", type=int, default=365)
    parser.add_argument("--bufftime", type=int, default=365)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--loss-type", choices=["rmse", "nse"], default="rmse")
    parser.add_argument("--target-train-nse", type=float, default=0.7)
    parser.add_argument("--theta-cap-mode", choices=["stocker", "direct"], default="direct")
    parser.add_argument("--veg-function", choices=["exp", "michaelis"], default="exp")
    parser.add_argument("--theta-cap-upper", type=float, default=1000.0)
    parser.add_argument("--drift-reg-weight", type=float, default=1e-3)
    parser.add_argument("--warm-ckpt", type=str, default=str(HBV_WARM_CKPT))
    parser.add_argument("--skip-prior-init", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_dir = PROJECT_DIR / args.run_name
    ensure_dir(run_dir)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.set_device(GPU_ID)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    prepared = prepare_data(args)
    (run_dir / "demo32_basin_ids.txt").write_text("\n".join(str(x) for x in prepared["subset_ids"]) + "\n")
    (run_dir / "lai_input_info.json").write_text(json.dumps(prepared["lai_info"], indent=2))

    model = build_model(args)
    load_report = {"mode": "random_init", "loaded": [], "skipped": []}
    if args.warm_ckpt and Path(args.warm_ckpt).exists():
        state = torch.load(args.warm_ckpt, map_location="cpu")
        partial = partial_load(state, model)
        load_report = {"mode": "partial", "source": args.warm_ckpt, **partial}
    prior_report = None
    if not args.skip_prior_init and "staticOut.bias" not in load_report["loaded"]:
        prior_report = apply_ecohbv_prior_init(model, float(args.theta_cap_upper))
        load_report["prior_init"] = prior_report
    (run_dir / "load_report.json").write_text(json.dumps(load_report, indent=2))

    train_state = train_model(model, prepared, run_dir, args)
    summarize_run(model, prepared, run_dir, args, train_state)


if __name__ == "__main__":
    main()
def build_static_lai_fallback(attrs_raw: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    lai_max = np.nan_to_num(attrs_raw[:, LAI_MAX_IDX], nan=2.0, posinf=2.0, neginf=0.5).astype(np.float32)
    lai_diff = np.nan_to_num(attrs_raw[:, LAI_DIFF_IDX], nan=0.5, posinf=0.5, neginf=0.1).astype(np.float32)
    lai_min = np.clip(lai_max - lai_diff, 0.05, None)
    doy = dates.dayofyear.to_numpy(dtype=np.float32)
    phase = 2.0 * np.pi * (doy - 1.0) / 365.0
    seasonal = 0.5 * (1.0 + np.sin(phase - np.pi / 2.0))
    lai = lai_min[:, None] + (lai_max - lai_min)[:, None] * seasonal[None, :]
    return lai[:, :, None].astype(np.float32)
