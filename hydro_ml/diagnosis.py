import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calc_nse(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 2:
        return np.nan
    o = obs[mask]
    s = sim[mask]
    den = np.sum((o - np.mean(o)) ** 2)
    if den <= 0:
        return np.nan
    return 1.0 - np.sum((o - s) ** 2) / den


def calc_kge(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 2:
        return np.nan
    o = obs[mask]
    s = sim[mask]
    so = np.std(o)
    ss = np.std(s)
    mo = np.mean(o)
    ms = np.mean(s)
    if so <= 0 or ss <= 0 or mo == 0:
        return np.nan
    r = np.corrcoef(o, s)[0, 1]
    alpha = ss / so
    beta = ms / mo
    return 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


def calc_r2(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 2:
        return np.nan
    o = obs[mask]
    s = sim[mask]
    if np.std(o) <= 0 or np.std(s) <= 0:
        return np.nan
    r = np.corrcoef(o, s)[0, 1]
    return r * r


def calc_corr(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 2:
        return np.nan
    o = obs[mask]
    s = sim[mask]
    if np.std(o) <= 0 or np.std(s) <= 0:
        return np.nan
    return np.corrcoef(o, s)[0, 1]


def calc_rmse(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 1:
        return np.nan
    return np.sqrt(np.mean((obs[mask] - sim[mask]) ** 2))


def calc_mae(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 1:
        return np.nan
    return np.mean(np.abs(obs[mask] - sim[mask]))


def calc_pbias(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 1:
        return np.nan
    den = np.sum(obs[mask])
    if abs(den) <= 1e-8:
        return np.nan
    return 100.0 * np.sum(sim[mask] - obs[mask]) / den


def calc_log_nse(obs, sim, eps=1e-3):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 2:
        return np.nan
    o = np.log(np.clip(obs[mask], eps, None))
    s = np.log(np.clip(sim[mask], eps, None))
    den = np.sum((o - np.mean(o)) ** 2)
    if den <= 0:
        return np.nan
    return 1.0 - np.sum((o - s) ** 2) / den


def lowflow_nse(obs, sim, q=0.3):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 2:
        return np.nan
    thr = np.quantile(obs[mask], q)
    sub = mask & (obs <= thr)
    if sub.sum() < 2:
        return np.nan
    return calc_nse(obs[sub], sim[sub])


def highflow_nse(obs, sim, q=0.7):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 2:
        return np.nan
    thr = np.quantile(obs[mask], q)
    sub = mask & (obs >= thr)
    if sub.sum() < 2:
        return np.nan
    return calc_nse(obs[sub], sim[sub])


def calc_fhv(obs, sim, q=0.999):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 5:
        return np.nan
    thr = np.quantile(obs[mask], q)
    idx = mask & (obs >= thr)
    if idx.sum() < 1:
        return np.nan
    den = np.sum(obs[idx])
    if abs(den) <= 1e-8:
        return np.nan
    return 100.0 * (np.sum(sim[idx]) - np.sum(obs[idx])) / den


def calc_flv(obs, sim, q=0.3, eps=1e-6):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 5:
        return np.nan
    thr = np.quantile(obs[mask], q)
    o = np.sort(np.clip(obs[mask & (obs <= thr)], eps, None))
    s = np.sort(np.clip(sim[mask & (obs <= thr)], eps, None))
    if len(o) < 2 or len(s) < 2:
        return np.nan
    n = min(len(o), len(s))
    o = o[:n]
    s = s[:n]
    o_min = np.min(o)
    s_min = np.min(s)
    den = np.sum(np.log(o) - np.log(o_min))
    if abs(den) <= 1e-8:
        return np.nan
    num = np.sum(np.log(s) - np.log(s_min)) - np.sum(np.log(o) - np.log(o_min))
    return -100.0 * num / den


def calc_fms(obs, sim, ql=0.2, qu=0.8, eps=1e-6):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 5:
        return np.nan
    o_l = max(np.quantile(obs[mask], ql), eps)
    o_u = max(np.quantile(obs[mask], qu), eps)
    s_l = max(np.quantile(sim[mask], ql), eps)
    s_u = max(np.quantile(sim[mask], qu), eps)
    den = np.log(o_l) - np.log(o_u)
    if abs(den) <= 1e-8:
        return np.nan
    num = (np.log(s_l) - np.log(s_u)) - (np.log(o_l) - np.log(o_u))
    return 100.0 * num / den


def calc_fdc_error(obs, sim, nq=25):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 5:
        return np.nan
    qs = np.linspace(0.02, 0.98, nq)
    oq = np.quantile(obs[mask], qs)
    sq = np.quantile(sim[mask], qs)
    scale = np.maximum(np.abs(oq), 1e-6)
    return np.sqrt(np.mean(((sq - oq) / scale) ** 2))


def safe_ratio(num, den):
    if not np.isfinite(den) or abs(den) <= 1e-8:
        return np.nan
    return num / den


def lh_baseflow(q, alpha=0.925, passes=3):
    q = np.asarray(q, dtype=np.float64)
    q = np.clip(q, 0.0, None)
    bf = q.copy()
    for _ in range(passes):
        f = np.zeros_like(bf)
        for t in range(1, len(bf)):
            f[t] = alpha * f[t - 1] + (1 + alpha) / 2.0 * (bf[t] - bf[t - 1])
            f[t] = min(max(f[t], 0.0), bf[t])
        bf = np.clip(bf - f, 0.0, q)
    return bf.astype(np.float32)


def best_lag_nse(obs, sim, max_lag=10):
    best_nse = np.nan
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            o = obs[-lag:]
            s = sim[:len(sim) + lag]
        elif lag > 0:
            o = obs[:-lag]
            s = sim[lag:]
        else:
            o = obs
            s = sim
        score = calc_nse(o, s)
        if np.isnan(best_nse) or (np.isfinite(score) and score > best_nse):
            best_nse = score
            best_lag = lag
    return best_nse, best_lag


def linear_corrected(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 2:
        return np.full_like(sim, np.nan), np.nan, np.nan
    x = sim[mask]
    y = obs[mask]
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_var = np.sum((x - x_mean) ** 2)
    if x_var <= 1e-12:
        a = 0.0
        b = y_mean
    else:
        a = np.sum((x - x_mean) * (y - y_mean)) / x_var
        b = y_mean - a * x_mean
    return a * sim + b, a, b


def annual_max_series(values, dates):
    years = pd.to_datetime(dates).year
    out = []
    for yr in np.unique(years):
        idx = np.where(years == yr)[0]
        if idx.size == 0:
            continue
        sub = values[idx]
        if np.all(~np.isfinite(sub)):
            continue
        local = np.nanargmax(sub)
        out.append(idx[local])
    return out


def peak_timing_metrics(obs, sim, dates, window=7):
    obs_idx = annual_max_series(obs, dates)
    if not obs_idx:
        return np.nan, np.nan, np.nan, np.nan
    peak_lags = []
    peak_abs_lags = []
    peak_bias = []
    hits = 0
    for oi in obs_idx:
        lo = max(0, oi - window)
        hi = min(len(sim), oi + window + 1)
        if hi - lo < 1:
            continue
        local = np.nanargmax(sim[lo:hi]) + lo
        lag = local - oi
        peak_lags.append(lag)
        peak_abs_lags.append(abs(lag))
        if np.isfinite(obs[oi]) and abs(obs[oi]) > 1e-8:
            peak_bias.append(100.0 * (sim[local] - obs[oi]) / obs[oi])
        if abs(lag) <= window:
            hits += 1
    if not peak_lags:
        return np.nan, np.nan, np.nan, np.nan
    return (
        float(np.median(peak_lags)),
        float(np.median(peak_abs_lags)),
        float(np.median(peak_bias)) if peak_bias else np.nan,
        hits / len(obs_idx),
    )


def top_flow_recall(obs, sim, q):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 5:
        return np.nan
    thr_obs = np.quantile(obs[mask], q)
    thr_sim = np.quantile(sim[mask], q)
    obs_evt = mask & (obs >= thr_obs)
    sim_evt = mask & (sim >= thr_sim)
    den = np.sum(obs_evt)
    if den == 0:
        return np.nan
    return np.sum(obs_evt & sim_evt) / den


def classify_failure_tags(row):
    tags = []
    if row["linear_corrected_nse"] - row["raw_nse"] > 0.15:
        tags.append("bias_or_amplitude")
    if row["best_lag_nse"] - row["raw_nse"] > 0.10:
        tags.append("timing_or_routing")
    if row["linear_corrected_nse"] < 0.6:
        tags.append("correlation_structure_failure")
    if (row["sse_frac_very_low"] + row["sse_frac_low"]) > 0.40:
        tags.append("low_flow_failure")
    if (row["sse_frac_high"] + row["sse_frac_extreme"]) > 0.40:
        tags.append("high_flow_failure")
    if (row["logNSE"] < 0) or (np.isfinite(row["FLV"]) and abs(row["FLV"]) > 20):
        tags.append("low_flow_recession")
    if row["false_wet_rate"] > 0.05:
        tags.append("false_flow_in_dry_period")
    if row["missed_flow_rate"] > 0.05:
        tags.append("over_suppressed_low_flow")
    if row["median_peak_abs_lag_days"] > 3:
        tags.append("peak_timing")
    if np.isfinite(row["FHV"]) and np.isfinite(row["peak_bias_percent"]) and abs(row["FHV"]) > 20:
        if row["peak_bias_percent"] < 0:
            tags.append("underpredicted_peaks")
        else:
            tags.append("overpredicted_peaks")
    return ";".join(sorted(set(tags)))


def classify_primary(row):
    if row["snow_flag"]:
        return "snow_dominated"
    if row["arid_flag"]:
        return "arid"
    if row["semi_arid_flag"]:
        return "semi_arid"
    if row["groundwater_flag"]:
        return "groundwater_dominated"
    if row["flashy_flag"]:
        return "flashy"
    return "humid"


def plot_simple_scatter(df, x, y, out_file, xlabel=None, ylabel=None, title=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df[x], df[y], s=18, alpha=0.7, edgecolors="none")
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.set_title(title or f"{y} vs {x}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_box(df, x, y, out_file, title):
    groups = []
    labels = []
    for label, sub in df.groupby(x):
        vals = sub[y].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        labels.append(label)
        groups.append(vals)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(groups, labels=labels, showfliers=False)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def main():
    project_root = Path(__file__).resolve().parent
    outputs_root = project_root / "outputs"
    model_root = outputs_root / "rnnStreamflow" / "CAMELSMODELSIX" / "DynamicSimHydModelSix" / "AllBasins" / "daymet" / "111111"
    eval_dir = model_root / "Train19801001_19951001Test19951001_20101001_ModelSix_Ep30Resume"
    analysis_dir = model_root / "analysis_ep30"
    out_dir = outputs_root / "Diagnosis_ModelSix_Ep30"
    plots_dir = out_dir / "plots"
    worst_dir = plots_dir / "worst20"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    worst_dir.mkdir(parents=True, exist_ok=True)

    eva = np.load(eval_dir / "Eva30.npy", allow_pickle=True).item()
    pred = np.load(eval_dir / "pred30.npy")[:, :, 0]
    obs = np.load(eval_dir / "obs.npy")[:, :, 0]
    diag = np.load(analysis_dir / "model_six_diagnostics_ep30.npz")
    per_basin = pd.read_csv(analysis_dir / "per_basin_model_six_metrics.csv")

    basin_ids = diag["basin_ids"].astype(int)
    dates = pd.date_range("1995-10-01", "2010-10-01", inclusive="left")
    if len(dates) != obs.shape[1]:
        dates = pd.date_range("1995-10-01", periods=obs.shape[1], freq="D")

    df = per_basin.copy()
    df = df.sort_values("gage_id").reset_index(drop=True)
    order = pd.Index(df["gage_id"].astype(int))
    pos = pd.Index(basin_ids).get_indexer(order)

    obs = obs[pos]
    pred = pred[pos]
    rainfall = diag["rainfall"][pos]
    snowfall = diag["snowfall"][pos]
    actual_et = diag["actual_ET"][pos]
    baseflow = diag["baseflow"][pos]
    groundwater_loss = diag["groundwater_loss"][pos]

    df["NSE"] = eva.get("NSE", np.array([calc_nse(obs[i], pred[i]) for i in range(obs.shape[0])], dtype=float))[pos]
    df["KGE"] = eva.get("KGE", np.array([calc_kge(obs[i], pred[i]) for i in range(obs.shape[0])], dtype=float))[pos]
    df["R2"] = eva.get("R2", np.array([calc_r2(obs[i], pred[i]) for i in range(obs.shape[0])], dtype=float))[pos]
    df["COR"] = eva.get("Corr", np.array([calc_corr(obs[i], pred[i]) for i in range(obs.shape[0])], dtype=float))[pos]
    df["RMSE"] = eva.get("RMSE", np.array([calc_rmse(obs[i], pred[i]) for i in range(obs.shape[0])], dtype=float))[pos]
    df["MAE"] = np.array([calc_mae(obs[i], pred[i]) for i in range(obs.shape[0])], dtype=float)
    df["PBias"] = eva.get("PBias", np.array([calc_pbias(obs[i], pred[i]) for i in range(obs.shape[0])], dtype=float))[pos]
    df["abs_PBias"] = np.abs(df["PBias"])
    df["FHV"] = eva.get("FHV", np.array([calc_fhv(obs[i], pred[i]) for i in range(obs.shape[0])], dtype=float))[pos]
    df["FLV"] = eva.get("FLV", np.array([calc_flv(obs[i], pred[i]) for i in range(obs.shape[0])], dtype=float))[pos]
    df["FMS"] = np.array([calc_fms(obs[i], pred[i]) for i in range(obs.shape[0])], dtype=float)
    df["logNSE"] = np.array([calc_log_nse(obs[i], pred[i]) for i in range(obs.shape[0])], dtype=float)
    df["low_flow_NSE"] = np.array([lowflow_nse(obs[i], pred[i]) for i in range(obs.shape[0])], dtype=float)
    df["high_flow_NSE"] = np.array([highflow_nse(obs[i], pred[i]) for i in range(obs.shape[0])], dtype=float)
    df["FDC_error"] = np.array([calc_fdc_error(obs[i], pred[i]) for i in range(obs.shape[0])], dtype=float)

    p_total = rainfall + snowfall
    df["Q/P"] = [safe_ratio(np.nansum(pred[i]), np.nansum(p_total[i])) for i in range(obs.shape[0])]
    df["ET/P"] = [safe_ratio(np.nansum(actual_et[i]), np.nansum(p_total[i])) for i in range(obs.shape[0])]

    obs_bfi = np.array([safe_ratio(np.nansum(lh_baseflow(obs[i])), np.nansum(obs[i])) for i in range(obs.shape[0])], dtype=float)
    pred_bfi = np.array([safe_ratio(np.nansum(baseflow[i]), np.nansum(pred[i])) for i in range(obs.shape[0])], dtype=float)
    df["BFI_obs_diag"] = obs_bfi
    df["BFI_pred_diag"] = pred_bfi
    df["BFI_error"] = np.abs(pred_bfi - obs_bfi)

    # Ladder test and regime decomposition
    raw_nse = []
    bias_nse = []
    var_nse = []
    linear_nse = []
    best_lag_nse_vals = []
    best_lag_days = []
    sse_very_low = []
    sse_low = []
    sse_mid = []
    sse_high = []
    sse_extreme = []
    zero_flow_acc = []
    false_wet_rate = []
    missed_flow_rate = []
    low_flow_bias = []
    recession_slope_error = []
    peak_lag = []
    peak_abs_lag = []
    peak_bias = []
    annual_max_recall = []
    top1_recall = []
    top5_recall = []

    for i in range(obs.shape[0]):
        o = obs[i]
        s = pred[i]
        raw = calc_nse(o, s)
        raw_nse.append(raw)

        pred_bc = s + (np.nanmean(o) - np.nanmean(s))
        bias_nse.append(calc_nse(o, pred_bc))

        s_std = np.nanstd(s)
        o_std = np.nanstd(o)
        if np.isfinite(s_std) and s_std > 1e-8 and np.isfinite(o_std):
            pred_vc = np.nanmean(s) + (o_std / s_std) * (s - np.nanmean(s))
        else:
            pred_vc = np.full_like(s, np.nan)
        var_nse.append(calc_nse(o, pred_vc))

        pred_lc, _, _ = linear_corrected(o, s)
        linear_nse.append(calc_nse(o, pred_lc))

        lag_nse, lag_day = best_lag_nse(o, s)
        best_lag_nse_vals.append(lag_nse)
        best_lag_days.append(lag_day)

        mask = np.isfinite(o) & np.isfinite(s)
        if mask.sum() >= 5:
            q20, q50, q80, q95 = np.quantile(o[mask], [0.2, 0.5, 0.8, 0.95])
            err = (o - s) ** 2
            total = np.nansum(err[mask])
            if total <= 0:
                total = np.nan
            bands = [
                mask & (o <= q20),
                mask & (o > q20) & (o <= q50),
                mask & (o > q50) & (o <= q80),
                mask & (o > q80) & (o <= q95),
                mask & (o > q95),
            ]
            vals = [safe_ratio(np.nansum(err[b]), total) for b in bands]
        else:
            vals = [np.nan] * 5
        sse_very_low.append(vals[0])
        sse_low.append(vals[1])
        sse_mid.append(vals[2])
        sse_high.append(vals[3])
        sse_extreme.append(vals[4])

        pos = o[np.isfinite(o) & (o > 0)]
        eps = max(1e-3, 0.01 * np.nanmean(pos)) if pos.size else 1e-3
        dry_obs = o <= eps
        dry_pred = s <= eps
        valid = np.isfinite(o) & np.isfinite(s)
        if valid.sum() > 0:
            zero_flow_acc.append(np.mean((dry_obs == dry_pred)[valid]))
            false_wet_rate.append(np.mean(((o <= eps) & (s > eps))[valid]))
            missed_flow_rate.append(np.mean(((o > eps) & (s <= eps))[valid]))
        else:
            zero_flow_acc.append(np.nan)
            false_wet_rate.append(np.nan)
            missed_flow_rate.append(np.nan)

        low_mask = valid & (o <= np.nanquantile(o[valid], 0.3))
        if low_mask.sum() > 0:
            low_flow_bias.append(np.nanmean(s[low_mask] - o[low_mask]))
        else:
            low_flow_bias.append(np.nan)

        precip = p_total[i]
        rec_mask = (
            np.isfinite(o[1:]) &
            np.isfinite(o[:-1]) &
            np.isfinite(s[1:]) &
            np.isfinite(s[:-1]) &
            (precip[1:] <= eps) &
            (o[1:] < o[:-1])
        )
        if rec_mask.sum() > 0:
            obs_rec = np.median(np.log(o[1:][rec_mask] + eps) - np.log(o[:-1][rec_mask] + eps))
            sim_rec = np.median(np.log(s[1:][rec_mask] + eps) - np.log(s[:-1][rec_mask] + eps))
            recession_slope_error.append(sim_rec - obs_rec)
        else:
            recession_slope_error.append(np.nan)

        med_lag, med_abs_lag, pk_bias, annual_rec = peak_timing_metrics(o, s, dates)
        peak_lag.append(med_lag)
        peak_abs_lag.append(med_abs_lag)
        peak_bias.append(pk_bias)
        annual_max_recall.append(annual_rec)
        top1_recall.append(top_flow_recall(o, s, 0.99))
        top5_recall.append(top_flow_recall(o, s, 0.95))

    df["raw_nse"] = raw_nse
    df["bias_corrected_nse"] = bias_nse
    df["variance_corrected_nse"] = var_nse
    df["linear_corrected_nse"] = linear_nse
    df["best_lag_nse"] = best_lag_nse_vals
    df["best_lag_days"] = best_lag_days
    df["sse_frac_very_low"] = sse_very_low
    df["sse_frac_low"] = sse_low
    df["sse_frac_mid"] = sse_mid
    df["sse_frac_high"] = sse_high
    df["sse_frac_extreme"] = sse_extreme
    df["zero_flow_accuracy"] = zero_flow_acc
    df["false_wet_rate"] = false_wet_rate
    df["missed_flow_rate"] = missed_flow_rate
    df["low_flow_bias"] = low_flow_bias
    df["recession_slope_error"] = recession_slope_error
    df["median_peak_lag_days"] = peak_lag
    df["median_peak_abs_lag_days"] = peak_abs_lag
    df["peak_bias_percent"] = peak_bias
    df["annual_max_recall"] = annual_max_recall
    df["top_1_percent_recall"] = top1_recall
    df["top_5_percent_recall"] = top5_recall

    # Attribute loading and classification
    attr_candidates = [
        'p_mean', 'pet_mean', 'frac_snow', 'aridity', 'elev_mean', 'slope_mean',
        'area_gages2', 'frac_forest'
    ]
    attr_path = project_root / "Camels"
    from hydroDL.data import camels
    camels.initcamels(str(attr_path))
    attr_df = camels.DataframeCamels(tRange=[19801001, 19951001], subset=df["gage_id"].astype(int).tolist(), forType="daymet")
    attrs = attr_df.getDataConst(varLst=attr_candidates, doNorm=False, rmNan=False)
    attrs_df = pd.DataFrame(attrs, columns=attr_candidates)
    attrs_df["gage_id"] = df["gage_id"].astype(int).values
    df = df.merge(attrs_df, on="gage_id", how="left")
    df["snow_fraction"] = df.get("frac_snow")
    df["aridity_index"] = df.get("aridity")
    df["forest_cover"] = df.get("frac_forest")
    df["soil_depth"] = np.nan
    df["elevation"] = df.get("elev_mean")
    df["snow_flag"] = (df["snow_fraction"] >= 0.25).fillna(False)
    df["arid_flag"] = (df["aridity_index"] >= 2.0).fillna(False)
    df["semi_arid_flag"] = ((df["aridity_index"] >= 1.0) & (df["aridity_index"] < 2.0)).fillna(False)
    df["humid_flag"] = (df["aridity_index"] < 1.0).fillna(False)
    slope_thr = np.nanmedian(df["slope_mean"]) if "slope_mean" in df else np.nan
    flow_cv = np.array([np.nanstd(obs[i]) / max(np.nanmean(obs[i]), 1e-6) for i in range(obs.shape[0])], dtype=float)
    df["flashy_flag"] = ((flow_cv > np.nanmedian(flow_cv)) | (df.get("slope_mean", 0) > slope_thr)).fillna(False)
    df["groundwater_flag"] = (df["BFI_obs_diag"] >= 0.6).fillna(False)
    df["basin_class_primary"] = df.apply(classify_primary, axis=1)

    poor = df[df["NSE"] < 0.5].copy()
    poor["failure_type"] = poor.apply(classify_failure_tags, axis=1)
    df["failure_type"] = ""
    df.loc[poor.index, "failure_type"] = poor["failure_type"]

    # Save tables
    df.to_csv(out_dir / "diagnosis_all_basins.csv", index=False)
    poor.to_csv(out_dir / "diagnosis_poor_basins.csv", index=False)

    # Group summaries
    summary_cols = [
        "NSE", "KGE", "logNSE", "low_flow_NSE", "high_flow_NSE", "FDC_error",
        "BFI_error", "PBias", "FLV", "FHV", "best_lag_days", "best_lag_nse",
        "linear_corrected_nse"
    ]
    group_summary = poor.groupby("basin_class_primary")[summary_cols].median().reset_index()
    group_summary.to_csv(out_dir / "diagnosis_group_summary.csv", index=False)

    failure_rows = []
    for tag in sorted({t for s in poor["failure_type"] for t in s.split(";") if t}):
        sub = poor[poor["failure_type"].str.contains(tag, regex=False)]
        if len(sub) == 0:
            continue
        row = {"failure_type": tag, "count": len(sub)}
        for col in summary_cols:
            row[col] = sub[col].median()
        failure_rows.append(row)
    failure_df = pd.DataFrame(failure_rows).sort_values("count", ascending=False)
    failure_df.to_csv(out_dir / "diagnosis_failure_type_summary.csv", index=False)

    for flag in ["snow_flag", "arid_flag", "semi_arid_flag", "humid_flag"]:
        flag_df = poor.groupby(flag)[summary_cols].median().reset_index()
        flag_df.to_csv(out_dir / f"diagnosis_{flag}_summary.csv", index=False)

    # Plots
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["NSE"], bins=40, color="lightgray", edgecolor="black")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1.5)
    ax.set_title("NSE histogram with poor-basin threshold")
    ax.set_xlabel("NSE")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(plots_dir / "nse_histogram_poor_highlight.png", dpi=180)
    plt.close(fig)

    plot_simple_scatter(poor, "raw_nse", "linear_corrected_nse", plots_dir / "raw_nse_vs_linear_corrected_nse.png")
    plot_simple_scatter(poor, "raw_nse", "best_lag_nse", plots_dir / "raw_nse_vs_best_lag_nse.png")
    plot_simple_scatter(df, "aridity_index", "NSE", plots_dir / "nse_vs_aridity.png")
    plot_simple_scatter(df, "snow_fraction", "NSE", plots_dir / "nse_vs_snow_fraction.png")
    plot_box(df, "basin_class_primary", "NSE", plots_dir / "nse_by_basin_class.png", "NSE by basin class")
    plot_box(df, "basin_class_primary", "low_flow_NSE", plots_dir / "lowflow_nse_by_basin_class.png", "Low-flow NSE by basin class")
    plot_box(df, "basin_class_primary", "high_flow_NSE", plots_dir / "highflow_nse_by_basin_class.png", "High-flow NSE by basin class")

    failure_counts = failure_df[["failure_type", "count"]].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(failure_counts["failure_type"], failure_counts["count"], color="steelblue")
    ax.set_title("Failure type counts for poor basins")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(plots_dir / "failure_type_counts.png", dpi=180)
    plt.close(fig)

    worst20 = poor.nsmallest(20, "NSE")
    for _, row in worst20.iterrows():
        i = order.get_loc(int(row["gage_id"]))
        o = obs[i]
        s = pred[i]
        gid = int(row["gage_id"])
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].plot(dates, o, label="obs", linewidth=1)
        axes[0].plot(dates, s, label="pred", linewidth=1)
        axes[0].set_title(f"{gid} hydrograph")
        axes[0].legend()
        oq = np.sort(o[np.isfinite(o)])[::-1]
        sq = np.sort(s[np.isfinite(s)])[::-1]
        n = min(len(oq), len(sq))
        p = np.linspace(0, 100, n)
        axes[1].plot(p, oq[:n], label="obs")
        axes[1].plot(p, sq[:n], label="pred")
        axes[1].set_title("Flow duration curve")
        axes[1].legend()
        mask = np.isfinite(o) & np.isfinite(s)
        if mask.sum() > 5:
            ranks = pd.Series(o[mask]).rank(pct=True).to_numpy() * 100
            axes[2].scatter(ranks, s[mask] - o[mask], s=8, alpha=0.6)
        axes[2].axhline(0, color="k", linewidth=1)
        axes[2].set_title("Residual vs flow percentile")
        fig.tight_layout()
        fig.savefig(worst_dir / f"{gid}.png", dpi=180)
        plt.close(fig)

    # Basin lists
    def write_ids(path, frame):
        with open(path, "w") as fp:
            for gid in frame["gage_id"].astype(int).tolist():
                fp.write(f"{gid}\n")

    write_ids(out_dir / "poor_basins_all.txt", poor)
    write_ids(out_dir / "poor_lowflow_basins.txt", poor[poor["failure_type"].str.contains("low_flow_failure|low_flow_recession", regex=True)])
    write_ids(out_dir / "poor_highflow_basins.txt", poor[poor["failure_type"].str.contains("high_flow_failure|underpredicted_peaks|overpredicted_peaks", regex=True)])
    write_ids(out_dir / "poor_snow_basins.txt", poor[poor["snow_flag"]])
    write_ids(out_dir / "poor_arid_basins.txt", poor[poor["arid_flag"] | poor["semi_arid_flag"]])
    write_ids(out_dir / "poor_timing_basins.txt", poor[(poor["best_lag_nse"] - poor["raw_nse"]) > 0.10])
    write_ids(out_dir / "poor_bias_basins.txt", poor[(poor["linear_corrected_nse"] - poor["raw_nse"]) > 0.15])
    write_ids(out_dir / "poor_structure_basins.txt", poor[poor["linear_corrected_nse"] < 0.6])

    # Final text report
    counts = {
        "poor_basins": len(poor),
        "low_flow_failure": int(poor["failure_type"].str.contains("low_flow_failure|low_flow_recession", regex=True).sum()),
        "high_flow_failure": int(poor["failure_type"].str.contains("high_flow_failure|underpredicted_peaks|overpredicted_peaks", regex=True).sum()),
        "snow_failure": int((poor["snow_flag"]).sum()),
        "arid_semi_arid_failure": int((poor["arid_flag"] | poor["semi_arid_flag"]).sum()),
        "timing_failure": int((poor["best_lag_nse"] - poor["raw_nse"] > 0.10).sum()),
        "bias_amplitude_failure": int((poor["linear_corrected_nse"] - poor["raw_nse"] > 0.15).sum()),
        "structure_failure": int((poor["linear_corrected_nse"] < 0.6).sum()),
    }
    recommendations = []
    if counts["low_flow_failure"] >= max(counts["high_flow_failure"], counts["snow_failure"], counts["timing_failure"]):
        recommendations.append("recommend ModelLowNSE with stronger slow groundwater/recession memory.")
    if counts["high_flow_failure"] > 0:
        recommendations.append("recommend ModelLowNSE with threshold excess runoff.")
    if counts["snow_failure"] > 0:
        recommendations.append("recommend ModelLowNSE-Snow with improved melt timing and snow partition.")
    if counts["arid_semi_arid_failure"] > 0:
        recommendations.append("recommend ModelLowNSE-Dry with conservative channel loss and zero-flow handling.")
    if counts["timing_failure"] > 0:
        recommendations.append("recommend component routing and dynamic routing scale.")
    if counts["structure_failure"] > 0:
        recommendations.append("recommend adding missing process states rather than tuning parameters.")

    with open(out_dir / "diagnosis_report.txt", "w") as fp:
        for k, v in counts.items():
            fp.write(f"{k}: {v}\n")
        fp.write("\nRecommendations:\n")
        for rec in recommendations:
            fp.write(f"- {rec}\n")

    print(f"Number of poor basins: {counts['poor_basins']}")
    print(f"Number of low-flow failure basins: {counts['low_flow_failure']}")
    print(f"Number of high-flow failure basins: {counts['high_flow_failure']}")
    print(f"Number of snow failure basins: {counts['snow_failure']}")
    print(f"Number of arid/semi-arid failure basins: {counts['arid_semi_arid_failure']}")
    print(f"Number of timing failure basins: {counts['timing_failure']}")
    print(f"Number of bias/amplitude failure basins: {counts['bias_amplitude_failure']}")
    print(f"Number of structure failure basins: {counts['structure_failure']}")
    for rec in recommendations:
        print(rec)


if __name__ == "__main__":
    import sys
    sys.path.append(str((Path(__file__).resolve().parent / "code" / "dPLHBVrelease" / "hydroDL-dev")))
    main()
