#!/usr/bin/env python3
from pathlib import Path
import sys
import math
import json
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/mircore/Desktop/diff_hydro')
CODE_ROOT = ROOT / 'code' / 'dPLHBVrelease' / 'hydroDL-dev'
sys.path.append(str(CODE_ROOT))
from hydroDL.data import camels
from hydroDL.master import loadModel
from hydroDL import utils

GPU_ID = int(os.environ.get('SNOWSIMHYDMC_HETER_REPORT_GPU_ID', '0'))
MC_EPOCH = int(os.environ.get('SNOWSIMHYDMC_HETER_REPORT_EPOCH', '20'))
MC_RESULT_SUFFIX = os.environ.get('SNOWSIMHYDMC_HETER_REPORT_RESULT_SUFFIX', '_SnowSIMHYDMC_HeterAll671_BS32_HS64_MaxIter100_Ep20')
OUT_NAME = os.environ.get('SNOWSIMHYDMC_HETER_REPORT_OUTDIR', 'report_snowsimhydmc_heter_vs_hbv_detailed')
OUT = ROOT / 'outputs' / OUT_NAME
OUT.mkdir(parents=True, exist_ok=True)
PARAM_MAP_DIR = OUT / 'parameter_maps'
PARAM_MAP_DIR.mkdir(parents=True, exist_ok=True)
PARAM_MAP_WAVG_DIR = PARAM_MAP_DIR / 'weighted_average'
PARAM_MAP_WAVG_DIR.mkdir(parents=True, exist_ok=True)
PARAM_MAP_MISC_DIR = PARAM_MAP_DIR / 'routing_weights'
PARAM_MAP_MISC_DIR.mkdir(parents=True, exist_ok=True)

HBV_DIR = ROOT / 'outputs' / 'rnnStreamflow' / 'CAMELSDemo' / 'dPLHBV' / 'ALL' / 'Testforc' / 'daymet' / 'BuffOpt0' / 'RMSE_para0.25' / '111111' / 'Train19801001_19951001Test19951001_20101001Buff5478Nmul16_HBVAll671_BS32_HS64_MaxIter100'
MC_DIR = ROOT / 'outputs' / 'rnnStreamflow' / 'CAMELSSNOWSIMHYDMC_HETER' / 'dPLSnowSIMHYDMC_Heter' / 'AllBasins' / 'daymet' / '111111' / f'Train19801001_19951001Test19951001_20101001{MC_RESULT_SUFFIX}'
MC_RUN_DIR = ROOT / 'outputs' / 'rnnStreamflow' / 'CAMELSSNOWSIMHYDMC_HETER' / 'dPLSnowSIMHYDMC_Heter' / 'AllBasins' / 'daymet' / '111111' / 'T_19801001_19951001_BS_32_HS_64_RHO_365_Buff_365_Mul_4_Route_1_CmpW_1_LGDyn_1_All671_BS32_HS64_MaxIter100'
MC_RUN = MC_RUN_DIR / 'run.csv'
PARAM_NAMES = ['INSC', 'COEF', 'SQ', 'SMSC', 'SUB', 'CRAK', 'K', 'LG', 'TT', 'CFMAX', 'CFR', 'CWH']
ATTR_VARS = [
    'p_mean', 'pet_mean', 'p_seasonality', 'frac_snow', 'aridity', 'high_prec_freq', 'high_prec_dur',
    'low_prec_freq', 'low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
    'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50',
    'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
    'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'geol_1st_class', 'glim_1st_class_frac',
    'geol_2nd_class', 'glim_2nd_class_frac', 'carbonate_rocks_frac', 'geol_porostiy', 'geol_permeability'
]

if torch.cuda.is_available():
    torch.cuda.set_device(GPU_ID)


def safe_stats(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    return obs[mask], sim[mask]


def pearson_r(obs, sim):
    if len(obs) < 2:
        return np.nan
    so = np.std(obs)
    ss = np.std(sim)
    if so == 0 or ss == 0:
        return np.nan
    return float(np.corrcoef(obs, sim)[0, 1])


def calc_metrics_per_basin(pred, obs, dates):
    n = pred.shape[0]
    out = {
        'NSE': np.full(n, np.nan),
        'KGE': np.full(n, np.nan),
        'alpha_nse': np.full(n, np.nan),
        'beta_nse': np.full(n, np.nan),
        'COR': np.full(n, np.nan),
        'R2corr': np.full(n, np.nan),
        'RMSE': np.full(n, np.nan),
        'MAE': np.full(n, np.nan),
        'PBias': np.full(n, np.nan),
        'FHV': np.full(n, np.nan),
        'FLV': np.full(n, np.nan),
        'FMS': np.full(n, np.nan),
        'PT': np.full(n, np.nan),
        'precision_2yr': np.full(n, np.nan),
        'recall_2yr': np.full(n, np.nan),
        'f1_2yr': np.full(n, np.nan),
        'precision_5yr': np.full(n, np.nan),
        'recall_5yr': np.full(n, np.nan),
        'f1_5yr': np.full(n, np.nan),
    }

    wy = dates.year + (dates.month >= 10).astype(int)
    years = np.unique(wy)

    for i in range(n):
        o, s = safe_stats(obs[i], pred[i])
        if len(o) < 2:
            continue
        mean_o = np.mean(o)
        std_o = np.std(o)
        mean_s = np.mean(s)
        std_s = np.std(s)
        ss_res = np.sum((o - s) ** 2)
        ss_tot = np.sum((o - mean_o) ** 2)
        r = pearson_r(o, s)

        out['NSE'][i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        out['COR'][i] = r
        out['R2corr'][i] = r ** 2 if np.isfinite(r) else np.nan
        out['RMSE'][i] = math.sqrt(np.mean((o - s) ** 2))
        out['MAE'][i] = np.mean(np.abs(o - s))
        out['PBias'][i] = 100.0 * np.sum(s - o) / np.sum(o) if np.sum(o) != 0 else np.nan

        alpha = std_s / std_o if std_o > 0 else np.nan
        beta_ratio = mean_s / mean_o if mean_o != 0 else np.nan
        beta_nse = (mean_s - mean_o) / std_o if std_o > 0 else np.nan
        out['alpha_nse'][i] = alpha
        out['beta_nse'][i] = beta_nse
        if np.isfinite(r) and np.isfinite(alpha) and np.isfinite(beta_ratio):
            out['KGE'][i] = 1.0 - math.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta_ratio - 1.0) ** 2)

        q999 = np.quantile(o, 0.999)
        high_mask = o >= q999
        if np.any(high_mask) and np.sum(o[high_mask]) != 0:
            out['FHV'][i] = 100.0 * (np.sum(s[high_mask]) - np.sum(o[high_mask])) / np.sum(o[high_mask])

        q30 = np.quantile(o, 0.30)
        low_mask = o <= q30
        if np.any(low_mask):
            lo_o = np.clip(np.sort(o[low_mask]), 1e-6, None)
            lo_s = np.clip(np.sort(s[low_mask]), 1e-6, None)
            num = np.sum(np.log(lo_s) - np.log(lo_s.min())) - np.sum(np.log(lo_o) - np.log(lo_o.min()))
            den = np.sum(np.log(lo_o) - np.log(lo_o.min()))
            out['FLV'][i] = -100.0 * num / den if den != 0 else np.nan

        q20o = max(np.quantile(o, 0.20), 1e-6)
        q80o = max(np.quantile(o, 0.80), 1e-6)
        q20s = max(np.quantile(s, 0.20), 1e-6)
        q80s = max(np.quantile(s, 0.80), 1e-6)
        den = np.log(q20o) - np.log(q80o)
        if den != 0:
            out['FMS'][i] = 100.0 * (((np.log(q20s) - np.log(q80s)) - (np.log(q20o) - np.log(q80o))) / den)

        pt_diffs = []
        for year in years:
            idx = np.where(wy == year)[0]
            oo = obs[i, idx]
            ss = pred[i, idx]
            mask = np.isfinite(oo) & np.isfinite(ss)
            if np.sum(mask) < 2:
                continue
            oo = oo[mask]
            ss = ss[mask]
            pt_diffs.append(abs(int(np.argmax(oo)) - int(np.argmax(ss))))
        if pt_diffs:
            out['PT'][i] = float(np.mean(pt_diffs))

        annual_max = []
        for year in years:
            idx = np.where(wy == year)[0]
            oo = obs[i, idx]
            if np.all(~np.isfinite(oo)):
                continue
            annual_max.append(np.nanmax(oo))
        annual_max = np.asarray(annual_max, dtype=float)
        if len(annual_max) >= 5:
            for T, prefix in [(2, '2yr'), (5, '5yr')]:
                thr = np.quantile(annual_max[np.isfinite(annual_max)], 1.0 - 1.0 / T)
                obs_evt = o >= thr
                sim_evt = s >= thr
                tp = np.sum(obs_evt & sim_evt)
                fp = np.sum(~obs_evt & sim_evt)
                fn = np.sum(obs_evt & ~sim_evt)
                prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
                rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                f1 = 2 * prec * rec / (prec + rec) if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0 else np.nan
                out[f'precision_{prefix}'][i] = prec
                out[f'recall_{prefix}'][i] = rec
                out[f'f1_{prefix}'][i] = f1

    return out


def ecdf(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = np.sort(vals)
    y = np.arange(1, len(vals) + 1) / len(vals)
    return vals, y


def metric_summary_row(name, mc_vals, hbv_vals, basis):
    mc_vals = np.asarray(mc_vals, dtype=float)
    hbv_vals = np.asarray(hbv_vals, dtype=float)
    mask = np.isfinite(mc_vals) & np.isfinite(hbv_vals)
    mc = mc_vals[mask]
    hbv = hbv_vals[mask]
    if basis == 'higher':
        better = mc > hbv
    else:
        better = mc < hbv
    return {
        'metric': name,
        'comparison_basis': basis,
        'mc_median': float(np.nanmedian(mc_vals)),
        'hbv_median': float(np.nanmedian(hbv_vals)),
        'mc_mean': float(np.nanmean(mc_vals)),
        'hbv_mean': float(np.nanmean(hbv_vals)),
        'n_valid_basins': int(mask.sum()),
        'mc_better_basins': int(np.sum(better)),
        'hbv_better_basins': int(np.sum(~better)),
        'mc_better_fraction': float(np.mean(better)) if len(better) > 0 else np.nan,
    }


def plot_binned_spatial(df, value_col, title, out_path, bin_edges, labels, colors, size_map):
    vals = df[value_col].to_numpy(dtype=float)
    bins = pd.cut(vals, bins=bin_edges, labels=labels, include_lowest=True)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.set_facecolor('#f6f3ee')
    for lab, color in zip(labels, colors):
        mask = bins.astype(str) == str(lab)
        if np.sum(mask) == 0:
            continue
        ax.scatter(df.loc[mask, 'lon'], df.loc[mask, 'lat'],
                   s=size_map[str(lab)], c=color, edgecolors='black', linewidths=0.2,
                   alpha=0.9, label=f'{lab} (n={int(np.sum(mask))})')
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(alpha=0.2)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_continuous_spatial(df, value_col, title, out_path, cmap='viridis'):
    vals = df[value_col].to_numpy(dtype=float)
    valid = np.isfinite(vals)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.set_facecolor('#f6f3ee')
    sc = ax.scatter(
        df.loc[valid, 'lon'],
        df.loc[valid, 'lat'],
        c=vals[valid],
        s=28,
        cmap=cmap,
        edgecolors='black',
        linewidths=0.15,
        alpha=0.92,
    )
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    ax.set_title(f'{title}\nmin={vmin:.4g}, max={vmax:.4g}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(alpha=0.2)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.86)
    cbar.set_label(value_col)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def preferred_cmap(name):
    if name in {'TT'}:
        return 'coolwarm'
    if name.startswith('weight_'):
        return 'plasma'
    if name.startswith('lg_dyn_std'):
        return 'magma'
    return 'viridis'


def extract_learned_parameters(meta):
    camels.initcamels(str(ROOT / 'Camels'))
    gageinfo = camels.gageDict
    gageid = gageinfo['id']
    gageidLst = gageid.tolist()
    train_ids = gageid.tolist()
    train_ind = [gageidLst.index(j) for j in train_ids]

    dfInv = camels.DataframeCamels(tRange=[19801001, 19951001], subset=train_ids, forType='daymet')
    forcInvUN = dfInv.getDataTs(varLst=['prcp', 'tmean'], doNorm=False, rmNan=False)
    attrsUN = dfInv.getDataConst(varLst=ATTR_VARS, doNorm=False, rmNan=False)

    varLstNL = ['PEVAP']
    tPETLst = utils.time.tRange2Array([19800101, 20150101])
    TinvLst = utils.time.tRange2Array([19801001, 19951001])
    PETDir = str(ROOT / 'Camels' / 'pet_harg' / 'daymet') + '/'
    ntime = len(tPETLst)
    PETfull = np.empty([len(gageid), ntime, len(varLstNL)])
    for k in range(len(gageid)):
        PETfull[k, :, :] = camels.readcsvGage(PETDir, gageid[k], varLstNL, ntime)
    _, _, ind2inv = np.intersect1d(TinvLst, tPETLst, return_indices=True)
    PETInvUN = PETfull[:, ind2inv, :][train_ind, :, :]

    series_inv = np.concatenate([forcInvUN, PETInvUN], axis=2)
    with open(MC_RUN_DIR / 'statDict.json', 'r') as fp:
        statDict = json.load(fp)
    attr_norm = camels.transNormbyDic(attrsUN, ATTR_VARS, statDict, toNorm=True)
    attr_norm[np.isnan(attr_norm)] = 0.0
    series_norm = camels.transNormbyDic(series_inv, ['prcp', 'tmean', 'pet'], statDict, toNorm=True)
    series_norm[np.isnan(series_norm)] = 0.0

    z_base = torch.from_numpy(np.swapaxes(series_norm, 1, 0)).float()
    rho = series_norm.shape[1]
    c_rep = np.repeat(attr_norm[:, None, :], rho, axis=1)
    c_tensor = torch.from_numpy(np.swapaxes(c_rep, 1, 0)).float()
    z_tensor = torch.cat((z_base, c_tensor), dim=2)

    model = loadModel(str(MC_RUN_DIR), epoch=MC_EPOCH)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda(GPU_ID)

    param_chunks = []
    dyn_chunks = []
    with torch.no_grad():
        for i0 in range(0, z_tensor.shape[1], 32):
            z_part = z_tensor[:, i0:i0 + 32, :]
            if torch.cuda.is_available():
                z_part = z_part.cuda(GPU_ID)
            static_feat = model.staticFeat(z_part[-1, :, -model.nattr:])
            static_params0 = model.staticOut(static_feat)
            param_chunks.append(static_params0.detach().cpu())
            lg_seq = torch.sigmoid(
                model.lstmdyn(z_part) +
                model.lgAttr(z_part[-1, :, -model.nattr:]).unsqueeze(0).repeat(z_part.shape[0], 1, 1)
            )
            dyn_chunks.append(lg_seq.detach().cpu())

    params0 = torch.cat(param_chunks, dim=0)
    dyn = torch.cat(dyn_chunks, dim=1).numpy()

    static0 = params0[:, :model.nstaticpm].view(params0.shape[0], model.nfea, model.nmul) + model.compStaticBias.detach().cpu()
    snowpara = torch.sigmoid(static0)
    theta = snowpara.permute(0, 2, 1).contiguous().view(params0.shape[0] * model.nmul, model.nfea)
    if torch.cuda.is_available():
        theta = theta.cuda(GPU_ID)
    denorm = model.simhyd.denorm_params(theta).detach().cpu().numpy().reshape(params0.shape[0], model.nmul, model.nfea)

    cursor = model.nstaticpm
    rout0 = params0[:, cursor:cursor + model.nroutpm]
    route_sig = torch.sigmoid(rout0).numpy()
    route_a = route_sig[:, 0] * 2.9
    route_b = route_sig[:, 1] * 6.5
    cursor += model.nroutpm

    w0 = params0[:, cursor:cursor + model.nwtspm]
    weights = torch.softmax(w0 + model.compWeightBias.detach().cpu(), dim=-1).numpy()

    dyn_mean = dyn.mean(axis=0)
    dyn_std = dyn.std(axis=0)
    weighted_avg = np.sum(denorm * weights[:, :, None], axis=1)

    param_df = meta[['gage_id', 'gage_name', 'lat', 'lon']].copy()
    summary_rows = []
    for j, pname in enumerate(PARAM_NAMES):
        col = f'{pname}_wavg'
        param_df[col] = weighted_avg[:, j]
        summary_rows.append({
            'parameter': col,
            'min': float(np.nanmin(param_df[col])),
            'max': float(np.nanmax(param_df[col])),
            'mean': float(np.nanmean(param_df[col])),
            'std': float(np.nanstd(param_df[col])),
        })
        plot_continuous_spatial(param_df, col, f'{pname} weighted-average map', PARAM_MAP_WAVG_DIR / f'{col}.png', cmap=preferred_cmap(pname))

    param_df['route_a'] = route_a
    param_df['route_b'] = route_b
    for col in ['route_a', 'route_b']:
        summary_rows.append({
            'parameter': col,
            'min': float(np.nanmin(param_df[col])),
            'max': float(np.nanmax(param_df[col])),
            'mean': float(np.nanmean(param_df[col])),
            'std': float(np.nanstd(param_df[col])),
        })
        plot_continuous_spatial(param_df, col, f'{col} map', PARAM_MAP_MISC_DIR / f'{col}.png', cmap=preferred_cmap(col))

    for c in range(weights.shape[1]):
        col = f'weight_c{c+1}'
        param_df[col] = weights[:, c]
        summary_rows.append({
            'parameter': col,
            'min': float(np.nanmin(param_df[col])),
            'max': float(np.nanmax(param_df[col])),
            'mean': float(np.nanmean(param_df[col])),
            'std': float(np.nanstd(param_df[col])),
        })
        plot_continuous_spatial(param_df, col, f'{col} map', PARAM_MAP_MISC_DIR / f'{col}.png', cmap=preferred_cmap(col))

    param_df['lg_dyn_mean_wavg'] = np.sum(dyn_mean * weights, axis=1)
    param_df['lg_dyn_std_wavg'] = np.sum(dyn_std * weights, axis=1)
    for col in ['lg_dyn_mean_wavg', 'lg_dyn_std_wavg']:
        summary_rows.append({
            'parameter': col,
            'min': float(np.nanmin(param_df[col])),
            'max': float(np.nanmax(param_df[col])),
            'mean': float(np.nanmean(param_df[col])),
            'std': float(np.nanstd(param_df[col])),
        })
        plot_continuous_spatial(param_df, col, f'{col} map', PARAM_MAP_MISC_DIR / f'{col}.png', cmap=preferred_cmap(col))

    param_df.to_csv(OUT / 'learned_parameter_values_by_basin.csv', index=False)
    pd.DataFrame(summary_rows).to_csv(OUT / 'learned_parameter_ranges.csv', index=False)


camels.initcamels(str(ROOT / 'Camels'))
g = camels.gageDict
meta = pd.DataFrame({
    'gage_id': g['id'].astype(int),
    'gage_name': g['name'],
    'huc': g['huc'].astype(int),
    'lat': g['lat'].astype(float),
    'lon': g['lon'].astype(float),
    'area_km2': g['area'].astype(float),
})

attr_vars = ['frac_snow', 'aridity', 'elev_mean', 'p_mean']
df_attr = camels.DataframeCamels(tRange=[19801001, 19951001], subset=g['id'].tolist(), forType='daymet')
attrs = df_attr.getDataConst(varLst=attr_vars, doNorm=False, rmNan=False)
for j, name in enumerate(attr_vars):
    meta[name] = attrs[:, j]

hbv_e = np.load(HBV_DIR / 'Eva10.npy', allow_pickle=True)[0]
mc_e = np.load(MC_DIR / f'Eva{MC_EPOCH}.npy', allow_pickle=True)[0]

progress_rows = [
    {'model': 'HBV', 'epoch': 10, 'median_test_nse': float(np.nanmedian(hbv_e['NSE'])), 'mean_test_nse': float(np.nanmean(hbv_e['NSE']))},
    {'model': 'Snow-SIMHYD-MC-Heter', 'epoch': MC_EPOCH, 'median_test_nse': float(np.nanmedian(mc_e['NSE'])), 'mean_test_nse': float(np.nanmean(mc_e['NSE']))},
]
progress = pd.DataFrame(progress_rows)
progress.to_csv(OUT / 'model_progress_and_baselines.csv', index=False)
if MC_RUN.exists():
    pd.read_csv(MC_RUN, header=None, names=['log']).to_csv(OUT / 'mc_train_log.csv', index=False)

hbv_pred = np.load(HBV_DIR / 'pred10.npy', allow_pickle=True)[:, :, 0]
hbv_obs = np.load(HBV_DIR / 'obs.npy', allow_pickle=True)[:, :, 0]
mc_pred = np.load(MC_DIR / f'pred{MC_EPOCH}.npy', allow_pickle=True)[:, :, 0]
mc_obs = np.load(MC_DIR / 'obs.npy', allow_pickle=True)[:, :, 0]

dates = pd.date_range('1995-10-01', periods=hbv_pred.shape[1], freq='D')

hbv_metrics = calc_metrics_per_basin(hbv_pred, hbv_obs, dates)
mc_metrics = calc_metrics_per_basin(mc_pred, mc_obs, dates)

for k in hbv_metrics:
    meta[f'hbv_{k}'] = hbv_metrics[k]
    meta[f'mc_{k}'] = mc_metrics[k]

meta['nse_diff_mc_minus_hbv'] = meta['mc_NSE'] - meta['hbv_NSE']
meta['winner_nse'] = np.where(meta['nse_diff_mc_minus_hbv'] > 0, 'Snow-SIMHYD-MC-Heter', 'HBV')
meta.to_csv(OUT / 'per_basin_metrics_snowsimhydmc_heter_vs_hbv.csv', index=False)

summary_rows = []
summary_rows.append(metric_summary_row('NSE', meta['mc_NSE'], meta['hbv_NSE'], 'higher'))
summary_rows.append(metric_summary_row('KGE', meta['mc_KGE'], meta['hbv_KGE'], 'higher'))
summary_rows.append(metric_summary_row('COR', meta['mc_COR'], meta['hbv_COR'], 'higher'))
summary_rows.append(metric_summary_row('R2corr', meta['mc_R2corr'], meta['hbv_R2corr'], 'higher'))
summary_rows.append(metric_summary_row('RMSE', meta['mc_RMSE'], meta['hbv_RMSE'], 'lower'))
summary_rows.append(metric_summary_row('MAE', meta['mc_MAE'], meta['hbv_MAE'], 'lower'))
summary_rows.append(metric_summary_row('abs_PBias', np.abs(meta['mc_PBias']), np.abs(meta['hbv_PBias']), 'lower'))
summary_rows.append(metric_summary_row('abs_FHV', np.abs(meta['mc_FHV']), np.abs(meta['hbv_FHV']), 'lower'))
summary_rows.append(metric_summary_row('abs_FLV', np.abs(meta['mc_FLV']), np.abs(meta['hbv_FLV']), 'lower'))
summary_rows.append(metric_summary_row('abs_FMS', np.abs(meta['mc_FMS']), np.abs(meta['hbv_FMS']), 'lower'))
summary_rows.append(metric_summary_row('PT', meta['mc_PT'], meta['hbv_PT'], 'lower'))
summary_rows.append(metric_summary_row('precision_2yr', meta['mc_precision_2yr'], meta['hbv_precision_2yr'], 'higher'))
summary_rows.append(metric_summary_row('recall_2yr', meta['mc_recall_2yr'], meta['hbv_recall_2yr'], 'higher'))
summary_rows.append(metric_summary_row('f1_2yr', meta['mc_f1_2yr'], meta['hbv_f1_2yr'], 'higher'))
summary_rows.append(metric_summary_row('precision_5yr', meta['mc_precision_5yr'], meta['hbv_precision_5yr'], 'higher'))
summary_rows.append(metric_summary_row('recall_5yr', meta['mc_recall_5yr'], meta['hbv_recall_5yr'], 'higher'))
summary_rows.append(metric_summary_row('f1_5yr', meta['mc_f1_5yr'], meta['hbv_f1_5yr'], 'higher'))
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / 'metric_summary_table.csv', index=False)

summary_by_snow = meta.copy()
summary_by_snow['snow_bin'] = pd.cut(summary_by_snow['frac_snow'], bins=[-1e-9, 0.1, 0.25, 0.5, 1.0], labels=['<0.1', '0.1-0.25', '0.25-0.5', '>=0.5'])
snow_group = summary_by_snow.groupby('snow_bin', observed=False).agg(
    basin_count=('gage_id', 'count'),
    hbv_median_nse=('hbv_NSE', 'median'),
    mc_median_nse=('mc_NSE', 'median'),
    mc_wins=('winner_nse', lambda x: int(np.sum(x == 'Snow-SIMHYD-MC-Heter')))
).reset_index()
snow_group['mc_win_fraction'] = snow_group['mc_wins'] / snow_group['basin_count']
snow_group.to_csv(OUT / 'summary_by_snow_fraction_bin.csv', index=False)

huc_group = meta.groupby('huc', observed=False).agg(
    basin_count=('gage_id', 'count'),
    hbv_median_nse=('hbv_NSE', 'median'),
    mc_median_nse=('mc_NSE', 'median'),
    mc_wins=('winner_nse', lambda x: int(np.sum(x == 'Snow-SIMHYD-MC-Heter')))
).reset_index()
huc_group['mc_win_fraction'] = huc_group['mc_wins'] / huc_group['basin_count']
huc_group.to_csv(OUT / 'summary_by_huc.csv', index=False)

# CDF
fig, ax = plt.subplots(figsize=(6.5, 4.5))
x_h, y_h = ecdf(meta['hbv_NSE'])
x_m, y_m = ecdf(meta['mc_NSE'])
ax.plot(x_h, y_h, lw=2.2, color='#2f6db3', label='HBV')
ax.plot(x_m, y_m, lw=2.2, color='#2a9d5b', label='Snow-SIMHYD-MC-Heter')
ax.set_xlabel('Basin NSE')
ax.set_ylabel('CDF')
ax.set_xlim(0.2, 1.0)
ax.grid(alpha=0.25)
ax.legend(frameon=False)
ax.set_title('CDF Of Basin NSE')
fig.tight_layout()
fig.savefig(OUT / 'cdf_basin_nse_hbv_vs_snowsimhydmc_heter.png', dpi=220)
plt.close(fig)

# Histogram overlay
fig, ax = plt.subplots(figsize=(7, 4.5))
bins = np.linspace(-1, 1, 41)
ax.hist(meta['hbv_NSE'].dropna(), bins=bins, alpha=0.55, label='HBV', color='#2f6db3')
ax.hist(meta['mc_NSE'].dropna(), bins=bins, alpha=0.55, label='Snow-SIMHYD-MC-Heter', color='#2a9d5b')
ax.set_xlabel('Basin NSE')
ax.set_ylabel('Count')
ax.grid(alpha=0.25)
ax.legend(frameon=False)
ax.set_title('NSE Histogram')
fig.tight_layout()
fig.savefig(OUT / 'hist_basin_nse_hbv_vs_snowsimhydmc_heter.png', dpi=220)
plt.close(fig)

# Side-by-side bins
labels = ['<0', '0-0.2', '0.2-0.4', '0.4-0.5', '0.5-0.65', '0.65-0.8', '>=0.8']
edges = [-1e9, 0, 0.2, 0.4, 0.5, 0.65, 0.8, 1e9]
hbv_bins = pd.cut(meta['hbv_NSE'], bins=edges, labels=labels, include_lowest=True).value_counts().reindex(labels, fill_value=0)
mc_bins = pd.cut(meta['mc_NSE'], bins=edges, labels=labels, include_lowest=True).value_counts().reindex(labels, fill_value=0)
bin_df = pd.DataFrame({'bin': labels, 'HBV': hbv_bins.values, 'SnowSIMHYDMCHeter': mc_bins.values})
bin_df.to_csv(OUT / 'test_nse_bins_hbv_vs_snowsimhydmc_heter.csv', index=False)

fig, ax = plt.subplots(figsize=(9, 4.8))
x = np.arange(len(labels))
w = 0.38
b1 = ax.bar(x - w/2, bin_df['HBV'], width=w, color='#2f6db3', label='HBV')
b2 = ax.bar(x + w/2, bin_df['SnowSIMHYDMCHeter'], width=w, color='#2a9d5b', label='Snow-SIMHYD-MC-Heter')
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 3, str(int(h)), ha='center', va='bottom', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Basin Count')
ax.set_title('NSE Bin Comparison')
ax.legend(frameon=False)
ax.grid(axis='y', alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / 'test_nse_bins_side_by_side.png', dpi=220)
plt.close(fig)

# Spatial maps
nse_labels = ['<0', '0-0.2', '0.2-0.4', '0.4-0.5', '0.5-0.65', '0.65-0.8', '>=0.8']
nse_edges = [-1e9, 0, 0.2, 0.4, 0.5, 0.65, 0.8, 1e9]
nse_colors = ['#7f1d1d', '#c96f5b', '#d7b66f', '#b9cf7a', '#6eb388', '#2e8b57', '#14532d']
size_map = {'<0': 18, '0-0.2': 24, '0.2-0.4': 30, '0.4-0.5': 38, '0.5-0.65': 48, '0.65-0.8': 58, '>=0.8': 72}
plot_binned_spatial(meta, 'hbv_NSE', 'HBV NSE Over CONUS', OUT / 'spatial_hbv_nse_conus_binned.png', nse_edges, nse_labels, nse_colors, size_map)
plot_binned_spatial(meta, 'mc_NSE', 'Snow-SIMHYD-MC-Heter NSE Over CONUS', OUT / 'spatial_snowsimhydmc_heter_nse_conus_binned.png', nse_edges, nse_labels, nse_colors, size_map)

diff_labels = ['<-0.2', '-0.2 to -0.05', '-0.05 to 0.05', '0.05 to 0.2', '>0.2']
diff_edges = [-1e9, -0.2, -0.05, 0.05, 0.2, 1e9]
diff_colors = ['#8b0000', '#d27d2d', '#d9d9d9', '#5fa8d3', '#0b5fa5']
diff_size_map = {'<-0.2': 60, '-0.2 to -0.05': 42, '-0.05 to 0.05': 24, '0.05 to 0.2': 42, '>0.2': 60}
plot_binned_spatial(meta, 'nse_diff_mc_minus_hbv', 'Snow-SIMHYD-MC-Heter NSE Minus HBV NSE', OUT / 'spatial_nse_difference_heter_minus_hbv_binned.png', diff_edges, diff_labels, diff_colors, diff_size_map)

fig, ax = plt.subplots(figsize=(10.5, 5.5))
ax.set_facecolor('#f6f3ee')
mask_mc = meta['winner_nse'] == 'Snow-SIMHYD-MC-Heter'
mask_hbv = ~mask_mc
ax.scatter(meta.loc[mask_hbv, 'lon'], meta.loc[mask_hbv, 'lat'], s=26, c='#2f6db3', alpha=0.85, label=f'HBV wins (n={int(mask_hbv.sum())})', edgecolors='black', linewidths=0.15)
ax.scatter(meta.loc[mask_mc, 'lon'], meta.loc[mask_mc, 'lat'], s=26, c='#2a9d5b', alpha=0.85, label=f'Snow-SIMHYD-MC-Heter wins (n={int(mask_mc.sum())})', edgecolors='black', linewidths=0.15)
ax.set_title('Per-Basin NSE Winner Map')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.grid(alpha=0.2)
ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))
fig.tight_layout()
fig.savefig(OUT / 'spatial_winner_map_nse.png', dpi=220, bbox_inches='tight')
plt.close(fig)

# Win count and difference histogram
fig, ax = plt.subplots(figsize=(5.8, 4.2))
counts = pd.Series({'HBV': int(mask_hbv.sum()), 'Snow-SIMHYD-MC-Heter': int(mask_mc.sum())})
bars = ax.bar(counts.index, counts.values, color=['#2f6db3', '#2a9d5b'])
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(int(bar.get_height())), ha='center')
ax.set_ylabel('Basin Count')
ax.set_title('How Many Basins Each Model Wins')
fig.tight_layout()
fig.savefig(OUT / 'basin_win_counts.png', dpi=220)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.hist(meta['nse_diff_mc_minus_hbv'].dropna(), bins=40, color='#6f9db8', edgecolor='white')
ax.axvline(0, color='black', lw=1.2, ls='--')
ax.set_xlabel('Snow-SIMHYD-MC-Heter NSE - HBV NSE')
ax.set_ylabel('Count')
ax.set_title('Per-Basin NSE Difference')
fig.tight_layout()
fig.savefig(OUT / 'nse_difference_histogram.png', dpi=220)
plt.close(fig)

# Regional plots
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.bar(snow_group['snow_bin'].astype(str), snow_group['mc_win_fraction'], color='#2a9d5b')
ax.set_ylim(0, 1)
ax.set_ylabel('MC Win Fraction')
ax.set_title('Snow-SIMHYD-MC-Heter Win Fraction By Snow Fraction Bin')
for i, v in enumerate(snow_group['mc_win_fraction']):
    ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
fig.tight_layout()
fig.savefig(OUT / 'win_fraction_by_snow_fraction_bin.png', dpi=220)
plt.close(fig)

top_huc = huc_group.sort_values('basin_count', ascending=False).head(12).sort_values('mc_win_fraction', ascending=False)
fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.bar(top_huc['huc'].astype(str), top_huc['mc_win_fraction'], color='#2a9d5b')
ax.set_ylim(0, 1)
ax.set_ylabel('MC Win Fraction')
ax.set_title('Snow-SIMHYD-MC-Heter Win Fraction By HUC (Top 12 By Basin Count)')
ax.tick_params(axis='x', rotation=25)
fig.tight_layout()
fig.savefig(OUT / 'win_fraction_by_huc.png', dpi=220)
plt.close(fig)

top_mc = meta.sort_values('nse_diff_mc_minus_hbv', ascending=False).head(20)
top_hbv = meta.sort_values('nse_diff_mc_minus_hbv', ascending=True).head(20)
top_mc.to_csv(OUT / 'top20_basins_where_heter_beats_hbv.csv', index=False)
top_hbv.to_csv(OUT / 'top20_basins_where_hbv_beats_heter.csv', index=False)

extract_learned_parameters(meta)

with open(OUT / 'report.md', 'w') as f:
    f.write('# Snow-SIMHYD-MC-Heter vs HBV Detailed Report\n\n')
    f.write(f'- HBV median NSE: {np.nanmedian(meta["hbv_NSE"]):.4f}\n')
    f.write(f'- Snow-SIMHYD-MC-Heter median NSE: {np.nanmedian(meta["mc_NSE"]):.4f}\n')
    f.write(f'- Snow-SIMHYD-MC-Heter wins on NSE in {int(mask_mc.sum())} / {len(meta)} basins\n')
    f.write(f'- HBV wins on NSE in {int(mask_hbv.sum())} / {len(meta)} basins\n')
