#!/usr/bin/env python3
from pathlib import Path
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/mircore/Desktop/diff_hydro')
CODE_ROOT = ROOT / 'code' / 'dPLHBVrelease' / 'hydroDL-dev'
sys.path.append(str(CODE_ROOT))

from hydroDL import utils
from hydroDL.data import camels
from hydroDL.master import loadModel

GPU_ID = int(os.environ.get('SNOWSIMHYDMC_HETER_ANALYSIS_GPU_ID', '1'))
EPOCH = int(os.environ.get('SNOWSIMHYDMC_HETER_ANALYSIS_EPOCH', '14'))
FOR_TYPE = 'daymet'
OUT = ROOT / 'outputs' / f'report_snowsimhydmc_heter_states_fluxes_ep{EPOCH}'
OUT.mkdir(parents=True, exist_ok=True)
MAP_DIR = OUT / 'maps'
MAP_DIR.mkdir(parents=True, exist_ok=True)
FLUX_MAP_DIR = MAP_DIR / 'flux_means'
FLUX_MAP_DIR.mkdir(parents=True, exist_ok=True)
PARAM_MAP_DIR = MAP_DIR / 'parameters'
PARAM_MAP_DIR.mkdir(parents=True, exist_ok=True)
SCATTER_DIR = OUT / 'scatter'
SCATTER_DIR.mkdir(parents=True, exist_ok=True)
TS_DIR = OUT / 'timeseries'
TS_DIR.mkdir(parents=True, exist_ok=True)

RUN_DIR = (
    ROOT / 'outputs' / 'rnnStreamflow' / 'CAMELSSNOWSIMHYDMC_HETER'
    / 'dPLSnowSIMHYDMC_Heter' / 'AllBasins' / FOR_TYPE / '111111'
    / 'T_19801001_19951001_BS_32_HS_64_RHO_365_Buff_365_Mul_4_Route_1_CmpW_1_LGDyn_1_All671_BS32_HS64_MaxIter100'
)
RESULT_DIR = (
    ROOT / 'outputs' / 'rnnStreamflow' / 'CAMELSSNOWSIMHYDMC_HETER'
    / 'dPLSnowSIMHYDMC_Heter' / 'AllBasins' / FOR_TYPE / '111111'
    / f'Train19801001_19951001Test19951001_20101001_SnowSIMHYDMC_HeterAll671_BS32_HS64_MaxIter100_Ep{EPOCH}'
)
HBV_DIR = (
    ROOT / 'outputs' / 'rnnStreamflow' / 'CAMELSDemo' / 'dPLHBV' / 'ALL'
    / 'Testforc' / FOR_TYPE / 'BuffOpt0' / 'RMSE_para0.25' / '111111'
    / 'Train19801001_19951001Test19951001_20101001Buff5478Nmul16_HBVAll671_BS32_HS64_MaxIter100'
)

TTRAIN = [19801001, 19951001]
TINV = [19801001, 19951001]
TTEST = [19951001, 20101001]
LOWFLOW_THR = 0.1
ATTR_VARS = [
    'p_mean', 'pet_mean', 'p_seasonality', 'frac_snow', 'aridity', 'high_prec_freq', 'high_prec_dur',
    'low_prec_freq', 'low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
    'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50',
    'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
    'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'geol_1st_class', 'glim_1st_class_frac',
    'geol_2nd_class', 'glim_2nd_class_frac', 'carbonate_rocks_frac', 'geol_porostiy', 'geol_permeability'
]
PARAM_NAMES = ['INSC', 'COEF', 'SQ', 'SMSC', 'SUB', 'CRAK', 'K', 'LG', 'TT', 'CFMAX', 'CFR', 'CWH']
PARAM_RANGES = {
    'INSC': (0.5, 5.0),
    'COEF': (50.0, 400.0),
    'SQ': (0.0, 6.0),
    'SMSC': (50.0, 500.0),
    'SUB': (0.0, 1.0),
    'CRAK': (0.0, 1.0),
    'K': (0.003, 0.3),
    'LG': (0.0, 1.0),
    'TT': (-2.5, 2.5),
    'CFMAX': (0.5, 10.0),
    'CFR': (0.0, 0.1),
    'CWH': (0.0, 0.2),
    'route_a': (0.0, 2.9),
    'route_b': (0.0, 6.5),
}


def pos(x, eps=1e-4):
    return 0.5 * (x + np.sqrt(x * x + eps ** 2))


def min_smooth(a, b, eps=1e-4):
    return a - pos(a - b, eps=eps)


def safe_div(a, b):
    out = np.full_like(a, np.nan, dtype=np.float32)
    mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-8)
    out[mask] = a[mask] / b[mask]
    return out


def calc_nse(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if np.sum(mask) < 2:
        return np.nan
    o = obs[mask]
    s = sim[mask]
    den = np.sum((o - np.mean(o)) ** 2)
    if den <= 0:
        return np.nan
    return 1.0 - np.sum((o - s) ** 2) / den


def calc_log_nse(obs, sim, eps=1e-3):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if np.sum(mask) < 2:
        return np.nan
    lo = np.log(np.clip(obs[mask], eps, None))
    ls = np.log(np.clip(sim[mask], eps, None))
    den = np.sum((lo - np.mean(lo)) ** 2)
    if den <= 0:
        return np.nan
    return 1.0 - np.sum((lo - ls) ** 2) / den


def pearson_r(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if np.sum(mask) < 2:
        return np.nan
    o = obs[mask]
    s = sim[mask]
    so = np.std(o)
    ss = np.std(s)
    if so == 0 or ss == 0:
        return np.nan
    return float(np.corrcoef(o, s)[0, 1])


def calc_kge(obs, sim):
    r = pearson_r(obs, sim)
    mask = np.isfinite(obs) & np.isfinite(sim)
    if np.sum(mask) < 2:
        return np.nan
    o = obs[mask]
    s = sim[mask]
    alpha = np.std(s) / np.std(o) if np.std(o) > 0 else np.nan
    beta = np.mean(s) / np.mean(o) if np.mean(o) != 0 else np.nan
    if np.isfinite(r) and np.isfinite(alpha) and np.isfinite(beta):
        return 1.0 - math.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return np.nan


def lowflow_nse(obs, sim, q=0.3):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if np.sum(mask) < 2:
        return np.nan
    thr = np.quantile(obs[mask], q)
    sub = mask & (obs <= thr)
    if np.sum(sub) < 2:
        return np.nan
    return calc_nse(obs[sub], sim[sub])


def dry_day_accuracy(obs, sim, thr=LOWFLOW_THR):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if np.sum(mask) == 0:
        return np.nan
    return float(np.mean((obs[mask] <= thr) == (sim[mask] <= thr)))


def recession_slope(series):
    q0 = series[:-1]
    q1 = series[1:]
    mask = np.isfinite(q0) & np.isfinite(q1) & (q0 > 1e-4) & (q1 > 1e-4) & (q1 < q0)
    if np.sum(mask) < 5:
        return np.nan
    return float(np.median(np.log(q1[mask] / q0[mask])))


def plot_continuous_map(df, col, title, out_path, cmap='viridis'):
    vals = df[col].to_numpy(dtype=float)
    mask = np.isfinite(vals)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.set_facecolor('#f6f3ee')
    sc = ax.scatter(df.loc[mask, 'lon'], df.loc[mask, 'lat'], c=vals[mask], s=30, cmap=cmap,
                    edgecolors='black', linewidths=0.15, alpha=0.92)
    ax.set_title(f'{title}\nmin={float(np.nanmin(vals)):.4g}, max={float(np.nanmax(vals)):.4g}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(alpha=0.2)
    fig.colorbar(sc, ax=ax, shrink=0.86)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    camels.initcamels(str(ROOT / 'Camels'))
    g = camels.gageDict
    gage_ids = g['id'].astype(int)
    gage_lst = gage_ids.tolist()
    meta = pd.DataFrame({
        'gage_id': gage_ids,
        'gage_name': g['name'],
        'huc': g['huc'].astype(int),
        'lat': g['lat'].astype(float),
        'lon': g['lon'].astype(float),
        'area_km2': g['area'].astype(float),
    })

    dfAttr = camels.DataframeCamels(tRange=TTRAIN, subset=gage_lst, forType=FOR_TYPE)
    attr_small = ['frac_snow', 'aridity', 'elev_mean', 'frac_forest', 'soil_depth_statsgo']
    attrs_small = dfAttr.getDataConst(varLst=attr_small, doNorm=False, rmNan=False)
    for j, name in enumerate(attr_small):
        meta[name] = attrs_small[:, j]

    dfTrain = camels.DataframeCamels(tRange=TTRAIN, subset=gage_lst, forType=FOR_TYPE)
    forcTrain = dfTrain.getDataTs(varLst=['prcp', 'tmean'], doNorm=False, rmNan=False).astype(np.float32)
    dfInv = camels.DataframeCamels(tRange=TINV, subset=gage_lst, forType=FOR_TYPE)
    forcInv = dfInv.getDataTs(varLst=['prcp', 'tmean'], doNorm=False, rmNan=False).astype(np.float32)
    attrsUN = dfInv.getDataConst(varLst=ATTR_VARS, doNorm=False, rmNan=False).astype(np.float32)
    dfTest = camels.DataframeCamels(tRange=TTEST, subset=gage_lst, forType=FOR_TYPE)
    forcTest = dfTest.getDataTs(varLst=['prcp', 'tmean'], doNorm=False, rmNan=False).astype(np.float32)
    obsTest = dfTest.getDataObs(doNorm=False, rmNan=False, basinnorm=False).astype(np.float32)

    areas = g['area'].astype(np.float32)
    area_tile = np.tile(areas[:, None, None], (1, obsTest.shape[1], 1))
    obsTest = (obsTest * 0.0283168 * 3600 * 24) / (area_tile * (10 ** 6)) * 10 ** 3
    obsTest = obsTest[:, :, 0].astype(np.float32)

    varLstNL = ['PEVAP']
    tPETRange = [19800101, 20150101]
    tPETLst = utils.time.tRange2Array(tPETRange)
    PETDir = str(ROOT / 'Camels' / 'pet_harg' / FOR_TYPE) + '/'
    ntime = len(tPETLst)
    PETfull = np.empty([len(gage_lst), ntime, len(varLstNL)], dtype=np.float32)
    for k in range(len(gage_lst)):
        PETfull[k, :, :] = camels.readcsvGage(PETDir, gage_lst[k], varLstNL, ntime)
    TtrainLst = utils.time.tRange2Array(TTRAIN)
    TinvLst = utils.time.tRange2Array(TINV)
    TtestLst = utils.time.tRange2Array(TTEST)
    _, _, ind2 = np.intersect1d(TtrainLst, tPETLst, return_indices=True)
    PETTrain = PETfull[:, ind2, :]
    _, _, ind2inv = np.intersect1d(TinvLst, tPETLst, return_indices=True)
    PETInv = PETfull[:, ind2inv, :]
    _, _, ind2test = np.intersect1d(TtestLst, tPETLst, return_indices=True)
    PETTest = PETfull[:, ind2test, :]

    xTrain = np.concatenate([forcTrain, PETTrain], axis=2).astype(np.float32)
    xTest = np.concatenate([forcTest, PETTest], axis=2).astype(np.float32)
    xTrain[np.isnan(xTrain)] = 0.0
    xTest[np.isnan(xTest)] = 0.0

    with open(RUN_DIR / 'statDict.json', 'r') as fp:
        statDict = json.load(fp)
    series_inv = np.concatenate([forcInv, PETInv], axis=2)
    attr_norm = camels.transNormbyDic(attrsUN, ATTR_VARS, statDict, toNorm=True).astype(np.float32)
    attr_norm[np.isnan(attr_norm)] = 0.0
    series_test = np.concatenate([forcTest, PETTest], axis=2)
    series_test_norm = camels.transNormbyDic(series_test, ['prcp', 'tmean', 'pet'], statDict, toNorm=True).astype(np.float32)
    series_test_norm[np.isnan(series_test_norm)] = 0.0
    cTemp = np.repeat(attr_norm[:, None, :], series_test_norm.shape[1], axis=1)
    zTest = np.concatenate([series_test_norm, cTemp], axis=2)

    model = loadModel(str(RUN_DIR), epoch=EPOCH)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.set_device(GPU_ID)
        model = model.cuda(GPU_ID)

    z_tensor = torch.from_numpy(np.swapaxes(zTest, 1, 0)).float()

    static_chunks = []
    route_a_chunks = []
    route_b_chunks = []
    weight_chunks = []
    lg_dyn_chunks = []
    chunk_size = 64
    with torch.no_grad():
        for i0 in range(0, len(gage_lst), chunk_size):
            i1 = min(i0 + chunk_size, len(gage_lst))
            z_part = z_tensor[:, i0:i1, :]
            if torch.cuda.is_available():
                z_part = z_part.cuda(GPU_ID)
            basin_attr = z_part[-1, :, -model.nattr:]
            static_feat = model.staticFeat(basin_attr)
            static_params0 = model.staticOut(static_feat)
            cursor = 0
            static0 = static_params0[:, cursor:cursor + model.nstaticpm].view(i1 - i0, model.nfea, model.nmul)
            static0 = static0 + model.compStaticBias
            snowpara = torch.sigmoid(static0)
            theta = snowpara.permute(0, 2, 1).contiguous().view((i1 - i0) * model.nmul, model.nfea)
            theta_denorm_part = model.simhyd.denorm_params(theta).detach().cpu().numpy().reshape(i1 - i0, model.nmul, model.nfea).astype(np.float32)
            static_chunks.append(theta_denorm_part)

            cursor += model.nstaticpm
            rout0 = static_params0[:, cursor:cursor + model.nroutpm]
            route_sig = torch.sigmoid(rout0).detach().cpu().numpy().astype(np.float32)
            route_a_chunks.append(route_sig[:, 0] * 2.9)
            route_b_chunks.append(route_sig[:, 1] * 6.5)
            cursor += model.nroutpm

            w0 = static_params0[:, cursor:cursor + model.nwtspm] + model.compWeightBias
            weights_part = torch.softmax(w0, dim=-1).detach().cpu().numpy().astype(np.float32)
            weight_chunks.append(weights_part)

            lg_dyn_seq = torch.sigmoid(model.lstmdyn(z_part) + model.lgAttr(basin_attr).unsqueeze(0).repeat(z_part.shape[0], 1, 1))
            lg_dyn_chunks.append(lg_dyn_seq.detach().cpu().numpy().astype(np.float32))

            del z_part, basin_attr, static_feat, static_params0, static0, snowpara, theta, lg_dyn_seq
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    theta_denorm = np.concatenate(static_chunks, axis=0)
    route_a = np.concatenate(route_a_chunks, axis=0)
    route_b = np.concatenate(route_b_chunks, axis=0)
    weights = np.concatenate(weight_chunks, axis=0)
    lg_dyn = np.concatenate(lg_dyn_chunks, axis=1)

    weighted_params = np.sum(theta_denorm * weights[:, :, None], axis=1)
    meta_params = meta.copy()
    for i, pname in enumerate(PARAM_NAMES):
        meta_params[f'{pname}_wavg'] = weighted_params[:, i]
        plot_continuous_map(meta_params, f'{pname}_wavg', f'{pname} weighted average', PARAM_MAP_DIR / f'{pname}_wavg.png')
    meta_params['route_a'] = route_a
    meta_params['route_b'] = route_b
    meta_params['lg_dyn_mean_wavg'] = np.sum(lg_dyn.mean(axis=0) * weights, axis=1)
    meta_params['lg_dyn_std_wavg'] = np.sum(lg_dyn.std(axis=0) * weights, axis=1)
    for name in ['route_a', 'route_b', 'lg_dyn_mean_wavg', 'lg_dyn_std_wavg']:
        plot_continuous_map(meta_params, name, name, PARAM_MAP_DIR / f'{name}.png')

    ng = len(gage_lst)
    nm = model.nmul
    nc = ng * nm
    theta_flat = theta_denorm.reshape(nc, model.nfea)
    weights_flat = weights.reshape(ng, nm)

    xTrain_rep = np.repeat(xTrain[:, None, :, :], nm, axis=1).reshape(nc, xTrain.shape[1], 3)
    xTest_rep = np.repeat(xTest[:, None, :, :], nm, axis=1).reshape(nc, xTest.shape[1], 3)
    lg_flat = np.transpose(lg_dyn, (1, 2, 0)).reshape(nc, xTest.shape[1], 1)

    INSC, COEF, SQ, SMSC, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH = [theta_flat[:, i:i+1] for i in range(12)]
    w_flat = weights_flat.reshape(ng, nm)

    def run_period(inputs, lg_dynamic=None, save=False, init_state=None):
        B, Tlen, _ = inputs.shape
        if init_state is None:
            SMS = np.zeros((B, 1), dtype=np.float32)
            GW = np.zeros((B, 1), dtype=np.float32)
            SNOWPACK = np.zeros((B, 1), dtype=np.float32)
            MELTWATER = np.zeros((B, 1), dtype=np.float32)
        else:
            SMS, GW, SNOWPACK, MELTWATER = [arr.copy().astype(np.float32) for arr in init_state]

        saved = {}
        if save:
            names = [
                'snowpack', 'soil_moisture', 'groundwater', 'snowfall', 'rainfall', 'snowmelt',
                'interception_evaporation', 'actual_et', 'infiltration', 'recharge_to_groundwater',
                'surface_runoff', 'interflow', 'baseflow', 'groundwater_loss', 'total_unrouted_q',
                'interception_storage', 'delta_storage'
            ]
            for n in names:
                saved[n] = np.zeros((ng, Tlen), dtype=np.float32)

        for t in range(Tlen):
            Pt = pos(inputs[:, t:t+1, 0])
            Tt = inputs[:, t:t+1, 1]
            E0t = pos(inputs[:, t:t+1, 2])

            SMS0 = min_smooth(pos(SMS), SMSC)
            GW0 = pos(GW)
            SNOWPACK0 = pos(SNOWPACK)
            MELTWATER0 = pos(MELTWATER)
            total_store0 = SMS0 + GW0 + SNOWPACK0 + MELTWATER0

            frac_rain = 1.0 / (1.0 + np.exp(-5.0 * (Tt - TT)))
            RAIN = Pt * frac_rain
            SNOW = Pt * (1.0 - frac_rain)
            SNOWPACK1 = SNOWPACK0 + SNOW

            melt_pot = CFMAX * pos(Tt - TT)
            melt = min_smooth(melt_pot, SNOWPACK1)
            MELTWATER1 = MELTWATER0 + melt
            SNOWPACK2 = SNOWPACK1 - melt

            refreeze_pot = CFR * CFMAX * pos(TT - Tt)
            refreezing = min_smooth(refreeze_pot, MELTWATER1)
            SNOWPACK3 = SNOWPACK2 + refreezing
            MELTWATER2 = MELTWATER1 - refreezing

            water_holding = CWH * SNOWPACK3
            tosoil = pos(MELTWATER2 - water_holding)
            MELTWATER_next = pos(MELTWATER2 - tosoil)
            SNOWPACK_next = pos(SNOWPACK3)

            Peff = RAIN + tosoil
            IMAX = min_smooth(INSC, E0t)
            INT = min_smooth(IMAX, Peff)
            INR = pos(Peff - INT)
            wetness = SMS0 / (SMSC + 1e-8)
            infil_cap = COEF * np.exp(-SQ * wetness)
            RMO = min_smooth(infil_cap, INR)
            IRUN = pos(INR - RMO)
            SRUN = SUB * wetness * RMO
            REC = pos(CRAK * wetness * (RMO - SRUN))
            SMF = pos(RMO - SRUN - REC)
            POT = pos(E0t - INT)
            ETS_cap = 10.0 * wetness
            ETS = min_smooth(min_smooth(ETS_cap, POT), SMS0 + SMF)
            SMS_pre = SMS0 + SMF - ETS
            SOIL_EXCESS = pos(SMS_pre - SMSC)
            SMS_next = pos(SMS_pre - SOIL_EXCESS)
            REC_total = REC + SOIL_EXCESS
            BAS = K * GW0
            if lg_dynamic is None:
                LG_t = LG
            else:
                LG_t = (1.0 - model.lgdynweight) * LG + model.lgdynweight * lg_dynamic[:, t:t+1, 0]
            GW_next = pos(GW0 + REC_total - BAS - LG_t)
            Q = pos(IRUN + SRUN + BAS)

            total_store1 = SMS_next + GW_next + SNOWPACK_next + MELTWATER_next

            if save:
                for name, comp_arr in {
                    'snowpack': SNOWPACK_next,
                    'soil_moisture': SMS_next,
                    'groundwater': GW_next,
                    'snowfall': SNOW,
                    'rainfall': RAIN,
                    'snowmelt': melt,
                    'interception_evaporation': INT,
                    'actual_et': ETS,
                    'infiltration': RMO,
                    'recharge_to_groundwater': REC_total,
                    'surface_runoff': IRUN,
                    'interflow': SRUN,
                    'baseflow': BAS,
                    'groundwater_loss': LG_t,
                    'total_unrouted_q': Q,
                    'interception_storage': np.zeros_like(INT),
                    'delta_storage': total_store1 - total_store0,
                }.items():
                    comp = comp_arr.reshape(ng, nm)
                    saved[name][:, t] = np.sum(comp * w_flat, axis=1).astype(np.float32)

            SMS, GW, SNOWPACK, MELTWATER = SMS_next, GW_next, SNOWPACK_next, MELTWATER_next

        return (SMS, GW, SNOWPACK, MELTWATER), saved

    warm_state, _ = run_period(xTrain_rep, lg_dynamic=None, save=False, init_state=None)
    _, saved = run_period(xTest_rep, lg_dynamic=lg_flat, save=True, init_state=warm_state)

    q_mix = saved['total_unrouted_q'].T[:, :, None]
    q_mix_t = torch.from_numpy(q_mix).float()
    route_t = torch.from_numpy(np.stack([route_a / 2.9, route_b / 6.5], axis=1)).float()
    with torch.no_grad():
        if torch.cuda.is_available():
            q_mix_t = q_mix_t.cuda(GPU_ID)
            route_t = route_t.cuda(GPU_ID)
            model = model.cuda(GPU_ID)
        q_routed = model._route_q(q_mix_t, route_t).detach().cpu().numpy()[:, :, 0].T.astype(np.float32)
    saved['channel_loss'] = np.zeros_like(saved['total_unrouted_q'], dtype=np.float32)
    saved['total_simulated_discharge'] = q_routed

    P = xTest[:, :, 0].astype(np.float32)
    Q = saved['total_simulated_discharge']
    ET = saved['actual_et']
    BFI = safe_div(saved['baseflow'], Q)
    gw_contrib = safe_div(saved['baseflow'], saved['surface_runoff'] + saved['interflow'] + saved['baseflow'])
    gw_loss_frac = safe_div(saved['groundwater_loss'], saved['baseflow'] + saved['groundwater_loss'])
    runoff_ratio = safe_div(Q, P)
    et_ratio = safe_div(ET, P)
    wb_closure = P - ET - Q - saved['groundwater_loss'] - saved['delta_storage']

    derived = {
        'groundwater_contribution_fraction': gw_contrib,
        'groundwater_loss_fraction': gw_loss_frac,
        'runoff_ratio': runoff_ratio,
        'et_ratio': et_ratio,
        'baseflow_index': BFI,
        'water_balance_closure': wb_closure,
    }

    np.savez_compressed(
        OUT / 'states_fluxes_ep14.npz',
        basin_ids=np.array(gage_lst, dtype=np.int32),
        dates=np.array(pd.date_range('1995-10-01', periods=xTest.shape[1], freq='D').astype(str)),
        obs_q=obsTest.astype(np.float32),
        **{k: v.astype(np.float32) for k, v in saved.items()},
        **{k: v.astype(np.float32) for k, v in derived.items()},
    )

    basin = meta_params.copy()
    for k, v in saved.items():
        basin[f'{k}_mean'] = np.nanmean(v, axis=1)
    for k, v in derived.items():
        basin[f'{k}_mean'] = np.nanmean(v, axis=1)

    basin['NSE'] = [calc_nse(obsTest[i], Q[i]) for i in range(ng)]
    basin['KGE'] = [calc_kge(obsTest[i], Q[i]) for i in range(ng)]
    basin['logNSE'] = [calc_log_nse(obsTest[i], Q[i]) for i in range(ng)]
    basin['lowflow_NSE'] = [lowflow_nse(obsTest[i], Q[i]) for i in range(ng)]
    basin['dry_day_accuracy'] = [dry_day_accuracy(obsTest[i], Q[i]) for i in range(ng)]
    basin['recession_slope_obs'] = [recession_slope(obsTest[i]) for i in range(ng)]
    basin['recession_slope_sim'] = [recession_slope(Q[i]) for i in range(ng)]
    basin['recession_slope_error'] = np.abs(basin['recession_slope_sim'] - basin['recession_slope_obs'])
    basin['water_balance_error_mean'] = np.nanmean(np.abs(wb_closure), axis=1)

    def lh_baseflow(q, alpha=0.925, passes=3):
        q = np.asarray(q, dtype=np.float64)
        q = np.clip(q, 0.0, None)
        bf = q.copy()
        for _ in range(passes):
            f = np.zeros_like(bf)
            for t in range(1, len(bf)):
                f[t] = alpha * f[t - 1] + (1 + alpha) / 2.0 * (bf[t] - bf[t - 1])
                if f[t] < 0:
                    f[t] = 0
                if f[t] > bf[t]:
                    f[t] = bf[t]
            bf = bf - f
            bf = np.clip(bf, 0.0, q)
        return bf

    obs_bfi = []
    for i in range(ng):
        bf_obs = lh_baseflow(obsTest[i])
        denom = np.nanmean(obsTest[i])
        obs_bfi.append(np.nanmean(bf_obs) / denom if denom > 0 else np.nan)
    basin['observed_bfi_lh'] = np.array(obs_bfi, dtype=np.float32)
    basin['sim_bfi_mean'] = basin['baseflow_index_mean']

    basin.to_csv(OUT / 'per_basin_states_fluxes_metrics.csv', index=False)

    # Parameter maps
    for name in [f'{p}_wavg' for p in PARAM_NAMES] + ['route_a', 'route_b', 'lg_dyn_mean_wavg', 'lg_dyn_std_wavg']:
        plot_continuous_map(basin, name, name, PARAM_MAP_DIR / f'{name}.png')

    # Basin-average flux maps
    flux_map_cols = [
        'snowfall_mean', 'rainfall_mean', 'snowmelt_mean', 'interception_evaporation_mean', 'actual_et_mean',
        'infiltration_mean', 'recharge_to_groundwater_mean', 'surface_runoff_mean', 'interflow_mean',
        'baseflow_mean', 'groundwater_loss_mean', 'total_simulated_discharge_mean',
        'groundwater_contribution_fraction_mean', 'groundwater_loss_fraction_mean',
        'runoff_ratio_mean', 'et_ratio_mean', 'baseflow_index_mean', 'water_balance_closure_mean'
    ]
    for col in flux_map_cols:
        plot_continuous_map(basin, col, col, FLUX_MAP_DIR / f'{col}.png')

    # Scatter plots
    scat_specs = [
        ('LG_wavg', 'aridity', 'LG vs aridity index', 'LG_vs_aridity.png'),
        ('K_wavg', 'baseflow_index_mean', 'K vs baseflow index', 'K_vs_baseflow_index.png'),
        ('TT_wavg', 'elev_mean', 'TT vs elevation', 'TT_vs_elevation.png'),
        ('INSC_wavg', 'frac_forest', 'INSC vs forest cover', 'INSC_vs_forest.png'),
        ('SMSC_wavg', 'soil_depth_statsgo', 'SMSC vs soil depth', 'SMSC_vs_soildepth.png'),
    ]
    for xcol, ycol, title, fname in scat_specs:
        fig, ax = plt.subplots(figsize=(5.4, 4.4))
        ax.scatter(basin[xcol], basin[ycol], s=18, alpha=0.65, color='#356b8c', edgecolors='none')
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(title)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(SCATTER_DIR / fname, dpi=220)
        plt.close(fig)

    # Observed baseflow separation scatter
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.scatter(basin['observed_bfi_lh'], basin['sim_bfi_mean'], s=18, alpha=0.65, color='#2a9d5b', edgecolors='none')
    ax.set_xlabel('Observed BFI (Lyne-Hollick)')
    ax.set_ylabel('Simulated BFI')
    ax.set_title('Simulated vs observed baseflow index')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(SCATTER_DIR / 'sim_vs_obs_bfi.png', dpi=220)
    plt.close(fig)

    # Correlation matrix
    corr_cols = [f'{p}_wavg' for p in PARAM_NAMES] + ['route_a', 'route_b', 'lg_dyn_mean_wavg', 'lg_dyn_std_wavg',
                'frac_snow', 'aridity', 'elev_mean', 'frac_forest', 'soil_depth_statsgo']
    corr = basin[corr_cols].corr()
    corr.to_csv(OUT / 'parameter_attribute_correlation_matrix.csv')
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(corr_cols)))
    ax.set_yticklabels(corr_cols, fontsize=7)
    ax.set_title('Correlation matrix: learned parameters and attributes')
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / 'parameter_attribute_correlation_matrix.png', dpi=220)
    plt.close(fig)

    # Selected basin time series
    sel = basin[['gage_id', 'gage_name', 'NSE']].sort_values('NSE')
    idxs = [sel.index[0], sel.index[len(sel) // 2], sel.index[-1]]
    dates = pd.date_range('1995-10-01', periods=xTest.shape[1], freq='D')
    for idx in idxs:
        row = basin.loc[idx]
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(dates, obsTest[idx], color='black', lw=1.0, label='Obs Q')
        axes[0].plot(dates, Q[idx], color='#2a9d5b', lw=1.0, label='Sim Q')
        axes[0].legend(frameon=False, ncol=2)
        axes[0].set_title(f'{int(row.gage_id)} {row.gage_name.strip()} | NSE={row.NSE:.3f}')
        axes[1].plot(dates, saved['snowpack'][idx], label='Snowpack', color='#4c72b0')
        axes[1].plot(dates, saved['soil_moisture'][idx], label='Soil moisture', color='#8c564b')
        axes[1].plot(dates, saved['groundwater'][idx], label='Groundwater', color='#2a9d8f')
        axes[1].legend(frameon=False, ncol=3)
        axes[2].plot(dates, saved['rainfall'][idx], label='Rainfall', color='#1f77b4')
        axes[2].plot(dates, saved['snowmelt'][idx], label='Snowmelt', color='#6baed6')
        axes[2].plot(dates, saved['actual_et'][idx], label='ET', color='#d95f0e')
        axes[2].legend(frameon=False, ncol=3)
        axes[3].plot(dates, saved['surface_runoff'][idx], label='Surface runoff', color='#e41a1c')
        axes[3].plot(dates, saved['interflow'][idx], label='Interflow', color='#ff7f00')
        axes[3].plot(dates, saved['baseflow'][idx], label='Baseflow', color='#377eb8')
        axes[3].plot(dates, saved['groundwater_loss'][idx], label='GW loss', color='#984ea3')
        axes[3].legend(frameon=False, ncol=4)
        for ax in axes:
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(TS_DIR / f'{int(row.gage_id)}.png', dpi=220)
        plt.close(fig)

    # Identifiability / variance / collapse
    ident_rows = []
    for p in [f'{x}_wavg' for x in PARAM_NAMES] + ['route_a', 'route_b', 'lg_dyn_mean_wavg', 'lg_dyn_std_wavg']:
        vals = basin[p].to_numpy(dtype=float)
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        std = float(np.nanstd(vals))
        if p in PARAM_RANGES:
            pr = PARAM_RANGES[p][1] - PARAM_RANGES[p][0]
            usage = (vmax - vmin) / pr if pr > 0 else np.nan
            collapsed = usage < 0.05
        else:
            usage = np.nan
            collapsed = std < 1e-3
        ident_rows.append({'parameter': p, 'min': vmin, 'max': vmax, 'std': std, 'range_usage_fraction': usage, 'collapsed_flag': collapsed})
    ident = pd.DataFrame(ident_rows)
    ident.to_csv(OUT / 'parameter_identifiability_summary.csv', index=False)
    ident[ident['collapsed_flag']].to_csv(OUT / 'collapsed_parameters.csv', index=False)

    # External data availability
    ext = pd.DataFrame([
        {'comparison': 'ET vs MODIS/GLEAM', 'available_local': False, 'note': 'No MODIS or GLEAM files found locally'},
        {'comparison': 'soil moisture vs SMAP/ESA-CCI', 'available_local': False, 'note': 'No SMAP or ESA-CCI files found locally'},
        {'comparison': 'snowpack vs SNODAS/SNOTEL', 'available_local': False, 'note': 'No SNODAS or SNOTEL files found locally'},
        {'comparison': 'groundwater storage vs GRACE', 'available_local': False, 'note': 'No GRACE files found locally'},
        {'comparison': 'baseflow index vs observed baseflow separation', 'available_local': True, 'note': 'Compared against Lyne-Hollick separation of observed Q'},
    ])
    ext.to_csv(OUT / 'external_product_availability.csv', index=False)

    # Report
    with open(OUT / 'report.md', 'w') as fp:
        fp.write(f'# Snow-SIMHYD-MC-Heter states/fluxes analysis for Ep{EPOCH}\n\n')
        fp.write('## Scope\n')
        fp.write('- Exported weighted basin-level internal states and fluxes for all 671 CAMELS basins over the test period.\n')
        fp.write('- Warmup used the full train period with static parameters, matching the saved Ep14 test setup.\n')
        fp.write('- `interception_storage` is not an explicit persistent state in this model and is exported as zero.\n')
        fp.write('- `channel_loss` is not an explicit process in this model and is exported as zero.\n\n')
        fp.write('## Aggregate metrics\n')
        fp.write(f"- Median NSE: {float(np.nanmedian(basin['NSE'])):.4f}\n")
        fp.write(f"- Median KGE: {float(np.nanmedian(basin['KGE'])):.4f}\n")
        fp.write(f"- Median logNSE: {float(np.nanmedian(basin['logNSE'])):.4f}\n")
        fp.write(f"- Median low-flow NSE: {float(np.nanmedian(basin['lowflow_NSE'])):.4f}\n")
        fp.write(f"- Median dry-day accuracy: {float(np.nanmedian(basin['dry_day_accuracy'])):.4f}\n")
        fp.write(f"- Median recession slope error: {float(np.nanmedian(basin['recession_slope_error'])):.4f}\n")
        fp.write(f"- Median mean water balance error: {float(np.nanmedian(basin['water_balance_error_mean'])):.4f}\n\n")
        fp.write('## Parameter variance / collapse\n')
        fp.write(f"- Collapsed parameter count: {int(ident['collapsed_flag'].sum())}\n")
        fp.write('- See `parameter_identifiability_summary.csv` and `collapsed_parameters.csv`.\n\n')
        fp.write('## External product comparisons\n')
        fp.write('- ET/soil-moisture/snowpack/GRACE comparisons could not be executed because those product files are not present locally.\n')
        fp.write('- Baseflow index comparison was performed against observed-flow Lyne-Hollick separation.\n')


if __name__ == '__main__':
    main()
