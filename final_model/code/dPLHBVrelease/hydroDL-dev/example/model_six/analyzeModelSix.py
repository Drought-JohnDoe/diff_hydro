import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.append('../../')
from hydroDL import utils
from hydroDL.data import camels
from hydroDL.master import loadModel


def add_bool_arg(parser, name, default=False):
    flag = '--' + name.replace('_', '-')
    parser.add_argument(flag, dest=name, action='store_true')
    parser.add_argument('--no-' + name.replace('_', '-'), dest=name, action='store_false')
    parser.set_defaults(**{name: default})


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Model Six diagnostics')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--rho', type=int, default=365)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--nmul', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--seed', type=int, default=111111)
    parser.add_argument('--forcing', type=str, default='daymet')
    parser.add_argument('--subset-limit', type=int, default=64)
    parser.add_argument('--chunk-size', type=int, default=32)
    parser.add_argument('--exp-info-suffix', type=str, default='')
    parser.add_argument('--max-iter-ep', type=int, default=200)
    parser.add_argument('--channel-loss-max', type=float, default=0.60)
    parser.add_argument('--zero-gate-hidden', type=int, default=None)

    add_bool_arg(parser, 'routing', default=True)
    add_bool_arg(parser, 'comprout', default=False)
    add_bool_arg(parser, 'compwts', default=True)
    add_bool_arg(parser, 'lgdyn', default=True)
    add_bool_arg(parser, 'dynamic_sq', default=True)
    add_bool_arg(parser, 'dynamic_etgam', default=True)
    add_bool_arg(parser, 'dynamic_partition', default=True)
    add_bool_arg(parser, 'dynamic_cfmax_snow', default=True)
    add_bool_arg(parser, 'dynamic_routing_scale', default=False)
    add_bool_arg(parser, 'dynamic_all', default=False)
    add_bool_arg(parser, 'component_routing', default=True)
    add_bool_arg(parser, 'dry_channel_loss', default=True)
    add_bool_arg(parser, 'zero_flow_gate', default=True)
    add_bool_arg(parser, 'use_all_basins', default=False)
    return parser.parse_args()


def seasonal_features(t_range):
    t_arr = utils.time.tRange2Array(t_range)
    dates = pd.to_datetime(t_arr.astype(str))
    doy = dates.dayofyear.to_numpy(dtype=np.float32)
    ang = 2.0 * np.pi * (doy - 1.0) / 365.0
    return np.stack([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)


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
    o = np.log(np.clip(obs[mask], eps, None))
    s = np.log(np.clip(sim[mask], eps, None))
    den = np.sum((o - np.mean(o)) ** 2)
    if den <= 0:
        return np.nan
    return 1.0 - np.sum((o - s) ** 2) / den


def calc_kge(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if np.sum(mask) < 2:
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


def lowflow_nse(obs, sim, q=0.3):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if np.sum(mask) < 2:
        return np.nan
    thr = np.quantile(obs[mask], q)
    sub = mask & (obs <= thr)
    if np.sum(sub) < 2:
        return np.nan
    return calc_nse(obs[sub], sim[sub])


def highflow_nse(obs, sim, q=0.7):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if np.sum(mask) < 2:
        return np.nan
    thr = np.quantile(obs[mask], q)
    sub = mask & (obs >= thr)
    if np.sum(sub) < 2:
        return np.nan
    return calc_nse(obs[sub], sim[sub])


def fdc_error(obs, sim, nq=25):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if np.sum(mask) < 5:
        return np.nan
    qs = np.linspace(0.02, 0.98, nq)
    oq = np.quantile(obs[mask], qs)
    sq = np.quantile(sim[mask], qs)
    scale = np.maximum(np.abs(oq), 1e-6)
    return np.sqrt(np.mean(((sq - oq) / scale) ** 2))


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


def safe_sum_ratio(num, den):
    den_sum = np.nansum(den, axis=1)
    num_sum = np.nansum(num, axis=1)
    out = np.full(num.shape[0], np.nan, dtype=np.float32)
    mask = np.isfinite(den_sum) & (np.abs(den_sum) > 1e-8)
    out[mask] = num_sum[mask] / den_sum[mask]
    return out


def plot_map(df, value_col, title, out_file, cmap='viridis'):
    vals = df[value_col].to_numpy()
    vmin = np.nanpercentile(vals, 2)
    vmax = np.nanpercentile(vals, 98)
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(df['lon'], df['lat'], c=vals, s=28, cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolors='k', linewidths=0.15)
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    cbar = fig.colorbar(sc, ax=ax, shrink=0.9)
    cbar.set_label(value_col)
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[5]
    root_database = os.environ.get('DYNAMIC_SIMHYD_ROOT_DB', str(project_root / 'Camels'))
    root_out = os.environ.get('DYNAMIC_SIMHYD_ROOT_OUT', str(project_root / 'outputs' / 'rnnStreamflow'))
    subset_file = os.environ.get(
        'DYNAMIC_SIMHYD_SUBSET_FILE',
        str(project_root / 'code' / 'dPLHBVrelease' / 'hydroDL-dev' / 'example' / 'model_six' / 'Sub531ID.txt'))

    t_train = [19801001, 19951001]
    t_inv = [19801001, 19951001]
    t_test = [19951001, 20101001]
    bufftime = 365
    camels.initcamels(root_database)

    gageinfo = camels.gageDict
    gageid = gageinfo['id']
    gageid_lst = gageid.tolist()
    if args.use_all_basins:
        basin_ids = gageid.tolist()
    else:
        with open(subset_file, 'r') as fp:
            subset_ids = json.load(fp)
        basin_ids = [int(x) for x in subset_ids[:args.subset_limit]]
    basin_ind = [gageid_lst.index(j) for j in basin_ids]

    if args.forcing == 'daymet':
        var_f = ['prcp', 'tmean']
        var_f_inv = ['prcp', 'tmean']
    else:
        var_f = ['prcp', 'tmax']
        var_f_inv = ['prcp', 'tmax']

    attr_lst = [
        'p_mean', 'pet_mean', 'p_seasonality', 'frac_snow', 'aridity', 'high_prec_freq', 'high_prec_dur',
        'low_prec_freq', 'low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
        'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50',
        'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
        'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'geol_1st_class', 'glim_1st_class_frac',
        'geol_2nd_class', 'glim_2nd_class_frac', 'carbonate_rocks_frac', 'geol_porostiy', 'geol_permeability'
    ]
    snow_frac_idx = attr_lst.index('frac_snow')

    exp_name = 'CAMELSMODELSIX'
    subset_tag = 'AllBasins' if args.use_all_basins else 'Subset' + str(len(basin_ids))
    exp_disp = 'DynamicSimHydModelSix/' + subset_tag + '/' + args.forcing + '/' + str(args.seed)
    exp_info = (
        'T_' + str(t_train[0]) + '_' + str(t_train[1]) +
        '_BS_' + str(args.batch_size) + '_HS_' + str(args.hidden_size) + '_RHO_' + str(args.rho) +
        '_Buff_' + str(bufftime) + '_Mul_' + str(args.nmul) +
        '_Route_' + str(int(args.routing)) + '_CmpW_' + str(int(args.compwts)) +
        '_LGDyn_' + str(int(args.lgdyn)) + '_DSQ_' + str(int(args.dynamic_sq or args.dynamic_all)) +
        '_DETGAM_' + str(int(args.dynamic_etgam or args.dynamic_all)) +
        '_DPART_' + str(int(args.dynamic_partition or args.dynamic_all)) +
        '_DCFMAX_' + str(int(args.dynamic_cfmax_snow or args.dynamic_all)) +
        '_DROUTE_' + str(int(args.dynamic_routing_scale or args.dynamic_all)) +
        '_CRoute_' + str(int(args.component_routing)) +
        '_DryCh_' + str(int(args.dry_channel_loss)) +
        '_ZGate_' + str(int(args.zero_flow_gate)) +
        '_MaxIter' + str(args.max_iter_ep)
    )
    if args.exp_info_suffix:
        exp_info = exp_info + args.exp_info_suffix
    model_dir = os.path.join(root_out, exp_name, exp_disp, exp_info)
    model = loadModel(model_dir, epoch=args.epoch)

    df_train = camels.DataframeCamels(tRange=t_train, subset=basin_ids, forType=args.forcing)
    forc_train = df_train.getDataTs(varLst=var_f, doNorm=False, rmNan=False)
    df_inv = camels.DataframeCamels(tRange=t_inv, subset=basin_ids, forType=args.forcing)
    forc_inv = df_inv.getDataTs(varLst=var_f_inv, doNorm=False, rmNan=False)
    attrs_un = df_inv.getDataConst(varLst=attr_lst, doNorm=False, rmNan=False)

    df_test = camels.DataframeCamels(tRange=t_test, subset=basin_ids, forType=args.forcing)
    forc_test = df_test.getDataTs(varLst=var_f, doNorm=False, rmNan=False)
    obs_test = df_test.getDataObs(doNorm=False, rmNan=False, basinnorm=False)

    areas = gageinfo['area'][basin_ind]
    temp_area = np.tile(areas[:, None, None], (1, obs_test.shape[1], 1))
    obs_test = (obs_test * 0.0283168 * 3600 * 24) / (temp_area * (10 ** 6)) * 10 ** 3
    obs_test = obs_test[:, :, 0].astype(np.float32)

    var_lst_nl = ['PEVAP']
    usgs_id_lst = gageid
    t_pet_range = [19800101, 20150101] if args.forcing != 'maurer' else [19800101, 20090101]
    t_pet_lst = utils.time.tRange2Array(t_pet_range)
    pet_dir = root_database + '/pet_harg/' + args.forcing + '/'
    ntime = len(t_pet_lst)
    pet_full = np.empty([len(usgs_id_lst), ntime, len(var_lst_nl)])
    for k in range(len(usgs_id_lst)):
        pet_full[k, :, :] = camels.readcsvGage(pet_dir, usgs_id_lst[k], var_lst_nl, ntime)

    t_train_lst = utils.time.tRange2Array(t_train)
    t_inv_lst = utils.time.tRange2Array(t_inv)
    t_test_lst = utils.time.tRange2Array(t_test)
    _, _, ind2 = np.intersect1d(t_train_lst, t_pet_lst, return_indices=True)
    pet_train = pet_full[:, ind2, :][basin_ind, :, :]
    _, _, ind2inv = np.intersect1d(t_inv_lst, t_pet_lst, return_indices=True)
    pet_inv = pet_full[:, ind2inv, :][basin_ind, :, :]
    _, _, ind2test = np.intersect1d(t_test_lst, t_pet_lst, return_indices=True)
    pet_test = pet_full[:, ind2test, :][basin_ind, :, :]

    with open(os.path.join(model_dir, 'statDict.json'), 'r') as fp:
        stat_dict = json.load(fp)
    series_inv = np.concatenate([forc_inv, pet_inv], axis=2)
    series_var_lst = var_f_inv + ['pet']
    attr_norm = camels.transNormbyDic(attrs_un, attr_lst, stat_dict, toNorm=True)
    attr_norm[np.isnan(attr_norm)] = 0.0
    series_test = np.concatenate([forc_test, pet_test], axis=2)
    series_test_norm = camels.transNormbyDic(series_test, series_var_lst, stat_dict, toNorm=True)
    series_test_norm[np.isnan(series_test_norm)] = 0.0
    snow_frac_raw = attrs_un[:, snow_frac_idx:snow_frac_idx + 1].astype(np.float32)
    snow_frac_ts = np.repeat(snow_frac_raw[:, None, :], series_test_norm.shape[1], axis=1)
    z_test = np.concatenate([series_test_norm, snow_frac_ts], axis=2)
    c_temp = np.repeat(np.reshape(attr_norm, [attr_norm.shape[0], 1, attr_norm.shape[-1]]), z_test.shape[1], axis=1)
    z_test = np.concatenate([z_test, c_temp], axis=2)

    season_train = np.tile(seasonal_features(t_train)[None, :, :], (len(basin_ids), 1, 1))
    season_test = np.tile(seasonal_features(t_test)[None, :, :], (len(basin_ids), 1, 1))
    x_train = np.concatenate([forc_train, pet_train, season_train], axis=2).astype(np.float32)
    x_train[np.isnan(x_train)] = 0.0
    x_test = np.concatenate([forc_test, pet_test, season_test], axis=2).astype(np.float32)
    x_test[np.isnan(x_test)] = 0.0
    x_test = np.concatenate([x_train[:, -x_train.shape[1]:, :], x_test], axis=1)

    out_dir = os.path.join(root_out, exp_name, exp_disp, 'analysis_ep' + str(args.epoch))
    os.makedirs(out_dir, exist_ok=True)
    map_param_dir = os.path.join(out_dir, 'maps', 'static_parameters')
    map_flux_dir = os.path.join(out_dir, 'maps', 'fluxes')
    os.makedirs(map_param_dir, exist_ok=True)
    os.makedirs(map_flux_dir, exist_ok=True)

    all_diag = None
    all_q = np.zeros((len(basin_ids), obs_test.shape[1]), dtype=np.float32)
    model.inittime = x_train.shape[1]
    model.train(mode=False)
    if torch.cuda.is_available():
        model = model.cuda()

    for i0 in range(0, len(basin_ids), args.chunk_size):
        i1 = min(i0 + args.chunk_size, len(basin_ids))
        x_part = torch.from_numpy(np.swapaxes(x_test[i0:i1], 1, 0)).float()
        z_part = torch.from_numpy(np.swapaxes(z_test[i0:i1], 1, 0)).float()
        if torch.cuda.is_available():
            x_part = x_part.cuda(args.gpu_id)
            z_part = z_part.cuda(args.gpu_id)
        with torch.no_grad():
            q_part, diag_part = model(x_part, z_part, return_diagnostics=True)
        all_q[i0:i1] = q_part.detach().cpu().numpy()[:, :, 0].T
        if all_diag is None:
            all_diag = {k: np.zeros((len(basin_ids), q_part.shape[0]), dtype=np.float32) for k in diag_part.keys()}
        for k, v in diag_part.items():
            all_diag[k][i0:i1] = v.detach().cpu().numpy()[:, :, 0].T

    p_total = all_diag['rainfall'] + all_diag['snowfall']
    et_ratio = safe_sum_ratio(all_diag['actual_ET'], p_total)
    runoff_ratio = safe_sum_ratio(all_diag['total_discharge'], p_total)
    bfi = safe_sum_ratio(all_diag['baseflow'], all_diag['total_discharge'])
    gw_loss_fraction = safe_sum_ratio(all_diag['groundwater_loss'], all_diag['baseflow'] + all_diag['groundwater_loss'])
    storage_start = all_diag['snowpack'][:, 0] + all_diag['soil_moisture'][:, 0] + all_diag['groundwater'][:, 0]
    storage_end = all_diag['snowpack'][:, -1] + all_diag['soil_moisture'][:, -1] + all_diag['groundwater'][:, -1]
    delta_storage = storage_end - storage_start
    wb_closure = (
        np.nansum(p_total, axis=1)
        - np.nansum(all_diag['actual_ET'], axis=1)
        - np.nansum(all_diag['total_discharge'], axis=1)
        - np.nansum(all_diag['groundwater_loss'], axis=1)
        - delta_storage
    )

    obs_bfi = np.full(len(basin_ids), np.nan, dtype=np.float32)
    for i in range(len(basin_ids)):
        bf_obs = lh_baseflow(obs_test[i])
        qsum = np.nansum(obs_test[i])
        if np.isfinite(qsum) and qsum > 1e-8:
            obs_bfi[i] = np.nansum(bf_obs) / qsum

    meta = pd.DataFrame({
        'gage_id': np.array(basin_ids, dtype=np.int32),
        'gage_name': [gageinfo['name'][i] for i in basin_ind],
        'lat': np.asarray([gageinfo['lat'][i] for i in basin_ind], dtype=float),
        'lon': np.asarray([gageinfo['lon'][i] for i in basin_ind], dtype=float),
    })
    meta['NSE'] = [calc_nse(obs_test[i], all_q[i]) for i in range(len(basin_ids))]
    meta['KGE'] = [calc_kge(obs_test[i], all_q[i]) for i in range(len(basin_ids))]
    meta['logNSE'] = [calc_log_nse(obs_test[i], all_q[i]) for i in range(len(basin_ids))]
    meta['lowflow_NSE'] = [lowflow_nse(obs_test[i], all_q[i]) for i in range(len(basin_ids))]
    meta['highflow_NSE'] = [highflow_nse(obs_test[i], all_q[i]) for i in range(len(basin_ids))]
    meta['fdc_error'] = [fdc_error(obs_test[i], all_q[i]) for i in range(len(basin_ids))]
    meta['ET_ratio'] = et_ratio
    meta['runoff_ratio'] = runoff_ratio
    meta['BFI'] = bfi
    meta['BFI_obs'] = obs_bfi
    meta['BFI_error'] = np.abs(meta['BFI'] - meta['BFI_obs'])
    meta['groundwater_loss_fraction'] = gw_loss_fraction
    meta['water_balance_closure'] = wb_closure

    # Flux sums for separate groundwater/surface bookkeeping
    meta['sum_surface_runoff'] = np.nansum(all_diag['surface_runoff'], axis=1)
    meta['sum_interflow'] = np.nansum(all_diag['interflow'], axis=1)
    meta['sum_recharge_to_groundwater'] = np.nansum(all_diag['recharge_to_groundwater'], axis=1)
    meta['sum_baseflow'] = np.nansum(all_diag['baseflow'], axis=1)
    meta['sum_groundwater_loss'] = np.nansum(all_diag['groundwater_loss'], axis=1)
    meta['sum_actual_ET'] = np.nansum(all_diag['actual_ET'], axis=1)
    meta['sum_total_discharge'] = np.nansum(all_diag['total_discharge'], axis=1)
    meta['mean_groundwater_storage'] = np.nanmean(all_diag['groundwater'], axis=1)
    meta['mean_soil_moisture'] = np.nanmean(all_diag['soil_moisture'], axis=1)
    meta['mean_snowpack'] = np.nanmean(all_diag['snowpack'], axis=1)
    meta['channel_loss_mean'] = np.nanmean(all_diag['channel_loss'], axis=1) if 'channel_loss' in all_diag else np.nan
    meta['zero_flow_probability_mean'] = np.nanmean(all_diag['zero_flow_probability'], axis=1) if 'zero_flow_probability' in all_diag else np.nan

    for name in ['COEF_t', 'SQ_t', 'ETGAM_t', 'SUB_t', 'CRAK_t', 'K_t', 'LG_t', 'SG_CRIT', 'CFMAX_t']:
        if name in all_diag:
            meta[name + '_mean'] = np.nanmean(all_diag[name], axis=1)

    # Recover learned static parameters from model head for parameter maps
    with torch.no_grad():
        basin_attr_t = torch.from_numpy(attr_norm).float()
        if torch.cuda.is_available():
            basin_attr_t = basin_attr_t.cuda(args.gpu_id)
        static_feat = model.staticFeat(basin_attr_t)
        static_params0 = model.staticOut(static_feat)
        ngage = len(basin_ids)
        static0 = static_params0[:, :model.nstaticpm].view(ngage, model.nfea, model.nmul) + model.compStaticBias
        theta = torch.sigmoid(static0)
        wts0 = static_params0[:, model.nstaticpm + model.nroutpm:model.nstaticpm + model.nroutpm + model.nwtspm]
        wts = torch.softmax(wts0 + model.compWeightBias, dim=-1)
        rout0 = torch.sigmoid(static_params0[:, model.nstaticpm:model.nstaticpm + model.nroutpm])

        theta_flat = theta.permute(0, 2, 1).contiguous().view(ngage * model.nmul, model.nfea)
        theta_phys = model.simhyd.denorm_params(theta_flat).view(ngage, model.nmul, model.nfea)
        theta_w = torch.sum(theta_phys * wts.unsqueeze(-1), dim=1).detach().cpu().numpy()
        wts_np = wts.detach().cpu().numpy()

        if getattr(model, 'component_routing', False):
            rout0 = rout0.view(ngage, model.nmul, 2)
            route_a = 0.0 + rout0[:, :, 0] * 2.9
            route_b = 0.0 + rout0[:, :, 1] * 6.5
            route_a_np = torch.sum(route_a * wts, dim=1).detach().cpu().numpy()
            route_b_np = torch.sum(route_b * wts, dim=1).detach().cpu().numpy()
        else:
            route_a_np = (0.0 + rout0[:, 0] * 2.9).detach().cpu().numpy()
            route_b_np = (0.0 + rout0[:, 1] * 6.5).detach().cpu().numpy()

    param_names = ['INSC', 'COEF', 'SQ', 'SMSC', 'SUB', 'CRAK', 'K', 'LG', 'TT', 'CFMAX', 'CFR', 'CWH', 'SG_CRIT']
    for i, nm in enumerate(param_names):
        meta[nm + '_wavg'] = theta_w[:, i]
    meta['route_a_static'] = route_a_np
    meta['route_b_static'] = route_b_np
    for i in range(wts_np.shape[1]):
        meta['weight_c' + str(i + 1)] = wts_np[:, i]

    np.savez_compressed(
        os.path.join(out_dir, 'model_six_diagnostics_ep' + str(args.epoch) + '.npz'),
        basin_ids=np.array(basin_ids, dtype=np.int32),
        obs_q=obs_test.astype(np.float32),
        pred_q=all_q.astype(np.float32),
        **{k: v.astype(np.float32) for k, v in all_diag.items()})
    meta.to_csv(os.path.join(out_dir, 'per_basin_model_six_metrics.csv'), index=False)

    summary = pd.DataFrame([{
        'median_NSE': float(np.nanmedian(meta['NSE'])),
        'median_KGE': float(np.nanmedian(meta['KGE'])),
        'median_logNSE': float(np.nanmedian(meta['logNSE'])),
        'median_lowflow_NSE': float(np.nanmedian(meta['lowflow_NSE'])),
        'median_highflow_NSE': float(np.nanmedian(meta['highflow_NSE'])),
        'median_fdc_error': float(np.nanmedian(meta['fdc_error'])),
        'median_BFI_error': float(np.nanmedian(meta['BFI_error'])),
        'median_ET_ratio': float(np.nanmedian(meta['ET_ratio'])),
        'median_runoff_ratio': float(np.nanmedian(meta['runoff_ratio'])),
        'median_water_balance_closure': float(np.nanmedian(meta['water_balance_closure'])),
    }])
    summary.to_csv(os.path.join(out_dir, 'summary_metrics.csv'), index=False)

    # Create requested separated map sets
    for nm in ['INSC_wavg', 'COEF_wavg', 'SQ_wavg', 'SMSC_wavg', 'SUB_wavg', 'CRAK_wavg',
               'K_wavg', 'LG_wavg', 'TT_wavg', 'CFMAX_wavg', 'CFR_wavg', 'CWH_wavg',
               'SG_CRIT_wavg', 'route_a_static', 'route_b_static']:
        plot_map(meta, nm, 'Model Six static ' + nm, os.path.join(map_param_dir, nm + '.png'), cmap='viridis')

    for nm in ['mean_groundwater_storage', 'sum_groundwater_loss', 'sum_baseflow',
               'sum_surface_runoff', 'sum_interflow', 'sum_recharge_to_groundwater',
               'sum_actual_ET', 'NSE', 'ET_ratio', 'runoff_ratio',
               'channel_loss_mean', 'zero_flow_probability_mean']:
        cmap = 'cividis' if 'NSE' not in nm else 'YlGnBu'
        plot_map(meta, nm, 'Model Six flux/metric ' + nm, os.path.join(map_flux_dir, nm + '.png'), cmap=cmap)

    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
