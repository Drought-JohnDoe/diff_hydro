import argparse
import json
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append('../../')
from hydroDL import master, utils
from hydroDL.data import camels
from hydroDL.master import default
from hydroDL.model import crit, rnn, train


def add_bool_arg(parser, name, default=False, help_text=''):
    flag = '--' + name.replace('_', '-')
    parser.add_argument(flag, dest=name, action='store_true', help=help_text)
    parser.add_argument('--no-' + name.replace('_', '-'), dest=name, action='store_false')
    parser.set_defaults(**{name: default})


def parse_args():
    parser = argparse.ArgumentParser(description='Train Model Six (structural DynamicSimHyd ablation)')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--rho', type=int, default=365)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--save-epoch', type=int, default=1)
    parser.add_argument('--max-iter-ep', type=int, default=200)
    parser.add_argument('--nmul', type=int, default=4)
    parser.add_argument('--lgdyn-weight', type=float, default=0.6)
    parser.add_argument('--forcing', type=str, default='daymet')
    parser.add_argument('--subset-limit', type=int, default=64)
    parser.add_argument('--exp-info-suffix', type=str, default='')
    parser.add_argument('--seed', type=int, default=111111)
    parser.add_argument('--reg-amp-w', type=float, default=1e-3)
    parser.add_argument('--reg-smooth-w', type=float, default=1e-3)
    parser.add_argument('--reg-part-w', type=float, default=1e-3)
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


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(args.gpu_id)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[5]
    root_database = os.environ.get('DYNAMIC_SIMHYD_ROOT_DB', str(project_root / 'Camels'))
    root_out = os.environ.get('DYNAMIC_SIMHYD_ROOT_OUT', str(project_root / 'outputs' / 'rnnStreamflow'))
    subset_file = os.environ.get(
        'DYNAMIC_SIMHYD_SUBSET_FILE',
        str(project_root / 'code' / 'dPLHBVrelease' / 'hydroDL-dev' / 'example' / 'dPLHBV' / 'Sub531ID.txt'))

    t_train = [19801001, 19951001]
    t_inv = [19801001, 19951001]
    bufftime = 365
    camels.initcamels(root_database)

    gageinfo = camels.gageDict
    gageid = gageinfo['id']
    gageid_lst = gageid.tolist()
    if args.use_all_basins:
        train_ls = gageid.tolist()
    else:
        with open(subset_file, 'r') as fp:
            subset_ids = json.load(fp)
        train_ls = [int(x) for x in subset_ids[:args.subset_limit]]
    train_ind = [gageid_lst.index(j) for j in train_ls]

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

    opt_data = default.optDataCamels
    opt_data = default.update(opt_data, tRange=t_train, varT=var_f_inv + ['pet'], varC=attr_lst, subset=train_ls, forType=args.forcing)

    df_train = camels.DataframeCamels(tRange=t_train, subset=train_ls, forType=args.forcing)
    forc_un = df_train.getDataTs(varLst=var_f, doNorm=False, rmNan=False)
    obs_un = df_train.getDataObs(doNorm=False, rmNan=False, basinnorm=False)

    df_inv = camels.DataframeCamels(tRange=t_inv, subset=train_ls, forType=args.forcing)
    forc_inv_un = df_inv.getDataTs(varLst=var_f_inv, doNorm=False, rmNan=False)
    attrs_un = df_inv.getDataConst(varLst=attr_lst, doNorm=False, rmNan=False)

    areas = gageinfo['area'][train_ind]
    temp_area = np.tile(areas[:, None, None], (1, obs_un.shape[1], 1))
    obs_un = (obs_un * 0.0283168 * 3600 * 24) / (temp_area * (10 ** 6)) * 10 ** 3

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
    _, _, ind2 = np.intersect1d(t_train_lst, t_pet_lst, return_indices=True)
    pet_un = pet_full[:, ind2, :][train_ind, :, :]
    _, _, ind2inv = np.intersect1d(t_inv_lst, t_pet_lst, return_indices=True)
    pet_inv_un = pet_full[:, ind2inv, :][train_ind, :, :]

    season_train = np.tile(seasonal_features(t_train)[None, :, :], (len(train_ls), 1, 1))

    series_inv = np.concatenate([forc_inv_un, pet_inv_un], axis=2)
    series_var_lst = var_f_inv + ['pet']
    stat_dict = camels.getStatDic(attrLst=attr_lst, attrdata=attrs_un, seriesLst=series_var_lst, seriesdata=series_inv)
    attr_norm = camels.transNormbyDic(attrs_un, attr_lst, stat_dict, toNorm=True)
    attr_norm[np.isnan(attr_norm)] = 0.0
    series_norm = camels.transNormbyDic(series_inv, series_var_lst, stat_dict, toNorm=True)
    series_norm[np.isnan(series_norm)] = 0.0

    # Keep raw snow fraction in z as an explicit non-normalized channel for snow-basin gating.
    snow_frac_raw = attrs_un[:, snow_frac_idx:snow_frac_idx + 1].astype(np.float32)
    snow_frac_ts = np.repeat(snow_frac_raw[:, None, :], series_norm.shape[1], axis=1)
    z_train = np.concatenate([series_norm, snow_frac_ts], axis=2)
    x_train = np.concatenate([forc_un, pet_un, season_train], axis=2).astype(np.float32)
    x_train[np.isnan(x_train)] = 0.0
    y_train = obs_un
    forc_tuple = (x_train, z_train)
    attrs = attr_norm

    alpha = 0.25
    opt_loss = default.update(default.optLossComb, name='hydroDL.model.crit.RmseLossComb', weight=alpha)
    loss_fun = crit.RmseLossComb(alpha=alpha)

    opt_train = default.update(default.optTrainCamels, miniBatch=[args.batch_size, args.rho], nEpoch=args.epochs, saveEpoch=args.save_epoch)
    exp_name = 'CAMELSMODELSIX'
    subset_tag = 'AllBasins' if args.use_all_basins else 'Subset' + str(len(train_ls))
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
    out = os.path.join(root_out, exp_name, exp_disp, exp_info)

    ninv = z_train.shape[-1] + attrs.shape[-1]
    model = rnn.MultiInv_DynamicSimHydModelSix(
        ninv=ninv,
        nmul=args.nmul,
        nattr=attrs.shape[-1],
        hiddeninv=args.hidden_size,
        inittime=bufftime,
        routOpt=args.routing,
        comprout=args.comprout,
        compwts=args.compwts,
        lgdyn=args.lgdyn,
        lgdynweight=args.lgdyn_weight,
        dynamic_sq=args.dynamic_sq,
        dynamic_etgam=args.dynamic_etgam,
        dynamic_partition=args.dynamic_partition,
        dynamic_cfmax_snow=args.dynamic_cfmax_snow,
        dynamic_routing_scale=args.dynamic_routing_scale,
        dynamic_all=args.dynamic_all,
        reg_amp_w=args.reg_amp_w,
        reg_smooth_w=args.reg_smooth_w,
        reg_part_w=args.reg_part_w,
        component_routing=args.component_routing,
        dry_channel_loss=args.dry_channel_loss,
        zero_flow_gate=args.zero_flow_gate,
        channel_loss_max=args.channel_loss_max,
        zero_gate_hidden=args.zero_gate_hidden)
    opt_model = OrderedDict(
        name='ModelSix',
        nx=ninv,
        ny=1,
        hiddenSize=args.hidden_size,
        Tinv=t_inv,
        Trainbuff=bufftime,
        subsetSize=len(train_ls),
        forType=args.forcing,
        nmul=args.nmul,
    )

    os.makedirs(out, exist_ok=True)
    master_dict = master.wrapMaster(out, opt_data, opt_model, opt_loss, opt_train)
    master.writeMasterFile(master_dict)
    with open(os.path.join(out, 'statDict.json'), 'w') as fp:
        json.dump(stat_dict, fp, indent=4)

    x, z = forc_tuple
    y = y_train
    c = attrs
    if torch.cuda.is_available():
        loss_fun = loss_fun.cuda()
        model = model.cuda()
    optim = torch.optim.Adadelta(model.parameters())
    model.zero_grad()

    run_file = os.path.join(out, 'run.csv')
    rf = open(run_file, 'w+')
    for i_epoch in range(1, args.epochs + 1):
        loss_ep = 0
        t0 = time.time()
        for i_iter in range(0, args.max_iter_ep):
            i_grid, i_t = train.randomIndex(len(train_ls), x.shape[1], [args.batch_size, args.rho], bufftime=bufftime)
            x_train_batch = train.selectSubset(x, i_grid, i_t, args.rho, bufftime=bufftime)
            y_train_batch = train.selectSubset(y, i_grid, i_t, args.rho)
            z_train_batch = train.selectSubset(z, i_grid, i_t, args.rho, c=c)
            y_p = model(x_train_batch, z_train_batch)
            loss = loss_fun(y_p, y_train_batch)
            if hasattr(model, 'get_auxiliary_loss'):
                aux = model.get_auxiliary_loss()
                if aux is not None:
                    loss = loss + aux
            loss.backward()
            optim.step()
            model.zero_grad()
            loss_ep = loss_ep + loss.item()
            if i_iter % 100 == 0:
                print('Iter {} of {}: Loss {:.3f}'.format(i_iter, args.max_iter_ep, loss.item()))
        loss_ep = loss_ep / args.max_iter_ep
        log_str = 'Epoch {} Loss {:.3f} time {:.2f}'.format(i_epoch, loss_ep, time.time() - t0)
        print(log_str)
        rf.write(log_str + '\n')
        rf.flush()
        if i_epoch % args.save_epoch == 0:
            torch.save(model, os.path.join(out, 'model_Ep' + str(i_epoch) + '.pt'))
    rf.close()


if __name__ == '__main__':
    main()
