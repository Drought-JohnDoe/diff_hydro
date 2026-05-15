import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append('../../')
from hydroDL import master, utils
from hydroDL.data import camels
from hydroDL.master import loadModel
from hydroDL.model import train
from hydroDL.post import stat


def add_bool_arg(parser, name, default=False, help_text=''):
    flag = '--' + name.replace('_', '-')
    parser.add_argument(flag, dest=name, action='store_true', help=help_text)
    parser.add_argument('--no-' + name.replace('_', '-'), dest=name, action='store_false')
    parser.set_defaults(**{name: default})


def parse_args():
    parser = argparse.ArgumentParser(description='Test Model Six')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--rho', type=int, default=365)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--nmul', type=int, default=4)
    parser.add_argument('--lgdyn-weight', type=float, default=0.6)
    parser.add_argument('--forcing', type=str, default='daymet')
    parser.add_argument('--subset-limit', type=int, default=64)
    parser.add_argument('--test-batch', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--seed', type=int, default=111111)
    parser.add_argument('--exp-info-suffix', type=str, default='')
    parser.add_argument('--test-out-suffix', type=str, default='_ModelSix')
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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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
        str(project_root / 'code' / 'dPLHBVrelease' / 'hydroDL-dev' / 'example' / 'model_six' / 'Sub531ID.txt'))

    t_train = [19801001, 19951001]
    t_inv = [19801001, 19951001]
    t_test = [19951001, 20101001]
    t_test_lst = utils.time.tRange2Array(t_test)
    camels.initcamels(root_database)

    gageinfo = camels.gageDict
    gageid = gageinfo['id']
    gageid_lst = gageid.tolist()
    if args.use_all_basins:
        subset_ids = gageid.tolist()
    else:
        with open(subset_file, 'r') as fp:
            subset_ids = json.load(fp)
        subset_ids = [int(x) for x in subset_ids[:args.subset_limit]]
    train_ls = subset_ids
    train_ind = [gageid_lst.index(j) for j in train_ls]
    test_ls = subset_ids
    test_ind = [gageid_lst.index(j) for j in test_ls]

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
    subset_tag = 'AllBasins' if args.use_all_basins else 'Subset' + str(len(train_ls))
    exp_disp = 'DynamicSimHydModelSix/' + subset_tag + '/' + args.forcing + '/' + str(args.seed)
    exp_info = (
        'T_' + str(t_train[0]) + '_' + str(t_train[1]) +
        '_BS_' + str(args.batch_size) + '_HS_' + str(args.hidden_size) + '_RHO_' + str(args.rho) +
        '_Buff_365_Mul_' + str(args.nmul) +
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
    test_out = os.path.join(root_out, exp_name, exp_disp, exp_info)
    test_model = loadModel(test_out, epoch=args.epoch)

    df_train = camels.DataframeCamels(tRange=t_train, subset=train_ls, forType=args.forcing)
    forc_un = df_train.getDataTs(varLst=var_f, doNorm=False, rmNan=False)
    df_inv = camels.DataframeCamels(tRange=t_inv, subset=train_ls, forType=args.forcing)
    forc_inv_un = df_inv.getDataTs(varLst=var_f_inv, doNorm=False, rmNan=False)
    attrs_un = df_inv.getDataConst(varLst=attr_lst, doNorm=False, rmNan=False)

    df_test = camels.DataframeCamels(tRange=t_test, subset=test_ls, forType=args.forcing)
    forc_test_un = df_test.getDataTs(varLst=var_f, doNorm=False, rmNan=False)
    obs_test_un = df_test.getDataObs(doNorm=False, rmNan=False, basinnorm=False)
    attrs_test_un = df_test.getDataConst(varLst=attr_lst, doNorm=False, rmNan=False)

    areas = gageinfo['area'][test_ind]
    temp_area = np.tile(areas[:, None, None], (1, obs_test_un.shape[1], 1))
    obs_test_un = (obs_test_un * 0.0283168 * 3600 * 24) / (temp_area * (10 ** 6)) * 10 ** 3

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
    _, _, ind2test = np.intersect1d(t_test_lst, t_pet_lst, return_indices=True)
    pet_test_un = pet_full[:, ind2test, :][test_ind, :, :]

    series_inv = np.concatenate([forc_inv_un, pet_inv_un], axis=2)
    series_var_lst = var_f_inv + ['pet']
    with open(os.path.join(test_out, 'statDict.json'), 'r') as fp:
        stat_dict = json.load(fp)

    attr_norm = camels.transNormbyDic(attrs_un, attr_lst, stat_dict, toNorm=True)
    attr_norm[np.isnan(attr_norm)] = 0.0
    attr_test_norm = camels.transNormbyDic(attrs_test_un, attr_lst, stat_dict, toNorm=True)
    attr_test_norm[np.isnan(attr_test_norm)] = 0.0
    series_test = np.concatenate([forc_test_un, pet_test_un], axis=2)
    series_test_norm = camels.transNormbyDic(series_test, series_var_lst, stat_dict, toNorm=True)
    series_test_norm[np.isnan(series_test_norm)] = 0.0

    season_train = np.tile(seasonal_features(t_train)[None, :, :], (len(train_ls), 1, 1))
    season_test = np.tile(seasonal_features(t_test)[None, :, :], (len(test_ls), 1, 1))

    x_train = np.concatenate([forc_un, pet_un, season_train], axis=2).astype(np.float32)
    x_train[np.isnan(x_train)] = 0.0
    x_test = np.concatenate([forc_test_un, pet_test_un, season_test], axis=2).astype(np.float32)
    x_test[np.isnan(x_test)] = 0.0
    x_test_buff = x_train[:, -x_train.shape[1]:, :]
    x_test = np.concatenate([x_test_buff, x_test], axis=1)

    snow_frac_raw = attrs_test_un[:, snow_frac_idx:snow_frac_idx + 1].astype(np.float32)
    snow_frac_ts = np.repeat(snow_frac_raw[:, None, :], series_test_norm.shape[1], axis=1)
    z_test = np.concatenate([series_test_norm, snow_frac_ts], axis=2)
    c_temp = np.repeat(np.reshape(attr_test_norm, [attr_test_norm.shape[0], 1, attr_test_norm.shape[-1]]), z_test.shape[1], axis=1)
    z_test = np.concatenate([z_test, c_temp], axis=2)

    test_model.inittime = x_train.shape[1]
    file_path_lst = master.master.namePred(test_out, t_test, 'All_Buff' + str(x_train.shape[1]), epoch=args.epoch, targLst=['Q'])
    test_tuple = (x_test, z_test)
    train.testModel(test_model, test_tuple, c=None, batchSize=args.test_batch, filePathLst=file_path_lst)

    data_pred = np.ndarray([obs_test_un.shape[0], obs_test_un.shape[1], len(file_path_lst)])
    for k in range(len(file_path_lst)):
        data_pred[:, :, k] = pd.read_csv(file_path_lst[k], dtype=float, header=None).values

    eva_dict = [stat.statError(data_pred[:, :, 0], obs_test_un[:, :, 0])]
    outname = 'Train' + str(t_train[0]) + '_' + str(t_train[1]) + 'Test' + str(t_test[0]) + '_' + str(t_test[1])
    if args.test_out_suffix:
        outname = outname + args.test_out_suffix
    outpath = os.path.join(root_out, exp_name, exp_disp, outname)
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    np.save(os.path.join(outpath, 'Eva' + str(args.epoch) + '.npy'), eva_dict)
    np.save(os.path.join(outpath, 'obs.npy'), obs_test_un)
    np.save(os.path.join(outpath, 'pred' + str(args.epoch) + '.npy'), data_pred)

    print('Model Six testing finished! Evaluation results saved in\n', outpath)
    print('Median NSE:', np.nanmedian(eva_dict[0]['NSE']))


if __name__ == '__main__':
    main()
