import sys
sys.path.append('../../')
from hydroDL import master, utils
from hydroDL.data import camels
from hydroDL.master import loadModel
from hydroDL.model import train
from hydroDL.post import stat

import os
import numpy as np
import torch
import pandas as pd
import json


randomseed = 111111
torch.manual_seed(randomseed)
np.random.seed(randomseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(randomseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

testgpuid = int(os.environ.get('DPLSNOWSIMHYDMC_GPU_ID', '0'))
if torch.cuda.is_available():
    torch.cuda.set_device(testgpuid)

BATCH_SIZE = int(os.environ.get('DPLSNOWSIMHYDMC_BATCH_SIZE', '64'))
RHO = int(os.environ.get('DPLSNOWSIMHYDMC_RHO', '365'))
HIDDENSIZE = int(os.environ.get('DPLSNOWSIMHYDMC_HIDDEN_SIZE', '64'))
BUFFTIME = 365
NMUL = int(os.environ.get('DPLSNOWSIMHYDMC_NMUL', '4'))
ROUTING = os.environ.get('DPLSNOWSIMHYDMC_ROUTING', '1') == '1'
COMPWTS = os.environ.get('DPLSNOWSIMHYDMC_COMPWTS', '1') == '1'
LGDYN = os.environ.get('DPLSNOWSIMHYDMC_LGDYN', '1') == '1'
Ttrain = [19801001, 19951001]
Tinv = [19801001, 19951001]
Ttest = [19951001, 20101001]
TtestLst = utils.time.tRange2Array(Ttest)
forType = os.environ.get('DPLSNOWSIMHYDMC_FORCING', 'daymet')
testbatch = int(os.environ.get('DPLSNOWSIMHYDMC_TEST_BATCH', '64'))
testepoch = int(os.environ.get('DPLSNOWSIMHYDMC_TEST_EPOCH', '10'))
test_out_suffix = os.environ.get('DPLSNOWSIMHYDMC_TEST_OUT_SUFFIX', '')

rootDatabase = os.environ.get(
    'DPLSNOWSIMHYDMC_ROOT_DB',
    os.path.join(os.path.sep, 'scratch', 'Camels'))
camels.initcamels(rootDatabase)

rootOut = os.environ.get(
    'DPLSNOWSIMHYDMC_ROOT_OUT',
    os.path.join(os.path.sep, 'data', 'rnnStreamflow'))

gageinfo = camels.gageDict
gageid = gageinfo['id']
gageidLst = gageid.tolist()

subset_file = os.environ.get(
    'DPLSNOWSIMHYDMC_SUBSET_FILE',
    os.path.join('..', 'dPLHBV', 'Sub531ID.txt'))
subset_limit = int(os.environ.get('DPLSNOWSIMHYDMC_SUBSET_LIMIT', '64'))
use_all_basins = os.environ.get('DPLSNOWSIMHYDMC_USE_ALL', '0') == '1'
if use_all_basins:
    subset_ids = gageid.tolist()
else:
    with open(subset_file, 'r') as fp:
        subset_ids = json.load(fp)
    subset_ids = [int(x) for x in subset_ids[:subset_limit]]
TrainLS = subset_ids
TrainInd = [gageidLst.index(j) for j in TrainLS]
TestLS = subset_ids
TestInd = [gageidLst.index(j) for j in TestLS]

if forType == 'daymet':
    varF = ['prcp', 'tmean']
    varFInv = ['prcp', 'tmean']
else:
    varF = ['prcp', 'tmax']
    varFInv = ['prcp', 'tmax']

attrnewLst = [
    'p_mean', 'pet_mean', 'p_seasonality', 'frac_snow', 'aridity', 'high_prec_freq', 'high_prec_dur',
    'low_prec_freq', 'low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
    'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50',
    'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
    'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'geol_1st_class', 'glim_1st_class_frac',
    'geol_2nd_class', 'glim_2nd_class_frac', 'carbonate_rocks_frac', 'geol_porostiy', 'geol_permeability'
]

exp_name = 'CAMELSSNOWSIMHYDMC'
subset_tag = 'AllBasins' if use_all_basins else 'Subset' + str(len(TrainLS))
exp_disp = 'dPLSnowSIMHYDMC/' + subset_tag + '/' + forType + '/' + str(randomseed)
exp_info = 'T_' + str(Ttrain[0]) + '_' + str(Ttrain[1]) + '_BS_' + str(BATCH_SIZE) + '_HS_' + str(HIDDENSIZE) + '_RHO_' + str(RHO) + '_Buff_' + str(BUFFTIME) + '_Mul_' + str(NMUL) + '_Route_' + str(int(ROUTING)) + '_CmpW_' + str(int(COMPWTS)) + '_LGDyn_' + str(int(LGDYN))
exp_suffix = os.environ.get('DPLSNOWSIMHYDMC_EXP_INFO_SUFFIX', '')
if exp_suffix:
    exp_info = exp_info + exp_suffix
testout = os.path.join(rootOut, exp_name, exp_disp, exp_info)
testmodel = loadModel(testout, epoch=testepoch)

dfTrain = camels.DataframeCamels(tRange=Ttrain, subset=TrainLS, forType=forType)
forcUN = dfTrain.getDataTs(varLst=varF, doNorm=False, rmNan=False)
dfInv = camels.DataframeCamels(tRange=Tinv, subset=TrainLS, forType=forType)
forcInvUN = dfInv.getDataTs(varLst=varFInv, doNorm=False, rmNan=False)
attrsUN = dfInv.getDataConst(varLst=attrnewLst, doNorm=False, rmNan=False)

dfTest = camels.DataframeCamels(tRange=Ttest, subset=TestLS, forType=forType)
forcTestUN = dfTest.getDataTs(varLst=varF, doNorm=False, rmNan=False)
obsTestUN = dfTest.getDataObs(doNorm=False, rmNan=False, basinnorm=False)
attrsTestUN = dfTest.getDataConst(varLst=attrnewLst, doNorm=False, rmNan=False)

areas = gageinfo['area'][TestInd]
temparea = np.tile(areas[:, None, None], (1, obsTestUN.shape[1], 1))
obsTestUN = (obsTestUN * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10 ** 3

varLstNL = ['PEVAP']
usgsIdLst = gageid
tPETRange = [19800101, 20150101] if forType != 'maurer' else [19800101, 20090101]
tPETLst = utils.time.tRange2Array(tPETRange)
PETDir = rootDatabase + '/pet_harg/' + forType + '/'
ntime = len(tPETLst)
PETfull = np.empty([len(usgsIdLst), ntime, len(varLstNL)])
for k in range(len(usgsIdLst)):
    PETfull[k, :, :] = camels.readcsvGage(PETDir, usgsIdLst[k], varLstNL, ntime)

TtrainLst = utils.time.tRange2Array(Ttrain)
TinvLst = utils.time.tRange2Array(Tinv)
_, _, ind2 = np.intersect1d(TtrainLst, tPETLst, return_indices=True)
PETUN = PETfull[:, ind2, :][TrainInd, :, :]
_, _, ind2inv = np.intersect1d(TinvLst, tPETLst, return_indices=True)
PETInvUN = PETfull[:, ind2inv, :][TrainInd, :, :]
_, _, ind2test = np.intersect1d(TtestLst, tPETLst, return_indices=True)
PETTestUN = PETfull[:, ind2test, :][TestInd, :, :]

series_inv = np.concatenate([forcInvUN, PETInvUN], axis=2)
seriesvarLst = varFInv + ['pet']
with open(os.path.join(testout, 'statDict.json'), 'r') as fp:
    statDict = json.load(fp)

attr_norm = camels.transNormbyDic(attrsUN, attrnewLst, statDict, toNorm=True)
attr_norm[np.isnan(attr_norm)] = 0.0
series_norm = camels.transNormbyDic(series_inv, seriesvarLst, statDict, toNorm=True)
series_norm[np.isnan(series_norm)] = 0.0

attrtest_norm = camels.transNormbyDic(attrsTestUN, attrnewLst, statDict, toNorm=True)
attrtest_norm[np.isnan(attrtest_norm)] = 0.0
series_test = np.concatenate([forcTestUN, PETTestUN], axis=2)
series_test_norm = camels.transNormbyDic(series_test, seriesvarLst, statDict, toNorm=True)
series_test_norm[np.isnan(series_test_norm)] = 0.0

xTrain = np.concatenate([forcUN, PETUN], axis=2)
xTrain[np.isnan(xTrain)] = 0.0

xTest = np.concatenate([forcTestUN, PETTestUN], axis=2)
xTest[np.isnan(xTest)] = 0.0
xTestBuff = xTrain[:, -xTrain.shape[1]:, :]
xTest = np.concatenate([xTestBuff, xTest], axis=1)
zTest = series_test_norm

cTemp = np.repeat(
    np.reshape(attrtest_norm, [attrtest_norm.shape[0], 1, attrtest_norm.shape[-1]]),
    zTest.shape[1],
    axis=1)
zTest = np.concatenate([zTest, cTemp], axis=2)

testmodel.inittime = xTrain.shape[1]
filePathLst = master.master.namePred(testout, Ttest, 'All_Buff' + str(xTrain.shape[1]), epoch=testepoch, targLst=['Q'])
testTuple = (xTest, zTest)

train.testModel(testmodel, testTuple, c=None, batchSize=testbatch, filePathLst=filePathLst)

dataPred = np.ndarray([obsTestUN.shape[0], obsTestUN.shape[1], len(filePathLst)])
for k in range(len(filePathLst)):
    dataPred[:, :, k] = pd.read_csv(filePathLst[k], dtype=float, header=None).values

evaDict = [stat.statError(dataPred[:, :, 0], obsTestUN[:, :, 0])]
outname = 'Train' + str(Ttrain[0]) + '_' + str(Ttrain[1]) + 'Test' + str(Ttest[0]) + '_' + str(Ttest[1])
if test_out_suffix:
    outname = outname + test_out_suffix
outpath = os.path.join(rootOut, exp_name, exp_disp, outname)
if not os.path.isdir(outpath):
    os.makedirs(outpath)

np.save(os.path.join(outpath, 'Eva' + str(testepoch) + '.npy'), evaDict)
np.save(os.path.join(outpath, 'obs.npy'), obsTestUN)
np.save(os.path.join(outpath, 'pred' + str(testepoch) + '.npy'), dataPred)

print('Snow-SIMHYD MC testing finished! Evaluation results saved in\n', outpath)
print('Median NSE:', np.nanmedian(evaDict[0]['NSE']))
