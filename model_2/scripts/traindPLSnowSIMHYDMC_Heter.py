import sys
sys.path.append('../../')
from hydroDL import master, utils
from hydroDL.data import camels
from hydroDL.master import default
from hydroDL.model import rnn, crit, train

import os
import numpy as np
import torch
from collections import OrderedDict
import random
import json
import time


randomseed = 111111
random.seed(randomseed)
torch.manual_seed(randomseed)
np.random.seed(randomseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(randomseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

traingpuid = int(os.environ.get('DPLSNOWSIMHYDMC_HETER_GPU_ID', '0'))
if torch.cuda.is_available():
    torch.cuda.set_device(traingpuid)

EPOCH = int(os.environ.get('DPLSNOWSIMHYDMC_HETER_EPOCHS', '5'))
BATCH_SIZE = int(os.environ.get('DPLSNOWSIMHYDMC_HETER_BATCH_SIZE', '64'))
RHO = int(os.environ.get('DPLSNOWSIMHYDMC_HETER_RHO', '365'))
HIDDENSIZE = int(os.environ.get('DPLSNOWSIMHYDMC_HETER_HIDDEN_SIZE', '64'))
saveEPOCH = int(os.environ.get('DPLSNOWSIMHYDMC_HETER_SAVE_EPOCH', '1'))
max_iter_ep = int(os.environ.get('DPLSNOWSIMHYDMC_HETER_MAX_ITER_EP', '20'))
NMUL = int(os.environ.get('DPLSNOWSIMHYDMC_HETER_NMUL', '4'))
ROUTING = os.environ.get('DPLSNOWSIMHYDMC_HETER_ROUTING', '1') == '1'
COMPROUT = os.environ.get('DPLSNOWSIMHYDMC_HETER_COMPROUT', '0') == '1'
COMPWTS = os.environ.get('DPLSNOWSIMHYDMC_HETER_COMPWTS', '1') == '1'
LGDYN = os.environ.get('DPLSNOWSIMHYDMC_HETER_LGDYN', '1') == '1'
LGDYNWEIGHT = float(os.environ.get('DPLSNOWSIMHYDMC_HETER_LGDYN_WEIGHT', '0.5'))
Ttrain = [19801001, 19951001]
Tinv = [19801001, 19951001]
BUFFTIME = 365
forType = os.environ.get('DPLSNOWSIMHYDMC_HETER_FORCING', 'daymet')

rootDatabase = os.environ.get(
    'DPLSNOWSIMHYDMC_HETER_ROOT_DB',
    os.path.join(os.path.sep, 'scratch', 'Camels'))
camels.initcamels(rootDatabase)

rootOut = os.environ.get(
    'DPLSNOWSIMHYDMC_HETER_ROOT_OUT',
    os.path.join(os.path.sep, 'data', 'rnnStreamflow'))

gageinfo = camels.gageDict
gageid = gageinfo['id']
gageidLst = gageid.tolist()

subset_file = os.environ.get(
    'DPLSNOWSIMHYDMC_HETER_SUBSET_FILE',
    os.path.join('..', 'dPLHBV', 'Sub531ID.txt'))
subset_limit = int(os.environ.get('DPLSNOWSIMHYDMC_HETER_SUBSET_LIMIT', '64'))
use_all_basins = os.environ.get('DPLSNOWSIMHYDMC_HETER_USE_ALL', '0') == '1'
if use_all_basins:
    TrainLS = gageinfo['id'].tolist()
else:
    with open(subset_file, 'r') as fp:
        subset_ids = json.load(fp)
    TrainLS = [int(x) for x in subset_ids[:subset_limit]]
TrainInd = [gageidLst.index(j) for j in TrainLS]

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

optData = default.optDataCamels
optData = default.update(optData, tRange=Ttrain, varT=varFInv + ['pet'], varC=attrnewLst, subset=TrainLS, forType=forType)

dfTrain = camels.DataframeCamels(tRange=Ttrain, subset=TrainLS, forType=forType)
forcUN = dfTrain.getDataTs(varLst=varF, doNorm=False, rmNan=False)
obsUN = dfTrain.getDataObs(doNorm=False, rmNan=False, basinnorm=False)

dfInv = camels.DataframeCamels(tRange=Tinv, subset=TrainLS, forType=forType)
forcInvUN = dfInv.getDataTs(varLst=varFInv, doNorm=False, rmNan=False)
attrsUN = dfInv.getDataConst(varLst=attrnewLst, doNorm=False, rmNan=False)

areas = gageinfo['area'][TrainInd]
temparea = np.tile(areas[:, None, None], (1, obsUN.shape[1], 1))
obsUN = (obsUN * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10 ** 3

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

series_inv = np.concatenate([forcInvUN, PETInvUN], axis=2)
seriesvarLst = varFInv + ['pet']
statDict = camels.getStatDic(attrLst=attrnewLst, attrdata=attrsUN, seriesLst=seriesvarLst, seriesdata=series_inv)
attr_norm = camels.transNormbyDic(attrsUN, attrnewLst, statDict, toNorm=True)
attr_norm[np.isnan(attr_norm)] = 0.0
series_norm = camels.transNormbyDic(series_inv, seriesvarLst, statDict, toNorm=True)
series_norm[np.isnan(series_norm)] = 0.0

zTrain = series_norm
xTrain = np.concatenate([forcUN, PETUN], axis=2)
xTrain[np.isnan(xTrain)] = 0.0
yTrainIn = obsUN
forcTuple = (xTrain, zTrain)
attrs = attr_norm

alpha = 0.25
optLoss = default.update(default.optLossComb, name='hydroDL.model.crit.RmseLossComb', weight=alpha)
lossFun = crit.RmseLossComb(alpha=alpha)

optTrain = default.update(default.optTrainCamels, miniBatch=[BATCH_SIZE, RHO], nEpoch=EPOCH, saveEpoch=saveEPOCH)
exp_name = 'CAMELSSNOWSIMHYDMC_HETER'
subset_tag = 'AllBasins' if use_all_basins else 'Subset' + str(len(TrainLS))
exp_disp = 'dPLSnowSIMHYDMC_Heter/' + subset_tag + '/' + forType + '/' + str(randomseed)
exp_info = 'T_' + str(Ttrain[0]) + '_' + str(Ttrain[1]) + '_BS_' + str(BATCH_SIZE) + '_HS_' + str(HIDDENSIZE) + '_RHO_' + str(RHO) + '_Buff_' + str(BUFFTIME) + '_Mul_' + str(NMUL) + '_Route_' + str(int(ROUTING)) + '_CmpW_' + str(int(COMPWTS)) + '_LGDyn_' + str(int(LGDYN))
exp_suffix = os.environ.get('DPLSNOWSIMHYDMC_HETER_EXP_INFO_SUFFIX', '')
if exp_suffix:
    exp_info = exp_info + exp_suffix
out = os.path.join(rootOut, exp_name, exp_disp, exp_info)

Ninv = zTrain.shape[-1] + attrs.shape[-1]
model = rnn.MultiInv_SnowSIMHYDMulTDHeterModel(
    ninv=Ninv,
    nfea=12,
    nmul=NMUL,
    nattr=attrs.shape[-1],
    hiddeninv=HIDDENSIZE,
    inittime=BUFFTIME,
    routOpt=ROUTING,
    comprout=COMPROUT,
    compwts=COMPWTS,
    lgdyn=LGDYN,
    lgdynweight=LGDYNWEIGHT)
optModel = OrderedDict(
    name='dPLSnowSIMHYDMC_Heter',
    nx=Ninv,
    ny=1,
    hiddenSize=HIDDENSIZE,
    Tinv=Tinv,
    Trainbuff=BUFFTIME,
    subsetSize=len(TrainLS),
    forType=forType,
    nmul=NMUL,
)

masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
master.writeMasterFile(masterDict)
with open(os.path.join(out, 'statDict.json'), 'w') as fp:
    json.dump(statDict, fp, indent=4)

x, z = forcTuple
y = yTrainIn
c = attrs
if torch.cuda.is_available():
    lossFun = lossFun.cuda()
    model = model.cuda()
optim = torch.optim.Adadelta(model.parameters())
model.zero_grad()
runFile = os.path.join(out, 'run.csv')
rf = open(runFile, 'w+')
for iEpoch in range(1, EPOCH + 1):
    lossEp = 0
    t0 = time.time()
    for iIter in range(0, max_iter_ep):
        iGrid, iT = train.randomIndex(len(TrainLS), x.shape[1], [BATCH_SIZE, RHO], bufftime=BUFFTIME)
        xTrainBatch = train.selectSubset(x, iGrid, iT, RHO, bufftime=BUFFTIME)
        yTrainBatch = train.selectSubset(y, iGrid, iT, RHO)
        zTrainBatch = train.selectSubset(z, iGrid, iT, RHO, c=c)
        yP = model(xTrainBatch, zTrainBatch)
        loss = lossFun(yP, yTrainBatch)
        loss.backward()
        optim.step()
        model.zero_grad()
        lossEp = lossEp + loss.item()
        if iIter % 100 == 0:
            print('Iter {} of {}: Loss {:.3f}'.format(iIter, max_iter_ep, loss.item()))
    lossEp = lossEp / max_iter_ep
    logStr = 'Epoch {} Loss {:.3f} time {:.2f}'.format(iEpoch, lossEp, time.time() - t0)
    print(logStr)
    rf.write(logStr + '\n')
    rf.flush()
    if iEpoch % saveEPOCH == 0:
        torch.save(model, os.path.join(out, 'model_Ep' + str(iEpoch) + '.pt'))
rf.close()
