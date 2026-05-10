import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydroDL.model.rnn import CudnnLstmModel


        return out


def UH_conv(x,UH,viewmode=1):
    # UH is a vector indicating the unit hydrograph
    # the convolved dimension will be the last dimension
    # UH convolution is
    # Q(t)=\integral(x(\tao)*UH(t-\tao))d\tao
    # conv1d does \integral(w(\tao)*x(t+\tao))d\tao
    # hence we flip the UH
    # https://programmer.group/pytorch-learning-conv1d-conv2d-and-conv3d.html
    # view
    # x: [batch, var, time]
    # UH:[batch, var, uhLen]
    # batch needs to be accommodated by channels and we make use of groups
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    # https://pytorch.org/docs/stable/nn.functional.html

    mm= x.shape; nb=mm[0]
    m = UH.shape[-1]
    padd = m-1
    if viewmode==1:
        xx = x.view([1,nb,mm[-1]])
        w  = UH.view([nb,1,m])
        groups = nb

    y = F.conv1d(xx, torch.flip(w,[2]), groups=groups, padding=padd, stride=1, bias=None)
    y=y[:,:,0:-padd]
    return y.view(mm)


def UH_gamma(a,b,lenF=10):
    # UH. a [time (same all time steps), batch, var]
    m = a.shape
    w = torch.zeros([lenF, m[1],m[2]])
    aa = F.relu(a[0:lenF,:,:]).view([lenF, m[1],m[2]])+0.1 # minimum 0.1. First dimension of a is repeat
    theta = F.relu(b[0:lenF,:,:]).view([lenF, m[1],m[2]])+0.5 # minimum 0.5
    t = torch.arange(0.5,lenF*1.0).view([lenF,1,1]).repeat([1,m[1],m[2]])
    t = t.cuda(aa.device)
    denom = (aa.lgamma().exp())*(theta**aa)
    mid= t**(aa-1)
    right=torch.exp(-t/theta)
    w = 1/denom*mid*right
    w = w/w.sum(0) # scale to 1 for each UH

    return w

class SnowSIMHYD8Differentiable(nn.Module):
    """
    SIMHYD with a groundwater-loss term plus an HBV-style snow module.

    Parameter order:
        [INSC, COEF, SQ, SMSC, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH]

    Inputs:
        inputs[..., 0] = precipitation P, mm/day
        inputs[..., 1] = temperature T, deg C
        inputs[..., 2] = PET, mm/day
    """

    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4, rain_snow_gain=5.0):
        super(SnowSIMHYD8Differentiable, self).__init__()
        assert mode in ('normal', 'analysis')
        self.mode = mode
        self.theta_is_raw = theta_is_raw
        self.smooth = smooth
        self.eps = eps
        self.rain_snow_gain = rain_snow_gain

    def _pos(self, x):
        if self.smooth:
            return 0.5 * (x + torch.sqrt(x * x + self.eps ** 2))
        return torch.relu(x)

    def _min(self, a, b):
        return a - self._pos(a - b)

    def _expand(self, theta):
        if self.theta_is_raw:
            theta = torch.sigmoid(theta)

        INSC = 0.5 + theta[:, 0:1] * (5.0 - 0.5)
        COEF = 50.0 + theta[:, 1:2] * (400.0 - 50.0)
        SQ = 0.0 + theta[:, 2:3] * (6.0 - 0.0)
        SMSC = 50.0 + theta[:, 3:4] * (500.0 - 50.0)
        SUB = 0.0 + theta[:, 4:5] * (1.0 - 0.0)
        CRAK = 0.0 + theta[:, 5:6] * (1.0 - 0.0)
        K = 0.003 + theta[:, 6:7] * (0.3 - 0.003)
        LG = 0.0 + theta[:, 7:8] * (1.0 - 0.0)
        TT = -2.5 + theta[:, 8:9] * (2.5 - (-2.5))
        CFMAX = 0.5 + theta[:, 9:10] * (10.0 - 0.5)
        CFR = 0.0 + theta[:, 10:11] * (0.1 - 0.0)
        CWH = 0.0 + theta[:, 11:12] * (0.2 - 0.0)
        return INSC, COEF, SQ, SMSC, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH

    @torch.no_grad()
    def denorm_params(self, theta):
        return torch.cat(self._expand(theta), dim=1)

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.5):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        INSC, COEF, SQ, SMSC, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH = self._expand(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            SMS = torch.zeros(B, 1, device=device, dtype=dtype)
            GW = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
        else:
            SMS = initial_state[:, 0:1]
            GW = initial_state[:, 1:2]
            SNOWPACK = initial_state[:, 2:3]
            MELTWATER = initial_state[:, 3:4]

        if lg_dyn_seq is not None:
            if lg_dyn_seq.shape[0] != B or lg_dyn_seq.shape[1] != Tlen:
                raise ValueError('lg_dyn_seq must have shape [B, T, 1] matching inputs')

        q_hist = []
        sms_hist = []
        gw_hist = []
        snow_hist = []
        meltwater_hist = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SMS0 = self._min(self._pos(SMS), SMSC)
            GW0 = self._pos(GW)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()

            RAIN = Pt * frac_rain
            SNOW = Pt * (1.0 - frac_rain)

            SNOWPACK1 = SNOWPACK0 + SNOW

            melt_pot = CFMAX * self._pos(Tt - TT)
            melt = self._min(melt_pot, SNOWPACK1)

            MELTWATER1 = MELTWATER0 + melt
            SNOWPACK2 = SNOWPACK1 - melt

            refreeze_pot = CFR * CFMAX * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)

            SNOWPACK3 = SNOWPACK2 + refreezing
            MELTWATER2 = MELTWATER1 - refreezing

            water_holding = CWH * SNOWPACK3
            tosoil = self._pos(MELTWATER2 - water_holding)

            MELTWATER_next = self._pos(MELTWATER2 - tosoil)
            SNOWPACK_next = self._pos(SNOWPACK3)

            Peff = RAIN + tosoil

            IMAX = self._min(INSC, E0t)
            INT = self._min(IMAX, Peff)
            INR = self._pos(Peff - INT)

            wetness = SMS0 / (SMSC + 1e-8)
            infil_cap = COEF * torch.exp(-SQ * wetness)
            RMO = self._min(infil_cap, INR)
            IRUN = self._pos(INR - RMO)

            SRUN = SUB * wetness * RMO
            REC = CRAK * wetness * (RMO - SRUN)
            REC = self._pos(REC)
            SMF = self._pos(RMO - SRUN - REC)

            POT = self._pos(E0t - INT)
            ETS_cap = 10.0 * wetness
            ETS = self._min(ETS_cap, POT)
            ETS = self._min(ETS, SMS0 + SMF)

            SMS_pre = SMS0 + SMF - ETS
            SOIL_EXCESS = self._pos(SMS_pre - SMSC)
            SMS_next = self._pos(SMS_pre - SOIL_EXCESS)
            REC_total = REC + SOIL_EXCESS

            BAS = K * GW0
            if lg_dyn_seq is None:
                LG_t = LG
            else:
                LG_dyn_t = lg_dyn_seq[:, t, :]
                LG_t = (1.0 - lg_dyn_weight) * LG + lg_dyn_weight * LG_dyn_t
            GW_next = self._pos(GW0 + REC_total - BAS - LG_t)
            Q = self._pos(IRUN + SRUN + BAS)

            SMS = SMS_next
            GW = GW_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            q_hist.append(Q)
            sms_hist.append(SMS)
            gw_hist.append(GW)
            snow_hist.append(SNOWPACK)
            meltwater_hist.append(MELTWATER)

        q_seq = torch.stack(q_hist, dim=1)
        sms_seq = torch.stack(sms_hist, dim=1)
        gw_seq = torch.stack(gw_hist, dim=1)
        snow_seq = torch.stack(snow_hist, dim=1)
        meltwater_seq = torch.stack(meltwater_hist, dim=1)

        if self.mode == 'normal':
            return q_seq

        return torch.cat([sms_seq, gw_seq, snow_seq, meltwater_seq, q_seq], dim=-1)


class MultiInv_SnowSIMHYDModel(torch.nn.Module):
    """
    Inversion LSTM infers one static Snow-SIMHYD parameter vector per basin.
    """

    def __init__(self, *, ninv, hiddeninv=256, inittime=0, drinv=0.5):
        super(MultiInv_SnowSIMHYDModel, self).__init__()
        self.lstminv = CudnnLstmModel(nx=ninv, ny=12, hiddenSize=hiddeninv, dr=drinv)
        self.simhyd = SnowSIMHYD8Differentiable(mode='normal', theta_is_raw=False, smooth=True)
        self.simhyd_analysis = SnowSIMHYD8Differentiable(mode='analysis', theta_is_raw=False, smooth=True)
        self.hiddeninv = hiddeninv
        self.inittime = inittime
        self.ny = 1

    def forward(self, x, z, doDropMC=False):
        param_seq = self.lstminv(z)
        theta = torch.sigmoid(param_seq[-1, :, :])

        x_bt = x.permute(1, 0, 2)
        if self.inittime > 0:
            warm_inputs = x_bt[:, :self.inittime, :]
            main_inputs = x_bt[:, self.inittime:, :]
            state_hist = self.simhyd_analysis(warm_inputs, theta)
            initial_state = state_hist[:, -1, 0:4]
            q_seq = self.simhyd(main_inputs, theta, initial_state=initial_state)
        else:
            q_seq = self.simhyd(x_bt, theta)

        return q_seq.permute(1, 0, 2)


class MultiInv_SnowSIMHYDMulTDModel(torch.nn.Module):
    """
    Experimental multi-component Snow-SIMHYD.

    Differences from MultiInv_SnowSIMHYDModel:
      - multiple Snow-SIMHYD components per basin
      - optional learned component weights
      - optional routing after component mixing
      - semi-dynamic LG: only groundwater loss changes through time
    """

    def __init__(self, *, ninv, nfea=12, nmul=4, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.5):
        super(MultiInv_SnowSIMHYDMulTDModel, self).__init__()
        self.ninv = ninv
        self.nfea = nfea
        self.nmul = nmul
        self.hiddeninv = hiddeninv
        self.inittime = inittime
        self.routOpt = routOpt
        self.comprout = comprout
        self.compwts = compwts
        self.lgdyn = lgdyn
        self.lgdynweight = lgdynweight
        self.ny = 1

        self.nstaticpm = nfea * nmul
        self.nroutpm = nmul * 2 if comprout else 2
        self.nwtspm = nmul if compwts else 0
        self.ndynpm = nmul if lgdyn else 0
        self.ntp = self.nstaticpm + self.nroutpm + self.nwtspm + self.ndynpm

        self.lstminv = CudnnLstmModel(nx=ninv, ny=self.ntp, hiddenSize=hiddeninv, dr=drinv)
        self.simhyd = SnowSIMHYD8Differentiable(mode='normal', theta_is_raw=False, smooth=True)
        self.simhyd_analysis = SnowSIMHYD8Differentiable(mode='analysis', theta_is_raw=False, smooth=True)

    def _route_q(self, qin, rtwts):
        # qin: [time, batch, 1], rtwts: [batch, 2] in [0, 1]
        Nstep = qin.shape[0]
        lenF = 15
        routscaLst = [[0, 2.9], [0, 6.5]]
        rf = qin.permute([1, 2, 0])  # [batch, 1, time]
        tempa = routscaLst[0][0] + rtwts[:, 0] * (routscaLst[0][1] - routscaLst[0][0])
        tempb = routscaLst[1][0] + rtwts[:, 1] * (routscaLst[1][1] - routscaLst[1][0])
        rept = max(Nstep, lenF)
        routa = tempa.repeat(rept, 1).unsqueeze(-1)
        routb = tempb.repeat(rept, 1).unsqueeze(-1)
        UH = UH_gamma(routa, routb, lenF=lenF).permute([1, 2, 0])
        qout = UH_conv(rf, UH).permute([2, 0, 1])
        return qout

    def forward(self, x, z, doDropMC=False):
        gen = self.lstminv(z)
        nt_dyn, ngage, _ = gen.shape
        nt_x = x.shape[0]

        params0 = gen[-1, :, :]
        static0 = params0[:, 0:self.nstaticpm]
        snowpara = torch.sigmoid(static0).view(ngage, self.nfea, self.nmul)

        cursor = self.nstaticpm
        routpara0 = params0[:, cursor:cursor + self.nroutpm]
        if self.comprout is False:
            routpara = torch.sigmoid(routpara0)
        else:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        cursor += self.nroutpm

        if self.compwts is False:
            wts = None
        else:
            wtspara = params0[:, cursor:cursor + self.nwtspm]
            wts = F.softmax(wtspara, dim=-1)
            cursor += self.nwtspm

        if self.lgdyn is False:
            lg_dyn = None
        else:
            lg_dyn = torch.sigmoid(gen[:, :, cursor:cursor + self.ndynpm])  # [Tmain, B, mu]

        x_rep = x.unsqueeze(2).repeat(1, 1, self.nmul, 1).view(nt_x, ngage * self.nmul, x.shape[2])
        x_bt = x_rep.permute(1, 0, 2)
        theta = snowpara.permute(0, 2, 1).contiguous().view(ngage * self.nmul, self.nfea)

        if lg_dyn is None:
            lg_bt = None
        else:
            lg_bt = lg_dyn.permute(1, 2, 0).contiguous().view(ngage * self.nmul, nt_dyn, 1)

        if self.inittime > 0:
            warm_inputs = x_bt[:, :self.inittime, :]
            main_inputs = x_bt[:, self.inittime:, :]
            if lg_bt is None:
                main_lg = None
            else:
                main_lg = lg_bt
            state_hist = self.simhyd_analysis(
                warm_inputs,
                theta,
                lg_dyn_seq=None,
                lg_dyn_weight=self.lgdynweight)
            initial_state = state_hist[:, -1, 0:4]
            q_seq = self.simhyd(
                main_inputs,
                theta,
                initial_state=initial_state,
                lg_dyn_seq=main_lg,
                lg_dyn_weight=self.lgdynweight)
        else:
            if lg_bt is not None and lg_bt.shape[1] != x_bt.shape[1]:
                raise ValueError('dynamic parameter length must match x when inittime=0')
            q_seq = self.simhyd(
                x_bt,
                theta,
                lg_dyn_seq=lg_bt,
                lg_dyn_weight=self.lgdynweight)

        q_comp = q_seq.view(ngage, self.nmul, q_seq.shape[1], 1).permute(2, 0, 1, 3)

        if self.routOpt is True and self.comprout is True:
            q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
            q_routed = self._route_q(q_for_routing, routpara)
            q_routed = q_routed.view(q_comp.shape[0], ngage, self.nmul, 1)
            if wts is None:
                out = torch.mean(q_routed, dim=2)
            else:
                out = torch.sum(q_routed * wts.unsqueeze(0).unsqueeze(-1), dim=2)
            return out

        if wts is None:
            q_mix = torch.mean(q_comp, dim=2)
        else:
            q_mix = torch.sum(q_comp * wts.unsqueeze(0).unsqueeze(-1), dim=2)

        if self.routOpt is True:
            out = self._route_q(q_mix, routpara)
