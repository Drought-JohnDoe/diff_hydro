import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def UH_conv(x, UH, viewmode=1):
    mm = x.shape
    nb = mm[0]
    m = UH.shape[-1]
    padd = m - 1
    if viewmode == 1:
        xx = x.view([1, nb, mm[-1]])
        w = UH.view([nb, 1, m])
        groups = nb
    y = F.conv1d(xx, torch.flip(w, [2]), groups=groups, padding=padd, stride=1, bias=None)
    y = y[:, :, 0:-padd]
    return y.view(mm)


def UH_gamma(a, b, lenF=10):
    m = a.shape
    aa = F.relu(a[0:lenF, :, :]).view([lenF, m[1], m[2]]) + 0.1
    theta = F.relu(b[0:lenF, :, :]).view([lenF, m[1], m[2]]) + 0.5
    t = torch.arange(0.5, lenF * 1.0).view([lenF, 1, 1]).repeat([1, m[1], m[2]])
    t = t.to(device=aa.device, dtype=aa.dtype)
    denom = (aa.lgamma().exp()) * (theta**aa)
    w = (1 / denom) * (t ** (aa - 1)) * torch.exp(-t / theta)
    w = w / w.sum(0)
    return w


class SafeLstmModel(nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.dr = dr
        self.linearIn = nn.Linear(nx, hiddenSize)
        self.lstm = nn.LSTM(input_size=hiddenSize, hidden_size=hiddenSize, num_layers=1, dropout=0.0)
        self.linearOut = nn.Linear(hiddenSize, ny)
        self._force_cpu_fallback = False

    def _forward_impl(self, x, doDropMC=False, dropoutFalse=False):
        x0 = F.relu(self.linearIn(x))
        if self.dr > 0 and (doDropMC or self.training) and not dropoutFalse:
            x0 = F.dropout(x0, p=self.dr, training=True)
        out_lstm, _ = self.lstm(x0)
        return self.linearOut(out_lstm)

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        if self._force_cpu_fallback:
            if next(self.parameters()).device.type != "cpu":
                self.cpu()
            out = self._forward_impl(x.cpu(), doDropMC=doDropMC, dropoutFalse=dropoutFalse)
            return out.to(x.device)
        try:
            return self._forward_impl(x, doDropMC=doDropMC, dropoutFalse=dropoutFalse)
        except RuntimeError as err:
            msg = str(err).lower()
            if x.device.type == "cuda" and ("cublas runtime error" in msg or "cuda" in msg):
                self._force_cpu_fallback = True
                self.cpu()
                warnings.warn(
                    "SafeLstmModel falling back to CPU after CUDA failure; continuing on CPU.",
                    RuntimeWarning,
                )
                out = self._forward_impl(x.cpu(), doDropMC=doDropMC, dropoutFalse=dropoutFalse)
                return out.to(x.device)
            raise


class DynamicSimHydModelFiveDifferentiable(nn.Module):
    def __init__(
        self,
        mode="normal",
        theta_is_raw=False,
        smooth=True,
        eps=1e-4,
        rain_snow_gain=5.0,
        dynamic_sq=True,
        dynamic_etgam=True,
        dynamic_partition=True,
        dynamic_cfmax_snow=True,
        dynamic_routing_scale=False,
        dynamic_all=False,
        dyn_hidden=32,
    ):
        super().__init__()
        self.mode = mode
        self.theta_is_raw = theta_is_raw
        self.smooth = smooth
        self.eps = eps
        self.rain_snow_gain = rain_snow_gain
        self.dynamic_sq = dynamic_sq or dynamic_all
        self.dynamic_etgam = dynamic_etgam or dynamic_all
        self.dynamic_partition = dynamic_partition or dynamic_all
        self.dynamic_cfmax_snow = dynamic_cfmax_snow or dynamic_all
        self.dynamic_routing_scale = dynamic_routing_scale or dynamic_all

        self.dyn_out_dim = 0
        self.dyn_slices = {}
        if self.dynamic_sq:
            self.dyn_slices["sq"] = slice(self.dyn_out_dim, self.dyn_out_dim + 1)
            self.dyn_out_dim += 1
        if self.dynamic_etgam:
            self.dyn_slices["etgam"] = slice(self.dyn_out_dim, self.dyn_out_dim + 1)
            self.dyn_out_dim += 1
        if self.dynamic_partition:
            self.dyn_slices["partition"] = slice(self.dyn_out_dim, self.dyn_out_dim + 3)
            self.dyn_out_dim += 3
        if self.dynamic_cfmax_snow:
            self.dyn_slices["cfmax"] = slice(self.dyn_out_dim, self.dyn_out_dim + 1)
            self.dyn_out_dim += 1

        self.dynHead = None
        self._dynhead_force_cpu_fallback = False
        if self.dyn_out_dim > 0:
            self.dynHead = nn.Sequential(
                nn.Linear(9, dyn_hidden),
                nn.ReLU(),
                nn.Linear(dyn_hidden, dyn_hidden),
                nn.ReLU(),
                nn.Linear(dyn_hidden, self.dyn_out_dim),
            )

    def _pos(self, x):
        if self.smooth:
            return 0.5 * (x + torch.sqrt(x * x + self.eps**2))
        return torch.relu(x)

    def _min(self, a, b):
        return a - self._pos(a - b)

    def _run_dyn_head(self, dyn_in):
        if self.dynHead is None:
            return None
        if self._dynhead_force_cpu_fallback:
            if next(self.dynHead.parameters()).device.type != "cpu":
                self.dynHead.cpu()
            out = self.dynHead(dyn_in.cpu())
            return out.to(dyn_in.device)
        try:
            return self.dynHead(dyn_in)
        except RuntimeError as err:
            msg = str(err).lower()
            if dyn_in.device.type == "cuda" and ("cublas runtime error" in msg or "cuda" in msg):
                self._dynhead_force_cpu_fallback = True
                self.dynHead.cpu()
                warnings.warn("Dynamic head falling back to CPU after CUDA failure.", RuntimeWarning)
                out = self.dynHead(dyn_in.cpu())
                return out.to(dyn_in.device)
            raise

    def _expand(self, theta):
        if self.theta_is_raw:
            theta = torch.sigmoid(theta)
        INSC = 0.5 + theta[:, 0:1] * (5.0 - 0.5)
        COEF = 50.0 + theta[:, 1:2] * (400.0 - 50.0)
        SQ = theta[:, 2:3] * 6.0
        SMSC = 50.0 + theta[:, 3:4] * (500.0 - 50.0)
        SUB = theta[:, 4:5]
        CRAK = theta[:, 5:6]
        K = 0.003 + theta[:, 6:7] * (0.3 - 0.003)
        LG = theta[:, 7:8] * 0.2
        TT = -2.5 + theta[:, 8:9] * 5.0
        CFMAX = 0.5 + theta[:, 9:10] * (10.0 - 0.5)
        CFR = theta[:, 10:11] * 0.1
        CWH = theta[:, 11:12] * 0.2
        SG_CRIT = theta[:, 12:13] * 300.0
        return INSC, COEF, SQ, SMSC, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH, SG_CRIT

    def _seasonal_feats(self, inputs, t, device, dtype):
        if inputs.shape[-1] >= 5:
            return inputs[:, t, 3:4], inputs[:, t, 4:5]
        ang = 2.0 * math.pi * float(t % 365) / 365.0
        return (
            torch.full((inputs.shape[0], 1), math.sin(ang), device=device, dtype=dtype),
            torch.full((inputs.shape[0], 1), math.cos(ang), device=device, dtype=dtype),
        )


class DynamicSimHydModelFiveDifferentiablePhysicalFix(DynamicSimHydModelFiveDifferentiable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimple(
    DynamicSimHydModelFiveDifferentiablePhysicalFix
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _expand_simple(self, theta):
        if self.theta_is_raw:
            theta = torch.sigmoid(theta)
        INSC = 0.5 + theta[:, 0:1] * (5.0 - 0.5)
        COEF = 50.0 + theta[:, 1:2] * (400.0 - 50.0)
        SQ = theta[:, 2:3] * 6.0
        SMSC_legacy = 50.0 + theta[:, 3:4] * (500.0 - 50.0)
        SUB = theta[:, 4:5]
        CRAK = theta[:, 5:6]
        K = 0.003 + theta[:, 6:7] * (0.3 - 0.003)
        TT = -2.5 + theta[:, 8:9] * 5.0
        CFMAX = 0.5 + theta[:, 9:10] * (10.0 - 0.5)
        CFR = theta[:, 10:11] * 0.1
        CWH = theta[:, 11:12] * 0.2
        theta_ab = 0.5 + theta[:, 13:14] * 0.5
        theta_ak = 1.0 + theta[:, 14:15] * 9.0
        theta_cap = 10.0 + theta[:, 15:16] * (1500.0 - 10.0)
        theta_efmax = 0.5 + theta[:, 16:17] * 0.5
        theta_wetpoint = 0.3 + theta[:, 17:18] * 0.6
        return (
            INSC,
            COEF,
            SQ,
            SMSC_legacy,
            SUB,
            CRAK,
            K,
            TT,
            CFMAX,
            CFR,
            CWH,
            theta_ab,
            theta_ak,
            theta_cap,
            theta_efmax,
            theta_wetpoint,
        )

    def forward(
        self,
        inputs,
        theta,
        initial_state=None,
        lg_dyn_seq=None,
        lg_dyn_weight=0.6,
        snow_frac_raw=None,
        return_diagnostics=False,
        return_final_state=False,
        return_regularization=False,
    ):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype
        (
            INSC,
            COEF,
            SQ,
            SMSC_legacy,
            SUB,
            CRAK,
            K,
            TT,
            CFMAX_base,
            CFR,
            CWH,
            theta_ab,
            theta_ak,
            theta_cap,
            theta_efmax,
            theta_wetpoint,
        ) = self._expand_simple(theta)
        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])
        if initial_state is None:
            SA = torch.zeros(B, 1, device=device, dtype=dtype)
            GW = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
            init_sa, init_gw, init_snow, init_melt = SA, GW, SNOWPACK, MELTWATER
        else:
            SA = initial_state[:, 0:1]
            GW = initial_state[:, 1:2]
            SNOWPACK = initial_state[:, 2:3]
            MELTWATER = initial_state[:, 3:4]
            init_sa, init_gw, init_snow, init_melt = SA, GW, SNOWPACK, MELTWATER
        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()
        q_hist = []
        diag_hist = {}
        if return_diagnostics:
            for name in [
                "SNOWPACK_prev",
                "MELTWATER_prev",
                "Sa_prev",
                "GW_prev",
                "SNOWPACK",
                "MELTWATER",
                "Sa",
                "GW",
                "precipitation",
                "rainfall",
                "snowfall",
                "snowmelt",
                "refreezing",
                "snow_release",
                "PL",
                "interception_evaporation",
                "actual_ET",
                "LAI_t",
                "LAI_scalar",
                "Smoist",
                "theta_cap",
                "alpha",
                "P_accessible",
                "P_inaccessible",
                "Sa_overflow",
                "surface_runoff",
                "interflow",
                "recharge_to_groundwater",
                "baseflow",
                "Q_process",
                "groundwater_loss",
                "channel_loss",
                "gate_loss",
                "partition_sum_error",
                "soil_local_residual",
                "gw_local_residual",
                "snowpack_local_residual",
                "meltwater_local_residual",
                "snow_total_local_residual",
                "process_local_residual",
                "INSC",
                "COEF_t",
                "SQ_t",
                "K_t",
                "TT",
                "CFMAX_t",
                "CFR",
                "CWH",
            ]:
                diag_hist[name] = []
        sq_mult_hist = []
        cfmax_mult_hist = []
        recharge_frac_hist = []
        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]
            SA0 = self._min(self._pos(SA), theta_cap)
            GW0 = self._pos(GW)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            Smoist0 = torch.clamp(SA0 / (theta_cap + 1e-8), 0.0, 1.0)
            sin_t, cos_t = self._seasonal_feats(inputs, t, device, dtype)
            dyn_in = torch.cat(
                [Pt / 20.0, Tt / 20.0, E0t / 10.0, SA0 / 300.0, Smoist0, GW0 / 300.0, SNOWPACK0 / 300.0, sin_t, cos_t],
                dim=1,
            )
            dyn_raw = self._run_dyn_head(dyn_in)
            m_sq = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices["sq"]]) if self.dynamic_sq else torch.ones_like(SQ)
            SQ_t = torch.clamp(SQ * m_sq, 0.0, 6.0)
            if self.dynamic_cfmax_snow:
                m_cf = 0.7 + 0.8 * torch.sigmoid(dyn_raw[:, self.dyn_slices["cfmax"]])
                m_cf_eff = snow_mask * m_cf + (1.0 - snow_mask)
            else:
                m_cf = torch.ones_like(SQ)
                m_cf_eff = m_cf
            CFMAX_t = CFMAX_base * m_cf_eff
            frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT)) if self.smooth else (Tt >= TT).float()
            rainfall = Pt * frac_rain
            snowfall = Pt * (1.0 - frac_rain)
            SNOWPACK1 = SNOWPACK0 + snowfall
            snowmelt_pot = CFMAX_t * self._pos(Tt - TT)
            snowmelt = self._min(snowmelt_pot, SNOWPACK1)
            SNOWPACK2 = self._pos(SNOWPACK1 - snowmelt)
            MELTWATER1 = MELTWATER0 + snowmelt
            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            MELTWATER2 = self._pos(MELTWATER1 - refreezing)
            SNOWPACK3 = SNOWPACK2 + refreezing
            snow_holding = CWH * SNOWPACK3
            snow_release_raw = self._pos(MELTWATER2 - snow_holding)
            snow_release = self._min(snow_release_raw, MELTWATER2)
            MELTWATER_next = self._pos(MELTWATER2 - snow_release)
            SNOWPACK_next = self._pos(SNOWPACK3)
            PL = rainfall + snow_release
            INT = self._min(INSC, self._min(E0t, PL))
            PL_after_int = self._pos(PL - INT)
            POT = self._pos(E0t - INT)
            if inputs.shape[-1] >= 6:
                LAI_t = self._pos(inputs[:, t, 5:6])
                LAI_scalar = 0.25 + 0.75 * torch.clamp(LAI_t / 5.0, 0.0, 1.0)
            else:
                LAI_t = torch.zeros_like(Pt)
                LAI_scalar = torch.ones_like(Pt)
            alpha = theta_ab * torch.pow(torch.clamp(1.0 - Smoist0, min=0.0), theta_ak)
            alpha = torch.clamp(alpha, 0.0, 1.0)
            P_accessible = alpha * PL_after_int
            P_inaccessible = (1.0 - alpha) * PL_after_int
            SA_pre = SA0 + P_accessible
            ET_stress = torch.clamp(Smoist0 / torch.clamp(theta_wetpoint, min=1e-6), 0.0, 1.0)
            ET_a_pot = POT * theta_efmax * ET_stress * LAI_scalar
            ET_a = self._min(ET_a_pot, SA_pre)
            SA_after_ET = self._pos(SA_pre - ET_a)
            SA_overflow = self._pos(SA_after_ET - theta_cap)
            SA_next = self._pos(SA_after_ET - SA_overflow)
            water_for_partition = self._pos(P_inaccessible + SA_overflow)
            base_surface = torch.clamp(SUB, 1e-6, 1.0)
            base_recharge = torch.clamp((1.0 - SUB) * CRAK, 1e-6, 1.0)
            base_interflow = torch.clamp((1.0 - SUB) * (1.0 - CRAK), 1e-6, 1.0)
            base_logits = torch.cat([torch.log(base_surface), torch.log(base_interflow), torch.log(base_recharge)], dim=1)
            part_logits = base_logits + dyn_raw[:, self.dyn_slices["partition"]] if self.dynamic_partition else base_logits
            part_frac = F.softmax(part_logits, dim=1)
            f_surface, f_inter, f_recharge = part_frac[:, 0:1], part_frac[:, 1:2], part_frac[:, 2:3]
            SRUN = f_surface * water_for_partition
            IFLOW = f_inter * water_for_partition
            REC = f_recharge * water_for_partition
            GW1 = GW0 + REC
            BAS_raw = K * GW1
            BAS = self._min(BAS_raw, GW1)
            GW_next = self._pos(GW1 - BAS)
            Q_process = self._pos(SRUN + IFLOW + BAS)
            process_local_residual = Pt - INT - ET_a - Q_process - (
                (SNOWPACK_next - SNOWPACK0) + (MELTWATER_next - MELTWATER0) + (SA_next - SA0) + (GW_next - GW0)
            )
            soil_local_residual = P_accessible - ET_a - SA_overflow - (SA_next - SA0)
            gw_local_residual = REC - BAS - (GW_next - GW0)
            snowpack_local_residual = snowfall - snowmelt + refreezing - (SNOWPACK_next - SNOWPACK0)
            meltwater_local_residual = snowmelt - refreezing - snow_release - (MELTWATER_next - MELTWATER0)
            snow_total_local_residual = snowfall - snow_release - (SNOWPACK_next - SNOWPACK0) - (MELTWATER_next - MELTWATER0)
            SA, GW, SNOWPACK, MELTWATER = SA_next, GW_next, SNOWPACK_next, MELTWATER_next
            q_hist.append(Q_process)
            sq_mult_hist.append(m_sq)
            cfmax_mult_hist.append(m_cf_eff)
            recharge_frac_hist.append(f_recharge)
            if return_diagnostics:
                diag_vals = {
                    "SNOWPACK_prev": SNOWPACK0,
                    "MELTWATER_prev": MELTWATER0,
                    "Sa_prev": SA0,
                    "GW_prev": GW0,
                    "SNOWPACK": SNOWPACK_next,
                    "MELTWATER": MELTWATER_next,
                    "Sa": SA_next,
                    "GW": GW_next,
                    "precipitation": Pt,
                    "rainfall": rainfall,
                    "snowfall": snowfall,
                    "snowmelt": snowmelt,
                    "refreezing": refreezing,
                    "snow_release": snow_release,
                    "PL": PL,
                    "interception_evaporation": INT,
                    "actual_ET": ET_a,
                    "LAI_t": LAI_t,
                    "LAI_scalar": LAI_scalar,
                    "Smoist": Smoist0,
                    "theta_cap": theta_cap,
                    "alpha": alpha,
                    "P_accessible": P_accessible,
                    "P_inaccessible": P_inaccessible,
                    "Sa_overflow": SA_overflow,
                    "surface_runoff": SRUN,
                    "interflow": IFLOW,
                    "recharge_to_groundwater": REC,
                    "baseflow": BAS,
                    "Q_process": Q_process,
                    "groundwater_loss": torch.zeros_like(Q_process),
                    "channel_loss": torch.zeros_like(Q_process),
                    "gate_loss": torch.zeros_like(Q_process),
                    "partition_sum_error": torch.abs(f_surface + f_inter + f_recharge - 1.0),
                    "soil_local_residual": soil_local_residual,
                    "gw_local_residual": gw_local_residual,
                    "snowpack_local_residual": snowpack_local_residual,
                    "meltwater_local_residual": meltwater_local_residual,
                    "snow_total_local_residual": snow_total_local_residual,
                    "process_local_residual": process_local_residual,
                    "INSC": INSC,
                    "COEF_t": COEF,
                    "SQ_t": SQ_t,
                    "K_t": K,
                    "TT": TT,
                    "CFMAX_t": CFMAX_t,
                    "CFR": CFR,
                    "CWH": CWH,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)
        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SA, GW, SNOWPACK, MELTWATER], dim=1)
        reg_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_part = torch.tensor(0.0, device=device, dtype=dtype)
        if sq_mult_hist:
            sq_seq = torch.stack(sq_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((sq_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((sq_seq[:, 1:, :] - sq_seq[:, :-1, :]) ** 2)
        if cfmax_mult_hist:
            cf_seq = torch.stack(cfmax_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((cf_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((cf_seq[:, 1:, :] - cf_seq[:, :-1, :]) ** 2)
        if recharge_frac_hist:
            r_seq = torch.stack(recharge_frac_hist, dim=1)
            reg_part = torch.mean(torch.relu(r_seq - 0.85) ** 2)
        sum_p = torch.clamp(P.sum(dim=1), min=1e-6)
        drift_loss = (
            torch.mean(torch.abs(SA - init_sa) / sum_p)
            + torch.mean(torch.abs(GW - init_gw) / sum_p)
            + torch.mean(torch.abs(SNOWPACK - init_snow) / sum_p)
            + torch.mean(torch.abs(MELTWATER - init_melt) / sum_p)
        )
        reg_terms = {
            "dynamic_amplitude_loss": reg_amp,
            "dynamic_smoothness_loss": reg_smooth,
            "partition_entropy_loss": reg_part,
            "storage_drift_loss": drift_loss,
        }
        if return_diagnostics:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization and return_final_state:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization:
                return q_seq, diag_out, reg_terms
            if return_final_state:
                return q_seq, diag_out, final_state
            return q_seq, diag_out
        if return_regularization and return_final_state:
            return q_seq, final_state, reg_terms
        if return_regularization:
            return q_seq, reg_terms
        if return_final_state:
            return q_seq, final_state
        return q_seq


class MultiInv_DynamicSimHydModelFive(nn.Module):
    def __init__(
        self,
        *,
        ninv,
        nmul=4,
        nattr=35,
        hiddeninv=256,
        drinv=0.5,
        inittime=0,
        routOpt=True,
        comprout=False,
        compwts=True,
        lgdyn=True,
        lgdynweight=0.6,
        dynamic_sq=True,
        dynamic_etgam=True,
        dynamic_partition=True,
        dynamic_cfmax_snow=True,
        dynamic_routing_scale=False,
        dynamic_all=False,
        reg_amp_w=1e-3,
        reg_smooth_w=1e-3,
        reg_part_w=1e-3,
    ):
        super().__init__()
        self.ninv = ninv
        self.nmul = nmul
        self.nattr = nattr
        self.hiddeninv = hiddeninv
        self.inittime = inittime
        self.routOpt = routOpt
        self.comprout = comprout
        self.compwts = compwts
        self.lgdyn = lgdyn
        self.lgdynweight = lgdynweight
        self.dynamic_sq = dynamic_sq or dynamic_all
        self.dynamic_etgam = dynamic_etgam or dynamic_all
        self.dynamic_partition = dynamic_partition or dynamic_all
        self.dynamic_cfmax_snow = dynamic_cfmax_snow or dynamic_all
        self.dynamic_routing_scale = dynamic_routing_scale or dynamic_all
        self.dynamic_all = dynamic_all
        self.reg_amp_w = reg_amp_w
        self.reg_smooth_w = reg_smooth_w
        self.reg_part_w = reg_part_w
        self._last_aux_loss = None
        self._last_aux_terms = None
        self.nfea = 13
        self.ny = 1
        self.nstaticpm = self.nfea * nmul
        self.nroutpm = nmul * 2 if comprout else 2
        self.nwtspm = nmul if compwts else 0
        self.ndynpm = nmul if lgdyn else 0
        self.staticFeat = nn.Sequential(
            nn.Linear(nattr, hiddeninv),
            nn.ReLU(),
            nn.Linear(hiddeninv, hiddeninv),
            nn.ReLU(),
        )
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)
        if self.lgdyn:
            self.lstmdyn = SafeLstmModel(nx=ninv, ny=self.ndynpm, hiddenSize=hiddeninv, dr=drinv)
            self.lgAttr = nn.Linear(nattr, self.ndynpm)
        if self.dynamic_routing_scale:
            self.routeDynHead = nn.Sequential(
                nn.Linear(5, hiddeninv // 2),
                nn.ReLU(),
                nn.Linear(hiddeninv // 2, 1),
            )
        else:
            self.routeDynHead = None
        self.simhyd = DynamicSimHydModelFiveDifferentiable(
            mode="normal",
            theta_is_raw=False,
            smooth=True,
            dynamic_sq=self.dynamic_sq,
            dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition,
            dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale,
            dynamic_all=self.dynamic_all,
        )
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiable(
            mode="analysis",
            theta_is_raw=False,
            smooth=True,
            dynamic_sq=self.dynamic_sq,
            dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition,
            dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale,
            dynamic_all=self.dynamic_all,
        )
        comp_bias = torch.linspace(-0.15, 0.15, nmul).view(1, 1, nmul).repeat(1, self.nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.compWeightBias = nn.Parameter(torch.linspace(-0.2, 0.2, nmul)) if self.nwtspm > 0 else None

    def get_auxiliary_loss(self):
        return self._last_aux_loss

    def _route_q(self, qin, rtwts):
        Nstep = qin.shape[0]
        lenF = 15
        rf = qin.permute([1, 2, 0])
        tempa = 0.0 + rtwts[:, 0] * 2.9
        tempb = 0.0 + rtwts[:, 1] * 6.5
        rept = max(Nstep, lenF)
        routa = tempa.repeat(rept, 1).unsqueeze(-1)
        routb = tempb.repeat(rept, 1).unsqueeze(-1)
        UH = UH_gamma(routa, routb, lenF=lenF).permute([1, 2, 0])
        return UH_conv(rf, UH).permute([2, 0, 1])

    def _route_q_dynamic_scale(self, qin, rtwts, x_base):
        T, B, _ = qin.shape
        route_b_static = rtwts[:, 1:2] * 6.5
        q_out = []
        q_prev = torch.zeros(B, 1, device=qin.device, dtype=qin.dtype)
        route_mult_hist = []
        for t in range(T):
            qin_t = qin[t, :, :]
            p_t = x_base[t, :, 0:1]
            smsc_t = torch.clamp(x_base[t, :, 2:3], min=1e-6)
            sm_t = x_base[t, :, 1:2]
            wetness_t = torch.clamp(sm_t / smsc_t, 0.0, 2.0)
            sin_t = x_base[t, :, 3:4] if x_base.shape[-1] >= 5 else torch.zeros_like(qin_t)
            cos_t = x_base[t, :, 4:5] if x_base.shape[-1] >= 5 else torch.ones_like(qin_t)
            dyn_in = torch.cat([p_t / 20.0, wetness_t, qin_t / 20.0, sin_t, cos_t], dim=1)
            m_route = 0.75 + 0.75 * torch.sigmoid(self.routeDynHead(dyn_in))
            route_b_t = torch.clamp(route_b_static * m_route, min=0.1, max=12.0)
            alpha = torch.exp(-1.0 / route_b_t)
            q_now = alpha * q_prev + (1.0 - alpha) * qin_t
            q_out.append(q_now)
            q_prev = q_now
            route_mult_hist.append(m_route)
        return torch.stack(q_out, dim=0), torch.stack(route_mult_hist, dim=0)

    def _mix_component_tensor(self, tensor_comp, ngage):
        tensor4 = tensor_comp.view(ngage, self.nmul, tensor_comp.shape[1], tensor_comp.shape[2]).permute(2, 0, 1, 3)
        if self.nwtspm == 0:
            return torch.mean(tensor4, dim=2)
        return torch.sum(tensor4 * self._last_wts.unsqueeze(0).unsqueeze(-1), dim=2)


class MultiInv_DynamicSimHydModelSix(MultiInv_DynamicSimHydModelFive):
    def __init__(
        self,
        *,
        component_routing=True,
        dry_channel_loss=True,
        zero_flow_gate=True,
        channel_loss_max=0.60,
        zero_gate_hidden=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.component_routing = component_routing
        self.dry_channel_loss = dry_channel_loss
        self.zero_flow_gate_enabled = zero_flow_gate
        self.channel_loss_max = channel_loss_max
        self.zero_gate_hidden = self.hiddeninv // 2 if zero_gate_hidden is None else zero_gate_hidden
        self.nroutpm = self.nmul * 2 if self.component_routing else (self.nmul * 2 if self.comprout else 2)
        self.staticOut = nn.Linear(self.hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)
        self.channelLossHead = nn.Linear(self.nattr, self.nmul)
        self.zeroFlowGate = nn.Sequential(
            nn.Linear(5, self.zero_gate_hidden),
            nn.ReLU(),
            nn.Linear(self.zero_gate_hidden, self.nmul),
        )

    def _component_tensor_4d(self, tensor_comp, ngage):
        return tensor_comp.view(ngage, self.nmul, tensor_comp.shape[1], tensor_comp.shape[2]).permute(2, 0, 1, 3)

    def _theta_to_smsc(self, theta, ngage):
        theta4 = theta.view(ngage, self.nmul, self.nfea)
        return 50.0 + theta4[:, :, 3:4] * (500.0 - 50.0)

    def _apply_channel_loss(self, q_comp, diag_comp, theta, basin_attr, ngage):
        if not self.dry_channel_loss:
            zeros = torch.zeros_like(q_comp)
            return q_comp, zeros, zeros
        soil_m = self._component_tensor_4d(diag_comp["soil_moisture"], ngage) if "soil_moisture" in diag_comp else torch.ones_like(q_comp) * 100.0
        smsc = self._theta_to_smsc(theta, ngage).unsqueeze(0)
        wetness = torch.clamp(soil_m / torch.clamp(smsc, min=1e-6), 0.0, 1.0)
        dryness = 1.0 - wetness
        gamma = self.channel_loss_max * torch.sigmoid(self.channelLossHead(basin_attr))
        gamma = gamma.unsqueeze(0).unsqueeze(-1)
        loss_frac = 1.0 - torch.exp(-gamma * dryness)
        loss_frac = torch.clamp(loss_frac, 0.0, 0.95)
        q_after = q_comp * (1.0 - loss_frac)
        return torch.clamp(q_after, min=0.0), q_comp - q_after, loss_frac

    def _apply_zero_flow_gate(self, q_comp, x, diag_comp, theta, ngage):
        if not self.zero_flow_gate_enabled:
            return q_comp, torch.ones_like(q_comp)
        T, B, M, _ = q_comp.shape
        if "soil_moisture" in diag_comp:
            soil_m = self._component_tensor_4d(diag_comp["soil_moisture"], ngage)
            smsc = self._theta_to_smsc(theta, ngage).unsqueeze(0)
            wetness = torch.clamp(soil_m / torch.clamp(smsc, min=1e-6), 0.0, 1.0)
        else:
            wetness = torch.ones_like(q_comp) * 0.5
        p_t = x[:, :, 0:1].unsqueeze(2).repeat(1, 1, M, 1)
        sin_t = x[:, :, 3:4].unsqueeze(2).repeat(1, 1, M, 1) if x.shape[-1] >= 5 else torch.zeros_like(q_comp)
        cos_t = x[:, :, 4:5].unsqueeze(2).repeat(1, 1, M, 1) if x.shape[-1] >= 5 else torch.ones_like(q_comp)
        gate_in = torch.cat([p_t / 20.0, wetness, q_comp / 20.0, sin_t, cos_t], dim=-1)
        gate_in_flat = gate_in.view(T * B * M, 5)
        logits_all = self.zeroFlowGate(gate_in_flat)
        comp_idx = torch.arange(M, device=q_comp.device).view(1, 1, M, 1).repeat(T, B, 1, 1).view(T * B * M, 1)
        logits = logits_all.gather(1, comp_idx)
        p_flow = torch.sigmoid(logits).view(T, B, M, 1)
        return torch.clamp(p_flow * q_comp, min=0.0), p_flow

    def _mix_or_mean(self, tensor4, wts):
        if wts is None:
            return torch.mean(tensor4, dim=2)
        return torch.sum(tensor4 * wts.unsqueeze(0).unsqueeze(-1), dim=2)


class MultiInv_DynamicSimHydModelSix_Physical(MultiInv_DynamicSimHydModelSix):
    def __init__(self, *, gate_variant="soft", gate_strength_max=0.0, **kwargs):
        super().__init__(**kwargs)
        self.gate_variant = gate_variant
        self.gate_strength_max = gate_strength_max
        self.simhyd = DynamicSimHydModelFiveDifferentiablePhysicalFix(
            mode="normal",
            theta_is_raw=False,
            smooth=True,
            dynamic_sq=self.dynamic_sq,
            dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition,
            dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale,
            dynamic_all=self.dynamic_all,
        )
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiablePhysicalFix(
            mode="analysis",
            theta_is_raw=False,
            smooth=True,
            dynamic_sq=self.dynamic_sq,
            dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition,
            dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale,
            dynamic_all=self.dynamic_all,
        )

    def _pos(self, x):
        return torch.clamp(x, min=0.0)

    def _min(self, a, b):
        return torch.minimum(a, b)

    def _apply_zero_flow_gate(self, q_comp, x, diag_comp, theta, ngage):
        if not self.zero_flow_gate_enabled:
            ones = torch.ones_like(q_comp)
            zeros = torch.zeros_like(q_comp)
            return q_comp, ones, zeros, ones
        T, B, M, _ = q_comp.shape
        if "SMS" in diag_comp:
            soil_m = self._component_tensor_4d(diag_comp["SMS"], ngage)
        elif "soil_moisture" in diag_comp:
            soil_m = self._component_tensor_4d(diag_comp["soil_moisture"], ngage)
        else:
            soil_m = torch.ones_like(q_comp) * 100.0
        smsc = self._theta_to_smsc(theta, ngage).unsqueeze(0)
        wetness = torch.clamp(soil_m / torch.clamp(smsc, min=1e-6), 0.0, 1.0)
        p_t = x[:, :, 0:1].unsqueeze(2).repeat(1, 1, M, 1)
        sin_t = x[:, :, 3:4].unsqueeze(2).repeat(1, 1, M, 1) if x.shape[-1] >= 5 else torch.zeros_like(q_comp)
        cos_t = x[:, :, 4:5].unsqueeze(2).repeat(1, 1, M, 1) if x.shape[-1] >= 5 else torch.ones_like(q_comp)
        gate_in = torch.cat([p_t / 20.0, wetness, q_comp / 20.0, sin_t, cos_t], dim=-1)
        gate_in_flat = gate_in.view(T * B * M, 5)
        logits_all = self.zeroFlowGate(gate_in_flat)
        comp_idx = torch.arange(M, device=q_comp.device).view(1, 1, M, 1).repeat(T, B, 1, 1).view(T * B * M, 1)
        logits = logits_all.gather(1, comp_idx)
        p_flow = torch.sigmoid(logits).view(T, B, M, 1)
        if self.gate_variant == "explicit":
            gate_loss_frac = 1.0 - p_flow
            keep_fraction = p_flow
        else:
            gate_loss_frac = self.gate_strength_max * (1.0 - p_flow)
            keep_fraction = 1.0 - gate_loss_frac
        gate_loss_raw = q_comp * gate_loss_frac
        gate_loss = self._min(gate_loss_raw, q_comp)
        q_after = self._pos(q_comp - gate_loss)
        return q_after, p_flow, gate_loss, keep_fraction


class MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimple(
    MultiInv_DynamicSimHydModelSix_Physical
):
    def __init__(
        self,
        *,
        ninv,
        nmul=4,
        nattr=35,
        hiddeninv=256,
        drinv=0.5,
        inittime=0,
        routOpt=True,
        comprout=False,
        compwts=True,
        lgdyn=True,
        lgdynweight=0.6,
        dynamic_sq=True,
        dynamic_etgam=True,
        dynamic_partition=True,
        dynamic_cfmax_snow=True,
        dynamic_routing_scale=False,
        dynamic_all=False,
        reg_amp_w=1e-3,
        reg_smooth_w=1e-3,
        reg_part_w=1e-3,
        component_routing=True,
    ):
        super().__init__(
            ninv=ninv,
            nmul=nmul,
            nattr=nattr,
            hiddeninv=hiddeninv,
            drinv=drinv,
            inittime=inittime,
            routOpt=routOpt,
            comprout=comprout,
            compwts=compwts,
            lgdyn=lgdyn,
            lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq,
            dynamic_etgam=dynamic_etgam,
            dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow,
            dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all,
            reg_amp_w=reg_amp_w,
            reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w,
            component_routing=component_routing,
            dry_channel_loss=False,
            zero_flow_gate=False,
            channel_loss_max=0.0,
            gate_variant="soft",
            gate_strength_max=0.0,
        )
        self.nfea = 18
        self.nstaticpm = self.nfea * nmul
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)
        comp_bias = torch.linspace(-0.15, 0.15, nmul).view(1, 1, nmul).repeat(1, self.nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.simhyd = DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimple(
            mode="normal",
            theta_is_raw=False,
            smooth=True,
            dynamic_sq=self.dynamic_sq,
            dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition,
            dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale,
            dynamic_all=self.dynamic_all,
        )
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimple(
            mode="analysis",
            theta_is_raw=False,
            smooth=True,
            dynamic_sq=self.dynamic_sq,
            dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition,
            dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale,
            dynamic_all=self.dynamic_all,
        )

    def _theta_to_smsc(self, theta, ngage):
        theta4 = theta.view(ngage, self.nmul, self.nfea)
        return 10.0 + theta4[:, :, 15:16] * (1500.0 - 10.0)

    def _apply_channel_loss(self, q_comp, diag_comp, theta, basin_attr, ngage):
        zeros = torch.zeros_like(q_comp)
        return q_comp, zeros, zeros

    def forward(self, x, z, doDropMC=False, return_diagnostics=False, return_component_diagnostics=False):
        nt_x = x.shape[0]
        ngage = z.shape[1]
        basin_attr = z[-1, :, -self.nattr :]
        snow_frac_raw = torch.clamp(z[-1, :, -self.nattr - 1 : -self.nattr], 0.0, 1.0) if z.shape[2] > self.nattr else None
        staticFeat = self.staticFeat(basin_attr)
        staticParams0 = self.staticOut(staticFeat)
        cursor = 0
        static0 = staticParams0[:, cursor : cursor + self.nstaticpm].view(ngage, self.nfea, self.nmul)
        static0 = static0 + self.compStaticBias
        snowpara = torch.sigmoid(static0)
        cursor += self.nstaticpm
        routpara0 = staticParams0[:, cursor : cursor + self.nroutpm]
        routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2) if self.component_routing else torch.sigmoid(routpara0)
        cursor += self.nroutpm
        if self.nwtspm == 0:
            wts = None
        else:
            wtspara = staticParams0[:, cursor : cursor + self.nwtspm] + self.compWeightBias
            wts = F.softmax(wtspara, dim=-1)
        self._last_wts = wts
        lg_dyn = None
        if self.lgdyn:
            lg_dyn_seq = self.lstmdyn(z)
            lg_attr_bias = self.lgAttr(basin_attr).unsqueeze(0).repeat(lg_dyn_seq.shape[0], 1, 1)
            lg_dyn = torch.sigmoid(lg_dyn_seq + lg_attr_bias)
        x_rep = x.unsqueeze(2).repeat(1, 1, self.nmul, 1).view(nt_x, ngage * self.nmul, x.shape[2])
        x_bt = x_rep.permute(1, 0, 2)
        theta = snowpara.permute(0, 2, 1).contiguous().view(ngage * self.nmul, self.nfea)
        lg_bt = None if lg_dyn is None else lg_dyn.permute(1, 2, 0).contiguous().view(ngage * self.nmul, lg_dyn.shape[0], 1)
        lg_bt_main = lg_bt[:, self.inittime :, :] if lg_bt is not None and self.inittime > 0 else lg_bt
        snow_frac_rep = None if snow_frac_raw is None else snow_frac_raw.unsqueeze(1).repeat(1, self.nmul, 1).view(ngage * self.nmul, 1)
        need_process_diag = return_diagnostics or return_component_diagnostics
        if self.inittime > 0:
            warm_inputs = x_bt[:, : self.inittime, :]
            main_inputs = x_bt[:, self.inittime :, :]
            x_use = x[self.inittime :, :, :]
            _, warm_state = self.simhyd_analysis(
                warm_inputs,
                theta,
                lg_dyn_seq=None,
                lg_dyn_weight=self.lgdynweight,
                snow_frac_raw=snow_frac_rep,
                return_final_state=True,
            )
            if need_process_diag:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    main_inputs,
                    theta,
                    initial_state=warm_state,
                    lg_dyn_seq=lg_bt_main,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_diagnostics=True,
                    return_regularization=True,
                )
            else:
                q_seq, reg_terms = self.simhyd(
                    main_inputs,
                    theta,
                    initial_state=warm_state,
                    lg_dyn_seq=lg_bt_main,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_regularization=True,
                )
                diag_comp = None
        else:
            x_use = x
            if need_process_diag:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    x_bt,
                    theta,
                    lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_diagnostics=True,
                    return_regularization=True,
                )
            else:
                q_seq, reg_terms = self.simhyd(
                    x_bt,
                    theta,
                    lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_regularization=True,
                )
                diag_comp = None
        reg_total = (
            self.reg_amp_w * reg_terms["dynamic_amplitude_loss"]
            + self.reg_smooth_w * reg_terms["dynamic_smoothness_loss"]
            + self.reg_part_w * reg_terms["partition_entropy_loss"]
        )
        self._last_aux_terms = reg_terms
        self._last_aux_loss = reg_total + self.reg_amp_w * reg_terms.get("storage_drift_loss", 0.0)
        q_comp_raw = q_seq.view(ngage, self.nmul, q_seq.shape[1], 1).permute(2, 0, 1, 3)
        q_comp = torch.clamp(q_comp_raw, min=0.0)
        q_mix_before_routing = self._mix_or_mean(q_comp, wts)
        route_mult_seq = None
        if self.routOpt and self.component_routing:
            q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
            q_routed = self._route_q(q_for_routing, routpara).view(q_comp.shape[0], ngage, self.nmul, 1)
            out = self._mix_or_mean(q_routed, wts)
        elif self.routOpt:
            out = self._route_q(q_mix_before_routing, routpara)
            q_routed = q_comp
        else:
            out = q_mix_before_routing
            q_routed = q_comp
        out = torch.clamp(out, min=0.0)
        if not return_diagnostics and not return_component_diagnostics:
            return out
        diag_out = {}
        if diag_comp is not None:
            for name, tensor_comp in diag_comp.items():
                diag_out[name] = self._mix_component_tensor(tensor_comp, ngage)
        diag_out["total_discharge"] = out
        diag_out["q_mix_before_routing"] = q_mix_before_routing
        diag_out["component_discharge_raw"] = self._mix_or_mean(q_comp_raw, wts)
        if route_mult_seq is not None:
            diag_out["route_b_t_multiplier"] = route_mult_seq
        if return_component_diagnostics:
            if diag_comp is not None:
                for name, tensor_comp in diag_comp.items():
                    diag_out[name + "_components"] = self._component_tensor_4d(tensor_comp, ngage).squeeze(-1)
            diag_out["q_raw_process_components"] = q_comp_raw.squeeze(-1)
            diag_out["q_routed_components"] = q_routed.squeeze(-1)
            diag_out["component_weights"] = wts
            if self.component_routing:
                routpara_view = torch.sigmoid(routpara0).view(ngage, self.nmul, 2)
                diag_out["route_a_components"] = routpara_view[:, :, 0] * 2.9
                diag_out["route_b_components"] = routpara_view[:, :, 1] * 6.5
        return out, diag_out
