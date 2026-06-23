import warnings
from typing import Dict, Optional, Sequence, Tuple

import pyro
import pyro.distributions as dist
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoIAFNormal, AutoLowRankMultivariateNormal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import constraints

from . import rnn


HBV_BASE_PARAM_NAMES = [
    "BETA",
    "FC",
    "K0",
    "K1",
    "K2",
    "LP",
    "PERC",
    "UZL",
    "TT",
    "CFMAX",
    "CFR",
    "CWH",
]

HBV_BASE_PARAM_LOWER = [
    1.0,
    50.0,
    0.05,
    0.01,
    0.001,
    0.2,
    0.0,
    0.0,
    -2.5,
    0.5,
    0.0,
    0.0,
]

HBV_BASE_PARAM_UPPER = [
    6.0,
    1000.0,
    0.9,
    0.5,
    0.2,
    1.0,
    10.0,
    100.0,
    2.5,
    10.0,
    0.1,
    0.2,
]


def _repeat_bounds_per_component(values: Sequence[float], nmul: int) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).repeat_interleave(nmul)


class VIRegionalizedHBV(nn.Module):
    """
    Variational regionalized HBV wrapper.

    This keeps the existing Chen-style regionalization pattern
    (sequence-conditioned inversion network -> sigmoid-bounded HBV parameters
    -> differentiable HBV forward model) while adding uncertainty only to the
    basin-specific HBV parameter logits.
    """

    def __init__(
        self,
        *,
        ninv: int,
        nfea: int = 12,
        nmul: int = 16,
        hiddeninv: int = 256,
        drinv: float = 0.0,
        routOpt: bool = True,
        comprout: bool = False,
        compwts: bool = False,
        pcorr: Optional[Sequence[float]] = None,
        spinup_days: int = 365,
        guide_type: str = "lowrank",
        lowrank_rank: Optional[int] = None,
        learn_sigma: bool = False,
        sigma_scale_divisor: float = 10.0,
        sigma_floor: float = 0.01,
        iaf_hidden_dim: int = 64,
        iaf_num_transforms: int = 2,
    ):
        super().__init__()
        if nfea != len(HBV_BASE_PARAM_NAMES):
            raise ValueError(
                f"VIRegionalizedHBV currently expects nfea={len(HBV_BASE_PARAM_NAMES)} "
                f"to match the existing HBV parameterization, got {nfea}."
            )

        self.ninv = ninv
        self.nfea = nfea
        self.nmul = nmul
        self.hiddeninv = hiddeninv
        self.drinv = drinv
        self.routOpt = routOpt
        self.comprout = comprout
        self.compwts = compwts
        self.pcorr = pcorr
        self.spinup_days = spinup_days
        self.guide_type = guide_type.lower()
        self.learn_sigma = learn_sigma
        self.sigma_scale_divisor = sigma_scale_divisor
        self.sigma_floor = sigma_floor
        self.iaf_hidden_dim = iaf_hidden_dim
        self.iaf_num_transforms = iaf_num_transforms

        self.nhbvpm = nfea * nmul
        self.nroutpm = nmul * 2 if comprout else 2
        self.nwtspm = nmul if compwts else 0
        self.ntp = self.nhbvpm + self.nroutpm + self.nwtspm + (1 if pcorr is not None else 0)

        self.regionalizer = rnn.SafeLstmModel(
            nx=ninv,
            ny=self.ntp,
            hiddenSize=hiddeninv,
            dr=drinv,
        )
        self.hbv = rnn.HBVMul()

        self.register_buffer(
            "hbv_lower_flat",
            _repeat_bounds_per_component(HBV_BASE_PARAM_LOWER, nmul),
        )
        self.register_buffer(
            "hbv_upper_flat",
            _repeat_bounds_per_component(HBV_BASE_PARAM_UPPER, nmul),
        )
        self.register_buffer(
            "tau_init",
            torch.full((self.nhbvpm,), 0.15, dtype=torch.float32),
        )

        self.lowrank_rank = lowrank_rank

    @property
    def hbv_param_names(self) -> Sequence[str]:
        return HBV_BASE_PARAM_NAMES

    def _expand_attrs(self, attrs: torch.Tensor, nt: int) -> torch.Tensor:
        return attrs.unsqueeze(1).expand(-1, nt, -1)

    def build_regionalizer_input(self, z_inputs: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        if z_inputs.dim() != 3:
            raise ValueError(f"Expected z_inputs [B, T, F], got shape {tuple(z_inputs.shape)}")
        if attrs.dim() != 2:
            raise ValueError(f"Expected attrs [B, A], got shape {tuple(attrs.shape)}")
        if z_inputs.shape[0] != attrs.shape[0]:
            raise ValueError("z_inputs and attrs must agree on basin dimension")
        nt = z_inputs.shape[1]
        attr_seq = self._expand_attrs(attrs, nt)
        reg_in = torch.cat([z_inputs, attr_seq], dim=-1)
        return reg_in.transpose(0, 1).contiguous()

    def _split_regionalizer_output(
        self, params0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        hbv_mu = params0[:, : self.nhbvpm]
        cursor = self.nhbvpm

        routpara0 = params0[:, cursor : cursor + self.nroutpm]
        cursor += self.nroutpm
        if self.comprout:
            routpara = torch.sigmoid(routpara0).view(params0.shape[0] * self.nmul, 2)
        else:
            routpara = torch.sigmoid(routpara0)

        if self.nwtspm > 0:
            wtspara = params0[:, cursor : cursor + self.nwtspm]
            cursor += self.nwtspm
            weights = F.softmax(wtspara, dim=-1)
        else:
            weights = None

        if self.pcorr is not None:
            corrpara0 = params0[:, cursor : cursor + 1]
            corrpara = torch.sigmoid(corrpara0)
        else:
            corrpara = None

        return hbv_mu, routpara, weights, corrpara

    def _theta_from_logits(self, z_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_unit_flat = torch.sigmoid(z_latent)
        theta_phys_flat = self.hbv_lower_flat + (
            self.hbv_upper_flat - self.hbv_lower_flat
        ) * theta_unit_flat
        theta_unit = theta_unit_flat.view(z_latent.shape[0], self.nfea, self.nmul)
        theta_phys = theta_phys_flat.view(z_latent.shape[0], self.nfea, self.nmul)
        return theta_unit, theta_phys

    def _run_hbv(
        self,
        forcings: torch.Tensor,
        theta_unit: torch.Tensor,
        routing: torch.Tensor,
        weights: Optional[torch.Tensor],
        corrpara: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_time_major = forcings.transpose(0, 1).contiguous()
        q_all = self.hbv(
            x_time_major,
            parameters=theta_unit,
            mu=self.nmul,
            muwts=weights,
            rtwts=routing,
            bufftime=0,
            routOpt=self.routOpt,
            comprout=self.comprout,
            corrwts=corrpara,
            pcorr=self.pcorr,
        )
        qsim = q_all[:, :, 0].transpose(0, 1).contiguous()
        return qsim, q_all

    def _effective_mask(
        self, qobs: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if mask is None:
            eff_mask = torch.isfinite(qobs)
        else:
            eff_mask = mask.bool() & torch.isfinite(qobs)
        if self.spinup_days > 0:
            eff_mask = eff_mask.clone()
            eff_mask[:, : self.spinup_days] = False
        return eff_mask

    def _fixed_sigma(self, qobs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        safe_obs = torch.where(mask, qobs, torch.zeros_like(qobs))
        valid_counts = mask.sum(dim=1).clamp(min=1)
        mean_flow = safe_obs.sum(dim=1) / valid_counts
        sigma = torch.clamp(mean_flow.abs() / self.sigma_scale_divisor, min=self.sigma_floor)
        return sigma

    def model(
        self,
        forcings: torch.Tensor,
        z_inputs: torch.Tensor,
        attrs: torch.Tensor,
        qobs: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pyro.module("regionalizer", self.regionalizer)

        reg_inputs = self.build_regionalizer_input(z_inputs, attrs)
        param_seq = self.regionalizer(reg_inputs)
        params0 = param_seq[-1, :, :]
        mu_z, routpara, weights, corrpara = self._split_regionalizer_output(params0)

        tau_z = pyro.param("tau_z", self.tau_init.clone(), constraint=constraints.positive)
        eps = pyro.sample(
            "eps_basin_param",
            dist.Normal(
                torch.zeros_like(mu_z),
                torch.ones_like(mu_z),
            ).to_event(2),
        )
        z_latent = mu_z + tau_z.unsqueeze(0) * eps
        theta_unit, theta_phys = self._theta_from_logits(z_latent)
        qsim, q_all = self._run_hbv(forcings, theta_unit, routpara, weights, corrpara)

        pyro.deterministic("mu_z", mu_z)
        pyro.deterministic("theta", theta_phys)
        pyro.deterministic("theta_unit", theta_unit)
        pyro.deterministic("qsim", qsim)

        sigma_basin = None
        if qobs is not None:
            obs_mask = self._effective_mask(qobs, mask)
            if self.learn_sigma:
                sigma_init = self._fixed_sigma(qobs, obs_mask).detach()
                sigma_basin = pyro.param(
                    "sigma_basin",
                    sigma_init,
                    constraint=constraints.positive,
                )
            else:
                sigma_basin = self._fixed_sigma(qobs, obs_mask)

            pyro.deterministic("sigma_basin", sigma_basin)

            flat_mask = obs_mask & torch.isfinite(qsim)
            if not torch.any(flat_mask):
                raise RuntimeError("No valid observations remain after masking and spinup.")
            pyro.sample(
                "obs",
                dist.Normal(
                    qsim[flat_mask],
                    sigma_basin.unsqueeze(1).expand_as(qsim)[flat_mask],
                ).to_event(1),
                obs=qobs[flat_mask],
            )

        return {
            "params0": params0,
            "mu_z": mu_z,
            "theta_unit": theta_unit,
            "theta": theta_phys,
            "qsim": qsim,
            "q_all": q_all,
            "sigma_basin": sigma_basin,
        }

    def forward(
        self,
        forcings: torch.Tensor,
        z_inputs: torch.Tensor,
        attrs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        reg_inputs = self.build_regionalizer_input(z_inputs, attrs)
        param_seq = self.regionalizer(reg_inputs)
        params0 = param_seq[-1, :, :]
        mu_z, routpara, weights, corrpara = self._split_regionalizer_output(params0)
        theta_unit, theta_phys = self._theta_from_logits(mu_z)
        qsim, q_all = self._run_hbv(forcings, theta_unit, routpara, weights, corrpara)
        return {
            "params0": params0,
            "mu_z": mu_z,
            "theta_unit": theta_unit,
            "theta": theta_phys,
            "qsim": qsim,
            "q_all": q_all,
        }

    def make_guide(self, latent_shape: Optional[Tuple[int, int]] = None):
        if self.guide_type == "lowrank":
            latent_dim = None
            if latent_shape is not None:
                latent_dim = latent_shape[0] * latent_shape[1]
            rank = self.lowrank_rank
            if rank is None:
                if latent_dim is None:
                    rank = 10
                else:
                    rank = min(10, max(1, latent_dim // 4))
            return AutoLowRankMultivariateNormal(self.model, rank=rank)

        if self.guide_type == "iaf":
            warnings.warn(
                "IAF is experimental; use only after lowrank VI passes tests.",
                RuntimeWarning,
            )
            return AutoIAFNormal(
                self.model,
                hidden_dim=self.iaf_hidden_dim,
                num_transforms=self.iaf_num_transforms,
            )

        raise ValueError(f"Unsupported guide_type: {self.guide_type}")

    @torch.no_grad()
    def posterior_predictive(
        self,
        guide,
        forcings: torch.Tensor,
        z_inputs: torch.Tensor,
        attrs: torch.Tensor,
        qobs: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        num_samples: int = 100,
    ) -> Dict[str, torch.Tensor]:
        predictive = Predictive(
            self.model,
            guide=guide,
            num_samples=num_samples,
            return_sites=("qsim", "theta", "theta_unit", "sigma_basin", "mu_z"),
        )
        return predictive(forcings, z_inputs, attrs, qobs=qobs, mask=mask)
