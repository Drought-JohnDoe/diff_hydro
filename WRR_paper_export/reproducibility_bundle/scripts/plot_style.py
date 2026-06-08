from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, TwoSlopeNorm


BASE_STYLE = {
    "figure.figsize": (11.5, 6.8),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}


MAP_STYLE = {
    "NSE": {
        "cmap": "turbo",
        "vmin": 0.0,
        "vmax": 1.0,
        "ticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "label": "NSE",
        "clip_for_viz": True,
        "supplementary_unclipped": True,
        "plot_mode": "point_bins",
        "boundaries": [0.0, 0.2, 0.4, 0.5, 0.65, 0.8, 1.0],
        "bin_labels": ["0-0.2", "0.2-0.4", "0.4-0.5", "0.5-0.65", "0.65-0.8", "0.8-1.0"],
        "colors": ["#d73027", "#fc8d14", "#ffd23f", "#91cf60", "#22a7a7", "#2c3e99"],
    },
    "KGE": {
        "cmap": "turbo",
        "vmin": 0.0,
        "vmax": 1.0,
        "ticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "label": "KGE",
        "clip_for_viz": True,
        "supplementary_unclipped": True,
        "plot_mode": "point_bins",
        "boundaries": [0.0, 0.2, 0.4, 0.5, 0.65, 0.8, 1.0],
        "bin_labels": ["0-0.2", "0.2-0.4", "0.4-0.5", "0.5-0.65", "0.65-0.8", "0.8-1.0"],
        "colors": ["#d73027", "#fc8d14", "#ffd23f", "#91cf60", "#22a7a7", "#2c3e99"],
    },
    "low_flow_NSE": {
        "cmap": "Spectral_r",
        "center": 0.0,
        "ticks": [-100, -50, -10, 0, 0.25, 0.5, 0.75],
        "label": "Low-flow NSE",
    },
    "high_flow_NSE": {
        "cmap": "turbo",
        "vmin": 0.0,
        "vmax": 1.0,
        "ticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "label": "High-flow NSE",
        "clip_for_viz": True,
    },
    "FHV": {
        "cmap": "RdBu_r",
        "center": 0.0,
        "label": "FHV",
    },
    "FLV": {
        "cmap": "RdBu_r",
        "center": 0.0,
        "label": "FLV",
    },
    "bias_daily_aux": {
        "cmap": "BrBG",
        "center": 0.0,
        "label": "Q bias (mm d$^{-1}$)",
    },
    "mean_abs_daily_wb_residual_mm_day": {
        "cmap": "magma",
        "label": "Mean abs WB residual (mm d$^{-1}$)",
    },
    "cumulative_relative_wb_error": {
        "cmap": "plasma",
        "label": "Cumulative relative WB error",
    },
    "theta_cap_mean": {
        "cmap": "plasma",
        "label": "Active root-zone capacity (mm)",
    },
    "theta_wetpoint_weighted_ep60": {
        "cmap": "cividis",
        "label": "theta_wetpoint (-)",
    },
    "K_weighted": {
        "cmap": "viridis",
        "label": "Groundwater recession K (-)",
    },
    "component_entropy_ep60": {
        "cmap": "magma",
        "label": "Component entropy",
    },
    "aSrz_capacity_mm": {
        "cmap": "plasma",
        "label": "aSrz capacity (mm)",
    },
    "mean_aSrz_mm": {
        "cmap": "viridis",
        "label": "Mean aSrz (mm)",
    },
    "model_mean_ET_mm_month": {
        "cmap": "viridis",
        "label": "Mean ET (mm month$^{-1}$)",
    },
    "model_ET_over_P": {
        "cmap": "cividis",
        "label": "ET/P (-)",
    },
    "ET_bias_FLUXCOM": {
        "cmap": "RdBu_r",
        "center": 0.0,
        "label": "ET bias vs FLUXCOM (mm month$^{-1}$)",
    },
    "R2_FLUXCOM": {
        "cmap": "turbo",
        "vmin": 0.0,
        "vmax": 1.0,
        "ticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "label": "ET R$^2$ vs FLUXCOM",
        "clip_for_viz": True,
    },
    "map_ET_product_uncertainty": {
        "cmap": "magma",
        "label": "ET product spread (mm yr$^{-1}$)",
    },
    "R2_JPL": {
        "cmap": "turbo",
        "vmin": 0.0,
        "vmax": 1.0,
        "ticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "label": "TWSA R$^2$ vs JPL",
        "clip_for_viz": True,
    },
    "corr_regional": {
        "cmap": "viridis",
        "vmin": 0.0,
        "vmax": 1.0,
        "ticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "label": "Regional TWSA correlation",
        "clip_for_viz": True,
    },
    "model_amplitude": {
        "cmap": "plasma",
        "label": "Model TWSA amplitude (mm)",
    },
    "grace_amplitude": {
        "cmap": "plasma",
        "label": "JPL TWSA amplitude (mm)",
    },
    "phase_difference": {
        "cmap": "coolwarm",
        "center": 0.0,
        "label": "Phase difference (months)",
    },
    "amplitude_ratio": {
        "cmap": "cividis",
        "label": "Amplitude ratio (-)",
    },
}


SUPPLEMENTARY_CAPTIONS = {
    "NSE": "Values below 0 are clipped to 0 for visualization; original values are retained in the basin-level metric table.",
    "KGE": "Values below 0 are clipped to 0 for visualization; original values are retained in the basin-level metric table.",
}


def apply_matplotlib_style() -> None:
    plt.rcParams.update(BASE_STYLE)


def infer_style(metric_col: str, units: str) -> dict:
    style = dict(MAP_STYLE.get(metric_col, {}))
    if "label" not in style:
        style["label"] = units
    if "cmap" not in style:
        style["cmap"] = "viridis"
    return style


def resolve_cmap(style: dict):
    if "colors" in style:
        return ListedColormap(style["colors"])
    return plt.get_cmap(style["cmap"])


def compute_norm(values: np.ndarray, style: dict):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0), None, None
    if style.get("clip_for_viz"):
        clipped = np.clip(values, style.get("vmin", np.nanmin(finite)), style.get("vmax", np.nanmax(finite)))
    else:
        clipped = values.copy()
    boundaries = style.get("boundaries")
    norm = None
    extend = "neither"
    if boundaries is not None:
        cmap_obj = resolve_cmap(style)
        ncolors = cmap_obj.N if hasattr(cmap_obj, "N") else len(boundaries) - 1
        norm = BoundaryNorm(boundaries, ncolors=ncolors, clip=False)
        if np.nanmin(clipped) < min(boundaries) and np.nanmax(clipped) > max(boundaries):
            extend = "both"
        elif np.nanmin(clipped) < min(boundaries):
            extend = "min"
        elif np.nanmax(clipped) > max(boundaries):
            extend = "max"
        return norm, clipped, extend
    if "center" in style:
        vcenter = style["center"]
        vmax = style.get("vmax", float(np.nanpercentile(np.abs(finite - vcenter), 97.5) + vcenter))
        vmin = style.get("vmin", float(vcenter - np.nanpercentile(np.abs(finite - vcenter), 97.5)))
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        if np.nanmin(clipped) < vmin and np.nanmax(clipped) > vmax:
            extend = "both"
        elif np.nanmin(clipped) < vmin:
            extend = "min"
        elif np.nanmax(clipped) > vmax:
            extend = "max"
        return norm, clipped, extend
    vmin = style.get("vmin", float(np.nanpercentile(finite, 2)))
    vmax = style.get("vmax", float(np.nanpercentile(finite, 98)))
    norm = Normalize(vmin=vmin, vmax=vmax)
    if np.nanmin(clipped) < vmin and np.nanmax(clipped) > vmax:
        extend = "both"
    elif np.nanmin(clipped) < vmin:
        extend = "min"
    elif np.nanmax(clipped) > vmax:
        extend = "max"
    return norm, clipped, extend


def default_ticks(style: dict, values: np.ndarray) -> list[float]:
    if "ticks" in style:
        return style["ticks"]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [0.0, 0.5, 1.0]
    lo = style.get("vmin", float(np.nanpercentile(finite, 2)))
    hi = style.get("vmax", float(np.nanpercentile(finite, 98)))
    return list(np.linspace(lo, hi, 6))


def caption_note(metric_col: str) -> str | None:
    return SUPPLEMENTARY_CAPTIONS.get(metric_col)
