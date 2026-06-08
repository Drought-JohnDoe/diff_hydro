from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from plot_style import apply_matplotlib_style, caption_note, compute_norm, default_ticks, infer_style, resolve_cmap


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as fp:
        return yaml.safe_load(fp)


def ensure_outputs(cfg: dict) -> None:
    for key in ["diagnostics_dir", "tables_dir", "figures_dir", "maps_dir", "manuscript_dir", "logs_dir"]:
        Path(cfg["outputs"][key]).mkdir(parents=True, exist_ok=True)
    Path(cfg["outputs"]["maps_dir"], "geodata").mkdir(parents=True, exist_ok=True)
    Path(cfg["outputs"]["figures_dir"], "hydrographs").mkdir(parents=True, exist_ok=True)


def load_camels_attributes(cfg: dict) -> pd.DataFrame:
    attrs_dir = Path(cfg["basins"]["attrs_dir"])
    files = [
        "camels_clim.txt",
        "camels_geol.txt",
        "camels_hydro.txt",
        "camels_name.txt",
        "camels_soil.txt",
        "camels_topo.txt",
        "camels_vege.txt",
    ]
    tables = []
    for name in files:
        p = attrs_dir / name
        df = pd.read_csv(p, sep=";")
        df = df.rename(columns={"gauge_id": "basin_id"})
        tables.append(df)
    out = tables[0]
    for df in tables[1:]:
        out = out.merge(df, on="basin_id", how="outer")
    out["basin_id"] = out["basin_id"].astype(int)
    return out


def load_basin_geometries(cfg: dict) -> gpd.GeoDataFrame:
    shp = Path(cfg["basins"]["shapefile"])
    basin_id_col = cfg["basins"]["basin_id_column"]
    gdf = gpd.read_file(shp)
    gdf[basin_id_col] = gdf[basin_id_col].astype(str).str.replace(r"\.0$", "", regex=True)
    gdf["basin_id"] = gdf[basin_id_col].astype(int)
    gdf = gdf[["basin_id", "geometry"]].dissolve(by="basin_id", as_index=False)
    return gdf


def merge_metric_geometries(cfg: dict, metric_df: pd.DataFrame, basin_id_col: str = "basin_id") -> gpd.GeoDataFrame:
    geom = load_basin_geometries(cfg)
    metric_df = metric_df.copy()
    metric_df[basin_id_col] = metric_df[basin_id_col].astype(int)
    gdf = geom.merge(metric_df, left_on="basin_id", right_on=basin_id_col, how="left")
    return gdf


def save_geodata(gdf: gpd.GeoDataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG")


def save_basin_map(
    cfg: dict,
    metric_df: pd.DataFrame,
    metric_col: str,
    output_path: str | Path,
    title: str,
    units: str,
    cmap: str | None = None,
    vmin=None,
    vmax=None,
) -> None:
    apply_matplotlib_style()
    gdf = merge_metric_geometries(cfg, metric_df)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    style = infer_style(metric_col, units)
    if cmap is not None:
        style["cmap"] = cmap
    if vmin is not None:
        style["vmin"] = vmin
    if vmax is not None:
        style["vmax"] = vmax
    values = gdf[metric_col].to_numpy(dtype=float)
    norm, plot_values, extend = compute_norm(values, style)
    gdf = gdf.copy()
    gdf["_plot_value"] = plot_values
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    cmap_obj = resolve_cmap(style)
    if style.get("plot_mode") == "point_bins":
        outline = gdf.copy()
        outline.plot(ax=ax, color="white", edgecolor="#bfbfbf", linewidth=0.25)
        pts = gdf.representative_point()
        x = pts.x.to_numpy()
        y = pts.y.to_numpy()
        vals = gdf["_plot_value"].to_numpy(dtype=float)
        scat = ax.scatter(
            x,
            y,
            c=vals,
            cmap=cmap_obj,
            norm=norm,
            s=42,
            linewidths=0.45,
            edgecolors="#333333",
            alpha=0.98,
        )
    else:
        gdf.plot(
            column="_plot_value",
            ax=ax,
            cmap=cmap_obj,
            linewidth=0.18,
            edgecolor="#222222",
            legend=False,
            missing_kwds={"color": cfg["plotting"]["missing_color"], "label": "Missing"},
            norm=norm,
        )
    ax.set_title(title, pad=10)
    ax.set_axis_off()
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    if style.get("plot_mode") == "point_bins" and style.get("bin_labels"):
        labels = style["bin_labels"]
        boundaries = style["boundaries"]
        centers = [(boundaries[i] + boundaries[i + 1]) / 2.0 for i in range(len(boundaries) - 1)]
        handles = []
        for color, label in zip(style["colors"], labels):
            h = plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color,
                markeredgecolor="#333333",
                markeredgewidth=0.45,
                markersize=9,
                label=label,
            )
            handles.append(h)
        ax.legend(
            handles=handles,
            title=style["label"],
            loc="upper right",
            frameon=False,
            ncol=3,
            fontsize=10,
            title_fontsize=11,
        )
    else:
        ticks = default_ticks(style, values)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, shrink=0.92, extend=extend)
        cbar.set_label(style["label"], labelpad=10)
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize=10)
    fig.tight_layout()
    formats = cfg["plotting"]["figure_format_main"]
    if "svg" not in formats:
        formats = list(formats) + ["svg"]
    for ext in formats:
        fig.savefig(output_path.with_suffix(f".{ext}"), dpi=cfg["plotting"]["dpi"], bbox_inches="tight")
    plt.close(fig)

    if style.get("supplementary_unclipped"):
        supp_dir = Path(cfg["outputs"]["maps_dir"]) / "supplementary"
        supp_dir.mkdir(parents=True, exist_ok=True)
        supp_fig, supp_ax = plt.subplots(figsize=(11.5, 6.8))
        finite = values[np.isfinite(values)]
        if finite.size:
            lo = float(np.nanpercentile(finite, 2))
            hi = float(np.nanpercentile(finite, 98))
        else:
            lo, hi = 0.0, 1.0
        supp_norm, _, supp_extend = compute_norm(values, {"cmap": style["cmap"], "vmin": lo, "vmax": hi})
        gdf.plot(
            column=metric_col,
            ax=supp_ax,
            cmap=cmap_obj,
            linewidth=0.18,
            edgecolor="#222222",
            legend=False,
            missing_kwds={"color": cfg["plotting"]["missing_color"], "label": "Missing"},
            norm=supp_norm,
        )
        supp_ax.set_title(f"{title} (unclipped supplementary)", pad=10)
        supp_ax.set_axis_off()
        sm2 = plt.cm.ScalarMappable(norm=supp_norm, cmap=cmap_obj)
        sm2.set_array([])
        cbar2 = supp_fig.colorbar(sm2, ax=supp_ax, fraction=0.046, pad=0.02, shrink=0.92, extend=supp_extend)
        cbar2.set_label(style["label"], labelpad=10)
        cbar2.ax.tick_params(labelsize=10)
        supp_fig.tight_layout()
        for ext in ["png", "pdf", "svg"]:
            supp_fig.savefig(supp_dir / f"{output_path.stem}_unclipped.{ext}", dpi=cfg["plotting"]["dpi"], bbox_inches="tight")
        plt.close(supp_fig)

    note = caption_note(metric_col)
    if note:
        note_path = output_path.with_suffix(".caption.txt")
        note_path.write_text(note + "\n")


def hydroclass_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["aridity_class"] = pd.cut(
        out["aridity"],
        bins=[-np.inf, 0.8, 1.2, np.inf],
        labels=["humid", "transitional", "arid"],
    ).astype(str)
    out["snow_class"] = np.where(out["frac_snow"] >= 0.35, "snow_dominated", "non_snow")
    out["forest_class"] = np.where(out["frac_forest"] >= 0.5, "forested", "non_forested")
    out["baseflow_class"] = np.where(out["baseflow_index"] >= out["baseflow_index"].median(), "high_bfi", "low_bfi")
    out["permeability_class"] = np.where(
        out["geol_permeability"] >= out["geol_permeability"].median(), "high_perm", "low_perm"
    )
    out["area_class"] = pd.qcut(out["area_gages2"], 4, labels=["small", "mid_small", "mid_large", "large"])
    return out


def summarize_by_group(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    agg = {c: ["median", "mean"] for c in metric_cols}
    out = df.groupby(group_cols, dropna=False).agg(agg)
    out.columns = ["_".join([a, b]) for a, b in out.columns]
    out = out.reset_index()
    out["n_basins"] = df.groupby(group_cols, dropna=False).size().values
    return out


def entropy_from_weights(weight_df: pd.DataFrame, weight_cols: list[str]) -> pd.Series:
    arr = weight_df[weight_cols].to_numpy(dtype=float)
    arr = np.clip(arr, 1e-12, 1.0)
    return pd.Series(-(arr * np.log(arr)).sum(axis=1), index=weight_df.index)


def monthly_from_daily(df: pd.DataFrame, value_cols: list[str], date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["year_month"] = out[date_col].dt.to_period("M").dt.to_timestamp("M")
    grouped = out.groupby(["basin_id", "year_month"], dropna=False)[value_cols].sum(min_count=1).reset_index()
    return grouped.rename(columns={"year_month": "date"})


def seasonal_correlation(obs: pd.Series, sim: pd.Series) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(obs[mask], sim[mask])[0, 1])


def compute_kge(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 3:
        return np.nan
    o = obs[mask]
    s = sim[mask]
    cc = np.corrcoef(o, s)[0, 1]
    alpha = np.std(s) / max(np.std(o), 1e-12)
    beta = np.mean(s) / max(np.mean(o), 1e-12)
    return float(1.0 - np.sqrt((cc - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
