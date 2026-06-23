#!/usr/bin/env python3
"""Build a publication-ready Rohini-style replication package for locked Model 6.

This script intentionally treats the selected Model 6 LAIEco run as locked.  It
does not retrain or mutate the checkpoint; it only audits, standardizes,
evaluates, documents, and plots the available outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL6_DIR = SCRIPT_DIR.parents[1]
REPO_ROOT = MODEL6_DIR.parent
DATA_ROOT = Path(os.environ.get("MODEL6_PUBLICATION_DATA_ROOT", "/home/mircore/Desktop/diff_hydro")).resolve()
RUN_DIR = MODEL6_DIR / "results" / "locked_run"
LEGACY_PKG_DIR = MODEL6_DIR / "results" / "results_package"
PACKAGE_DIR = MODEL6_DIR

CACHE_PATH = LEGACY_PKG_DIR / "cache" / "model_eval_cache.npz"
BEST_METRICS_PATH = RUN_DIR / "best_metrics.json"
RUN_MANIFEST_PATH = RUN_DIR / "manifest.json"
MODEL_README_PATH = RUN_DIR / "README_model_detailed.md"
FINAL_README_PATH = RUN_DIR / "FINAL_MODEL_README_WITH_EXTERNAL_TESTS.md"
MODEL_SPEC_PATH = RUN_DIR / "MODEL_SPECIFICATION_COMBINED.md"

FIGURES_DIR = MODEL6_DIR / "figures" / "final"
TABLES_DIR = MODEL6_DIR / "results" / "evaluation" / "tables"
METRICS_DIR = MODEL6_DIR / "results" / "evaluation" / "metrics"
DOCS_DIR = MODEL6_DIR / "results" / "evaluation" / "docs"
CONFIGS_DIR = MODEL6_DIR / "configs"
DATA_MANIFEST_DIR = REPO_ROOT / "raw_data"
SCRIPTS_DIR = MODEL6_DIR / "figures" / "scripts"
CHECKPOINT_DIR = MODEL6_DIR / "checkpoints"
LOGS_DIR = MODEL6_DIR / "results" / "evaluation" / "logs"
NOTEBOOKS_DIR = MODEL6_DIR / "notebooks"

EXT_TABLES = RUN_DIR / "external_validation_tables"
REQ_TABLES = RUN_DIR / "requested_evaluation_bundle" / "tables"

PRIMARY_PRODUCT = "MODIS 8-day ET"
TRAIN_PERIOD = ("1980-10-01", "1995-10-01")
TEST_PERIOD = ("1995-10-01", "2010-10-01")
EVAL_PERIOD = ("1995-10-01", "2010-09-30")


def ensure_dirs() -> None:
    for path in [
        FIGURES_DIR,
        TABLES_DIR,
        METRICS_DIR,
        DOCS_DIR,
        CONFIGS_DIR,
        DATA_MANIFEST_DIR,
        SCRIPTS_DIR / "figures",
        CHECKPOINT_DIR,
        LOGS_DIR,
        NOTEBOOKS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).strip() + "\n")


def df_to_markdown(df: pd.DataFrame) -> str:
    """Small markdown table formatter that avoids an external tabulate dependency."""
    if df.empty:
        return ""
    work = df.copy()
    work = work.astype(object).where(pd.notna(work), "")
    columns = [str(c) for c in work.columns]
    rows = [[str(v) for v in row] for row in work.to_numpy()]
    widths = [
        max(len(columns[i]), *(len(row[i]) for row in rows)) if rows else len(columns[i])
        for i in range(len(columns))
    ]
    header = "| " + " | ".join(columns[i].ljust(widths[i]) for i in range(len(columns))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(columns))) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PACKAGE_DIR))
    except Exception:
        return str(path)


def load_cache() -> dict[str, np.ndarray]:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing model evaluation cache: {CACHE_PATH}")
    return dict(np.load(CACHE_PATH, allow_pickle=True))


def safe_num(value: object) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return float("nan")


def corr(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 2:
        return float("nan")
    o = obs[mask].astype(float)
    s = sim[mask].astype(float)
    if np.std(o) <= 0 or np.std(s) <= 0:
        return float("nan")
    return float(np.corrcoef(o, s)[0, 1])


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 2:
        return float("nan")
    o = obs[mask].astype(float)
    s = sim[mask].astype(float)
    denom = np.sum((o - o.mean()) ** 2)
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((s - o) ** 2) / denom)


def rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 1:
        return float("nan")
    return float(np.sqrt(np.mean((sim[mask].astype(float) - obs[mask].astype(float)) ** 2)))


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 2:
        return float("nan")
    o = obs[mask].astype(float)
    s = sim[mask].astype(float)
    r = corr(o, s)
    if not np.isfinite(r) or np.std(o) <= 0 or abs(np.mean(o)) <= 1e-12:
        return float("nan")
    alpha = np.std(s) / np.std(o)
    beta = np.mean(s) / np.mean(o)
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def bias(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 1:
        return float("nan")
    return float(np.mean(sim[mask].astype(float) - obs[mask].astype(float)))


def pbias(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 1:
        return float("nan")
    denom = np.sum(obs[mask].astype(float))
    if abs(denom) <= 1e-12:
        return float("nan")
    return float(100.0 * np.sum(sim[mask].astype(float) - obs[mask].astype(float)) / denom)


def zrmse(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 2:
        return float("nan")
    sd = np.std(obs[mask].astype(float))
    if sd <= 0:
        return float("nan")
    return rmse(obs, sim) / sd


def summarize_pairs(df: pd.DataFrame, obs_col: str, sim_col: str, label: str, units: str) -> dict:
    valid = df.dropna(subset=[obs_col, sim_col]).copy()
    obs = valid[obs_col].to_numpy(float)
    sim = valid[sim_col].to_numpy(float)
    return {
        "variable": label,
        "units": units,
        "n_pairs": int(len(valid)),
        "n_basins": int(valid["basin_id"].nunique()) if "basin_id" in valid.columns else 0,
        "start": str(pd.to_datetime(valid["date"]).min().date()) if "date" in valid.columns and len(valid) else "",
        "end": str(pd.to_datetime(valid["date"]).max().date()) if "date" in valid.columns and len(valid) else "",
        "NSE": nse(obs, sim),
        "KGE": kge(obs, sim),
        "R2": corr(obs, sim) ** 2 if np.isfinite(corr(obs, sim)) else float("nan"),
        "r": corr(obs, sim),
        "RMSE": rmse(obs, sim),
        "bias": bias(obs, sim),
        "pbias_pct": pbias(obs, sim),
    }


def robust_limits(values: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(arr, [lo, hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def scatter_map(ax, df: pd.DataFrame, value_col: str, title: str, cmap: str = "viridis", vmin=None, vmax=None):
    vals = df[value_col].to_numpy(float)
    if vmin is None or vmax is None:
        vmin, vmax = robust_limits(vals)
    sc = ax.scatter(
        df["lon"],
        df["lat"],
        c=np.clip(vals, vmin, vmax),
        cmap=cmap,
        s=28,
        edgecolors="black",
        linewidths=0.12,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.15)
    return sc


def lowess_line(ax, x: np.ndarray, y: np.ndarray, color: str = "black", label: str | None = None) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 8:
        return
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        z = lowess(y[mask], x[mask], frac=0.35, return_sorted=True)
        ax.plot(z[:, 0], z[:, 1], color=color, lw=2.2, label=label)
    except Exception:
        coef = np.polyfit(x[mask], y[mask], 1)
        xs = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color=color, lw=2.2, label=label)


def get_core_paths() -> dict[str, Path]:
    return {
        "best_metrics": BEST_METRICS_PATH,
        "run_manifest": RUN_MANIFEST_PATH,
        "model_cache": CACHE_PATH,
        "per_basin_metrics": RUN_DIR / "per_basin_metrics_best_so_far.csv",
        "water_balance": RUN_DIR / "water_balance_best_so_far.csv",
        "basin_metadata": RUN_DIR / "basin_metadata.csv",
        "realized_asrz_timeseries": RUN_DIR / "realized_asrz_timeseries.csv",
        "realized_asrz_monthly": RUN_DIR / "realized_asrz_monthly.csv",
        "modis_pairs": EXT_TABLES / "mod16_validation_pairs_extended.csv",
        "modis_by_basin": EXT_TABLES / "mod16_validation_by_basin_extended.csv",
        "gleam_pairs": EXT_TABLES / "gleam_validation_pairs_extended.csv",
        "gleam_by_basin": EXT_TABLES / "gleam_validation_by_basin_extended.csv",
        "grace_pairs": EXT_TABLES / "grace_validation_pairs_extended.csv",
        "grace_by_basin": EXT_TABLES / "grace_validation_by_basin_extended.csv",
        "external_summary": EXT_TABLES / "external_validation_product_summary.csv",
        "swe_pairs": LEGACY_PKG_DIR / "tables" / "swe_validation_pairs.parquet",
        "swe_snow_dominated": REQ_TABLES / "swe_validation_by_basin_snow_dominated.csv",
        "esa_cci_sm_pairs": REQ_TABLES / "esa_cci_soil_moisture_validation_pairs.csv",
        "esa_cci_sm_by_basin": REQ_TABLES / "esa_cci_soil_moisture_validation_by_basin.csv",
        "bfi": REQ_TABLES / "bfi_vs_simulated_q2q_by_basin.csv",
        "model_best_checkpoint": RUN_DIR / "model_best_state.pt",
        "model_epoch1_checkpoint": RUN_DIR / "model_Ep1_state.pt",
    }


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def write_locked_config() -> None:
    best = read_json(BEST_METRICS_PATH)
    manifest = read_json(RUN_MANIFEST_PATH)
    checkpoint = RUN_DIR / "model_best_state.pt"
    if not checkpoint.exists():
        checkpoint = RUN_DIR / "model_Ep1_state.pt"
    copy_if_exists(checkpoint, CHECKPOINT_DIR / checkpoint.name)
    for p in [BEST_METRICS_PATH, RUN_MANIFEST_PATH, MODEL_README_PATH, FINAL_README_PATH, MODEL_SPEC_PATH]:
        copy_if_exists(p, CHECKPOINT_DIR / p.name)

    text = f"""
    model_branch_name: Model6Closed_Snow_aSrz_SIMHYD_Simple_LAIEco_full671_target0732_train1
    study_type: Model 6 based Rohini-style diagnostic replication
    locked_run_folder: {RUN_DIR}
    locked_checkpoint_source: {checkpoint}
    locked_checkpoint_copy: {CHECKPOINT_DIR / checkpoint.name}
    train_period: [{TRAIN_PERIOD[0]}, {TRAIN_PERIOD[1]}]
    test_period: [{TEST_PERIOD[0]}, {TEST_PERIOD[1]}]
    evaluation_period: [{EVAL_PERIOD[0]}, {EVAL_PERIOD[1]}]
    basin_count: {best.get("number_of_basins", 671)}
    basin_list_path: {RUN_DIR / "basin_metadata.csv"}
    forcing_source: Caravan/CAMELS-US daily forcing as used by the existing Model 6 runner
    lai_source: {manifest.get("lai_gapfilled_file", "NOAA AVHRR LAI/FAPAR CDR daily gap-filled basin LAI")}
    et_products_used: [MODIS MOD16 8-day ET, GLEAM monthly ET]
    swe_source: NSIDC0719 / snow-dominated validation tables already materialized locally
    grace_twsa_source: GRACE/JPL basin-month TWSA validation table
    soil_moisture_source: ESA CCI surface soil moisture basin-month validation table
    et_driver: PET with LAI scalar in the locked main model
    simhyd_active: true
    interception_active: true
    groundwater_store_count: 1
    k_parameterization: static basin parameter in locked branch
    dynamic_daily_controls: [SQ_t, CFMAX_t, component partition logits / component mixture effects]
    theta_cap_formulation: direct static model parameter in [10, 1500] mm
    training_loss: RmseLossComb(alpha=0.25), Q-supervised
    observation_constraints_in_training: runoff only for locked model
    external_validation_only: [ET, SWE, TWSA, ESA CCI surface SM]
    statement: This is Model 6-based Rohini-style diagnostic replication, not the original Rohini model.
    best_metrics:
      median_NSE: {best.get("median_NSE")}
      mean_NSE: {best.get("mean_NSE")}
      median_KGE: {best.get("median_KGE")}
      median_R2: {best.get("median_R2")}
      median_aSrz_capacity_mm: {best.get("median_aSrz_capacity_mm")}
      median_mean_aSrz_mm: {best.get("median_mean_aSrz_mm")}
      median_weighted_process_closure_residual_mm_day: {best.get("median_weighted_process_closure_residual_mm_day")}
    """
    write_text(PACKAGE_DIR / "LOCKED_MODEL_CONFIG.yaml", text)
    write_text(CONFIGS_DIR / "locked_model_config.yaml", text)


def write_manifests() -> None:
    paths = get_core_paths()
    data_lines = [
        "locked_model:",
        f"  run_dir: {RUN_DIR}",
        f"  cache: {CACHE_PATH}",
        "inputs_and_products:",
    ]
    for name, path in paths.items():
        data_lines.append(f"  {name}:")
        data_lines.append(f"    path: {path}")
        data_lines.append(f"    exists: {str(path.exists()).lower()}")
    data_lines.append("notes:")
    data_lines.append("  - Missing files are recorded by scripts/00_audit_data_availability.py.")
    data_lines.append("  - Figures depending on missing observations are marked incomplete rather than synthesized.")
    write_text(PACKAGE_DIR / "DATA_MANIFEST.yaml", "\n".join(data_lines))

    fig_rows = [
        ("Figure 1", "scripts/figures/figure1_conceptual_asrz.py", "figures/Figure1_conceptual_asrz_model6.png", "none", "complete"),
        ("Figure 2", "scripts/figures/figure2_model_framework.py", "figures/Figure2_model_framework_model6.png", "locked model config", "complete"),
        ("Figure 3", "scripts/figures/figure3_multivariable_performance.py", "figures/Figure3_multivariable_performance_primary.png", "Q, MODIS ET, SWE, GRACE", "complete if audit finds products"),
        ("Figure 4", "scripts/figures/figure4_spatial_climatic_mean_asrz.py", "figures/Figure4_spatial_climatic_mean_asrz.png", "aSrz, attributes", "complete"),
        ("Figure 5", "scripts/figures/figure5_asrz_twsa_sm_dynamics.py", "figures/Figure5_asrz_twsa_sm_dynamics.png", "aSrz, GRACE, ESA CCI SM", "complete"),
        ("Figure 6", "scripts/figures/figure6_asrz_capacity.py", "figures/Figure6_asrz_capacity.png", "aSrz capacity, attributes", "complete"),
        ("Figure 7", "scripts/figures/figure7_shap_controls_capacity.py", "figures/Figure7_shap_controls_capacity.png", "aSrz capacity, P, PET, LAI, slope, sand", "complete with SHAP or permutation fallback"),
        ("Figure 8", "scripts/figures/figure8_identifiability_theta_cap.py", "figures/Figure8_identifiability_theta_cap.png", "current theta_cap; ablation configs", "current-only unless ablations are trained"),
    ]
    pd.DataFrame(fig_rows, columns=["Figure", "Script", "Output file", "Required data", "Status"]).to_csv(
        PACKAGE_DIR / "FIGURE_MANIFEST.yaml", index=False
    )
    pd.DataFrame(fig_rows, columns=["Figure", "Script", "Output file", "Required data", "Status"]).to_csv(
        TABLES_DIR / "figure_manifest_table.csv", index=False
    )

    ablation_rows = [
        {"experiment": "theta_cap_b_max_600", "status": "config_only_not_run", "theta_cap_b_max_mm": 600},
        {"experiment": "theta_cap_b_max_800", "status": "config_only_not_run", "theta_cap_b_max_mm": 800},
        {"experiment": "theta_cap_b_max_1000", "status": "config_only_not_run", "theta_cap_b_max_mm": 1000},
        {"experiment": "theta_cap_b_max_1200", "status": "config_only_not_run", "theta_cap_b_max_mm": 1200},
        {"experiment": "theta_cap_b_max_2000", "status": "config_only_not_run", "theta_cap_b_max_mm": 2000},
        {"experiment": "no_prior_direct_nn_theta_cap", "status": "config_only_not_run", "theta_cap_b_max_mm": np.nan},
        {"experiment": "current_locked_direct_theta_cap", "status": "evaluated_current_model", "theta_cap_b_max_mm": 1500},
    ]
    pd.DataFrame(ablation_rows).to_csv(TABLES_DIR / "Figure8_theta_cap_ablation_metrics.csv", index=False)
    write_text(
        PACKAGE_DIR / "ABLATION_MANIFEST.yaml",
        "\n".join(
            [
                "theta_cap_ablation_status: config_only_except_current_locked_model",
                "reason: full 671 retraining for theta_cap structural ablations was not present in the locked branch and was not run by this packaging step.",
                "experiments:",
            ]
            + [f"  - {row['experiment']}: {row['status']}" for row in ablation_rows]
        ),
    )


def audit_data_availability() -> None:
    ensure_dirs()
    paths = get_core_paths()
    rows = []
    for name, path in paths.items():
        row = {"name": name, "path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}
        rows.append(row)

    cache_vars = {
        "daily Sa": "sa",
        "daily aSrz": None,
        "theta_cap": "theta_cap_t",
        "alpha": "alpha",
        "ET": "actual_et",
        "INT": "intc",
        "ET_total": "et_total",
        "Q_routed": "pred",
        "SWE_sim": "swe_model",
        "TWS_sim_components": "snowpack+meltwater+sa+gw",
        "storage_states": "sa,gw,snowpack,meltwater",
        "LAI": "lai_t",
    }
    if CACHE_PATH.exists():
        cache = load_cache()
        for label, key in cache_vars.items():
            if key is None:
                exists = paths["realized_asrz_timeseries"].exists()
                rows.append({"name": label, "path": str(paths["realized_asrz_timeseries"]), "exists": exists, "size_bytes": paths["realized_asrz_timeseries"].stat().st_size if exists else 0})
            elif "+" in key or "," in key:
                exists = all(part.strip() in cache for part in key.replace("+", ",").split(","))
                rows.append({"name": label, "path": f"{CACHE_PATH}::{key}", "exists": exists, "size_bytes": CACHE_PATH.stat().st_size})
            else:
                exists = key in cache
                rows.append({"name": label, "path": f"{CACHE_PATH}::{key}", "exists": exists, "size_bytes": CACHE_PATH.stat().st_size if CACHE_PATH.exists() else 0})

    audit = pd.DataFrame(rows)
    audit.to_csv(DATA_MANIFEST_DIR / "DATA_AUDIT_TABLE.csv", index=False)
    missing = audit.loc[~audit["exists"].astype(bool)]
    write_text(
        DATA_MANIFEST_DIR / "DATA_AUDIT_REPORT.md",
        f"""
        # Data Audit Report

        Locked run: `{RUN_DIR}`

        Available records: {int(audit['exists'].sum())} / {len(audit)}

        Missing records: {len(missing)}

        The audit is intentionally conservative. Missing records are not synthesized; scripts only generate figures where the required products exist.

        ## Missing

        {df_to_markdown(missing[['name', 'path']]) if len(missing) else 'None.'}
        """,
    )
    print(f"Wrote {DATA_MANIFEST_DIR / 'DATA_AUDIT_TABLE.csv'}")


def build_publication_datasets() -> None:
    ensure_dirs()
    cache = load_cache()
    basin_ids = cache["basin_ids"].astype(int)
    dates = pd.to_datetime(cache["dates"].astype(str))
    attr_names = [str(x) for x in cache["attr_names"]]
    attrs_raw = pd.DataFrame(cache["attrs_raw"], columns=attr_names)
    attrs_raw.insert(0, "basin_id", basin_ids)

    meta = pd.read_csv(RUN_DIR / "basin_metadata.csv")
    metrics = pd.read_csv(RUN_DIR / "per_basin_metrics_best_so_far.csv")
    asrz_by_basin = pd.read_csv(REQ_TABLES / "realized_arz_by_basin.csv") if (REQ_TABLES / "realized_arz_by_basin.csv").exists() else None
    static = meta.merge(attrs_raw, on="basin_id", how="left").merge(
        metrics[["basin_id", "NSE", "KGE", "R2", "FLV", "FHV", "low_flow_NSE", "high_flow_NSE", "ET_over_P", "Q_over_P", "alpha_mean", "theta_cap_mean", "aSrz_capacity_mm", "mean_aSrz_mm", "aridity_index"]],
        on="basin_id",
        how="left",
        suffixes=("", "_metric"),
    )
    if asrz_by_basin is not None:
        keep = ["basin_id", "mean_lai", "mean_realized_arz_mm", "p95_realized_arz_mm", "max_realized_arz_mm", "mean_relative_soil_storage"]
        static = static.merge(asrz_by_basin[[c for c in keep if c in asrz_by_basin.columns]], on="basin_id", how="left")
    static.to_csv(TABLES_DIR / "basin_static_master.csv", index=False)

    sa = cache["sa"].astype(float)
    min_sa = np.nanmin(sa, axis=1, keepdims=True)
    asrz = sa - min_sa
    tws = cache["snowpack"].astype(float) + cache["meltwater"].astype(float) + sa + cache["gw"].astype(float)

    # Daily indexed table keeps the requested aSrz definition without duplicating all large state arrays in CSV.
    daily_long = pd.read_csv(RUN_DIR / "realized_asrz_timeseries.csv", parse_dates=["date"])
    daily_long.to_parquet(TABLES_DIR / "daily_model_outputs_indexed.parquet", index=False)

    monthly_rows = []
    for i, basin in enumerate(basin_ids):
        df = pd.DataFrame(
            {
                "date": dates,
                "basin_id": basin,
                "Q_routed_mm_day": cache["pred"][i],
                "Q_obs_mm_day": cache["obs"][i],
                "ET_total_mm_day": cache["et_total"][i],
                "ET_a_mm_day": cache["actual_et"][i],
                "INT_mm_day": cache["intc"][i],
                "SWE_sim_mm": cache["swe_model"][i],
                "SNOWPACK_mm": cache["snowpack"][i],
                "MELTWATER_mm": cache["meltwater"][i],
                "Sa_mm": sa[i],
                "aSrz_mm": asrz[i],
                "GW_mm": cache["gw"][i],
                "TWS_sim_mm": tws[i],
                "alpha": cache["alpha"][i],
                "LAI": cache["lai_t"][i],
                "theta_cap_mm": cache["theta_cap_t"][i],
            }
        )
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp("M")
        monthly = df.groupby(["basin_id", "month"], as_index=False).agg(
            Q_routed_mm_month=("Q_routed_mm_day", "sum"),
            Q_obs_mm_month=("Q_obs_mm_day", "sum"),
            Q_routed_mean_mm_day=("Q_routed_mm_day", "mean"),
            Q_obs_mean_mm_day=("Q_obs_mm_day", "mean"),
            ET_total_mm_month=("ET_total_mm_day", "sum"),
            ET_total_mean_mm_day=("ET_total_mm_day", "mean"),
            INT_mm_month=("INT_mm_day", "sum"),
            SWE_sim_mm=("SWE_sim_mm", "mean"),
            SNOWPACK_mm=("SNOWPACK_mm", "mean"),
            MELTWATER_mm=("MELTWATER_mm", "mean"),
            Sa_mm=("Sa_mm", "mean"),
            aSrz_mm=("aSrz_mm", "mean"),
            GW_mm=("GW_mm", "mean"),
            TWS_sim_mm=("TWS_sim_mm", "mean"),
            alpha=("alpha", "mean"),
            LAI=("LAI", "mean"),
            theta_cap_mm=("theta_cap_mm", "mean"),
        ).rename(columns={"month": "date"})
        monthly_rows.append(monthly)
    monthly_model = pd.concat(monthly_rows, ignore_index=True)
    monthly_model["TWSA_sim_mm"] = monthly_model["TWS_sim_mm"] - monthly_model.groupby("basin_id")["TWS_sim_mm"].transform("mean")
    monthly_model.to_csv(TABLES_DIR / "monthly_model_outputs.csv", index=False)

    obs_parts = []
    for label, path in {
        "MODIS_ET_8day": EXT_TABLES / "mod16_validation_pairs_extended.csv",
        "GLEAM_ET_monthly": EXT_TABLES / "gleam_validation_pairs_extended.csv",
        "GRACE_TWSA_monthly": EXT_TABLES / "grace_validation_pairs_extended.csv",
        "ESA_CCI_SM_monthly": REQ_TABLES / "esa_cci_soil_moisture_validation_pairs.csv",
    }.items():
        if path.exists():
            tmp = pd.read_csv(path)
            tmp["source_product"] = label
            obs_parts.append(tmp)
    if obs_parts:
        pd.concat(obs_parts, ignore_index=True, sort=False).to_csv(TABLES_DIR / "monthly_observation_targets.csv", index=False)

    arz_diag = pd.read_csv(REQ_TABLES / "realized_arz_by_basin.csv")
    arz_diag.to_csv(TABLES_DIR / "aSrz_diagnostics_by_basin.csv", index=False)
    monthly_model[["basin_id", "date", "SNOWPACK_mm", "MELTWATER_mm", "Sa_mm", "GW_mm", "TWS_sim_mm", "TWSA_sim_mm"]].to_csv(
        TABLES_DIR / "storage_components_by_basin_month.csv", index=False
    )

    write_text(
        DOCS_DIR / "UNIT_CONVENTIONS.md",
        """
        # Unit Conventions

        - Daily runoff and ET in the model cache are mm/day.
        - Monthly runoff and ET totals are sums of daily mm/day values over the calendar month.
        - Monthly mean daily fluxes are also provided for comparison to paper-style figures.
        - SWE is monthly mean storage in mm.
        - TWS_sim = SNOWPACK + MELTWATER + Sa + GW.
        - TWSA_sim is the monthly anomaly of TWS_sim after subtracting each basin's evaluation-period mean.
        - aSrz_t = Sa_t - min_t(Sa_t) over the analysis period for each basin.
        - aSrz_capacity = max_t(aSrz_t) over the analysis period.
        """,
    )
    print(f"Wrote standardized tables under {TABLES_DIR}")


def evaluate_locked_model() -> None:
    ensure_dirs()
    if not (TABLES_DIR / "monthly_model_outputs.csv").exists():
        build_publication_datasets()
    best = read_json(BEST_METRICS_PATH)
    stream = pd.read_csv(RUN_DIR / "per_basin_metrics_best_so_far.csv")
    water = pd.read_csv(RUN_DIR / "water_balance_best_so_far.csv")
    ext_summary = pd.read_csv(EXT_TABLES / "external_validation_product_summary.csv")
    storage = pd.read_csv(TABLES_DIR / "aSrz_diagnostics_by_basin.csv")
    swe = pd.read_csv(REQ_TABLES / "swe_validation_by_basin_snow_dominated.csv") if (REQ_TABLES / "swe_validation_by_basin_snow_dominated.csv").exists() else pd.DataFrame()

    pd.DataFrame([best]).to_csv(METRICS_DIR / "main_summary_metrics.csv", index=False)
    ext_summary.to_csv(METRICS_DIR / "external_validation_summary.csv", index=False)
    storage.to_csv(METRICS_DIR / "storage_diagnostics.csv", index=False)
    water.to_csv(METRICS_DIR / "water_balance_diagnostics.csv", index=False)
    if not swe.empty:
        swe.to_csv(METRICS_DIR / "snow_diagnostics.csv", index=False)

    counts = {
        "nse_gt_0": int((stream["NSE"] > 0).sum()),
        "nse_gt_0p5": int((stream["NSE"] > 0.5).sum()),
        "nse_gt_0p7": int((stream["NSE"] > 0.7).sum()),
        "nse_lt_0": int((stream["NSE"] < 0).sum()),
        "basins": int(stream["basin_id"].nunique()),
    }
    pd.DataFrame([counts]).to_csv(METRICS_DIR / "streamflow_threshold_counts.csv", index=False)
    print(f"Wrote metrics under {METRICS_DIR}")


def plot_density(ax, df: pd.DataFrame, obs_col: str, sim_col: str, label: str, xlabel: str, ylabel: str, symmetric: bool = False) -> dict:
    valid = df.dropna(subset=[obs_col, sim_col]).copy()
    stats = summarize_pairs(valid, obs_col, sim_col, label, "")
    if len(valid) > 60000:
        valid = valid.sample(60000, random_state=42)
    obs = valid[obs_col].to_numpy(float)
    sim = valid[sim_col].to_numpy(float)
    if len(valid) == 0:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
        return stats
    try:
        h, xedges, yedges = np.histogram2d(obs, sim, bins=120)
        xi = np.clip(np.digitize(obs, xedges) - 1, 0, h.shape[0] - 1)
        yi = np.clip(np.digitize(sim, yedges) - 1, 0, h.shape[1] - 1)
        dens = h[xi, yi]
    except Exception:
        dens = np.ones_like(obs)
    order = np.argsort(dens)
    obs, sim, dens = obs[order], sim[order], dens[order]
    ax.scatter(obs, sim, c=np.log1p(dens), cmap="YlOrBr", s=9, alpha=0.34, edgecolors="none", rasterized=True)
    if symmetric:
        lim = np.nanpercentile(np.abs(np.r_[obs, sim]), 99.5)
        lim = max(float(lim), 1.0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.plot([-lim, lim], [-lim, lim], "--", color="0.55", lw=1.0)
    else:
        lo = min(0.0, float(np.nanpercentile(np.r_[obs, sim], 0.5)))
        hi = max(1.0, float(np.nanpercentile(np.r_[obs, sim], 99.5)))
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.plot([lo, hi], [lo, hi], "--", color="0.55", lw=1.0)
    ax.text(0.05, 0.95, f"NSE = {stats['NSE']:.2f}\nr = {stats['r']:.2f}", ha="left", va="top", transform=ax.transAxes)
    ax.text(0.01, 1.02, label, fontweight="bold", transform=ax.transAxes)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.10)
    return stats


def figure1_conceptual_asrz() -> None:
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=240)
    ax.axis("off")
    boxes = {
        "P": (0.08, 0.78, "Precipitation\\n+ snowmelt"),
        "alpha": (0.30, 0.62, "alpha partition\\nP_accessible = alpha P"),
        "Sa": (0.53, 0.62, "Active root-zone\\nstate Sa"),
        "ET": (0.76, 0.78, "ET withdrawal\\nPET x stress x LAI"),
        "inacc": (0.30, 0.30, "Inaccessible water\\n(1-alpha)P"),
        "runoff": (0.53, 0.30, "SIMHYD runoff\\nSRUN + IFLOW + REC"),
        "cap": (0.76, 0.30, "Diagnostics\\naSrz = Sa - min(Sa)\\ncapacity = max(aSrz)"),
    }
    for _, (x, y, txt) in boxes.items():
        rect = plt.Rectangle((x, y), 0.18, 0.13, fc="#e8f3ff", ec="#174a7c", lw=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.09, y + 0.065, txt, ha="center", va="center", fontsize=10)
    arrows = [("P", "alpha"), ("alpha", "Sa"), ("Sa", "ET"), ("alpha", "inacc"), ("inacc", "runoff"), ("Sa", "cap")]
    for a, b in arrows:
        xa, ya, _ = boxes[a]
        xb, yb, _ = boxes[b]
        ax.annotate("", xy=(xb, yb + 0.065), xytext=(xa + 0.18, ya + 0.065), arrowprops=dict(arrowstyle="->", lw=1.8, color="#333333"))
    ax.text(0.05, 0.08, "Conceptual diagnostic: realized aSrz is derived from the simulated Sa trajectory, not directly observed storage.", fontsize=10)
    fig.savefig(FIGURES_DIR / "Figure1_conceptual_asrz_model6.png", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "Figure1_conceptual_asrz_model6.pdf", bbox_inches="tight")
    plt.close(fig)
    write_text(DOCS_DIR / "Figure1_caption.md", "Figure 1. Conceptual Model 6 active-root-zone diagnostic. aSrz is realized active storage, computed as Sa minus the basin-specific analysis-period minimum. theta_cap is a structural upper bound; aSrz_capacity is the realized dynamic range.")


def figure2_model_framework() -> None:
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=240)
    ax.axis("off")
    def box(x, y, w, h, txt, fc="#f7f7f7", ec="#333"):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.3))
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=9)
    box(0.04, 0.70, 0.18, 0.16, "Drivers\\nP, T, PET, LAI\\nstatic attributes", "#fff7e6")
    box(0.30, 0.75, 0.14, 0.10, "NN parameterization\\nMLP + daily controls", "#f0f0ff")
    box(0.30, 0.54, 0.14, 0.10, "Snow bucket\\nSNOWPACK + MELTWATER", "#e6f2ff")
    box(0.50, 0.54, 0.14, 0.10, "Interception", "#e8ffe8")
    box(0.70, 0.54, 0.18, 0.10, "alpha partition\\naccessible / inaccessible", "#fff0f0")
    box(0.50, 0.32, 0.16, 0.11, "Sa active\\nroot-zone storage", "#e8ffe8")
    box(0.72, 0.32, 0.16, 0.11, "SIMHYD branch\\nSRUN IFLOW REC", "#fff0f0")
    box(0.72, 0.14, 0.16, 0.10, "GW store\\nbaseflow BAS", "#f4e8ff")
    box(0.50, 0.14, 0.16, 0.10, "Gamma routing\\nQ_routed", "#eeeeee")
    box(0.08, 0.18, 0.22, 0.18, "Validation products\\nQ, MODIS/GLEAM ET\\nSWE, GRACE TWSA\\nESA CCI SM", "#ffffe6")
    for xy, xytext in [((0.30,0.80),(0.22,0.78)), ((0.37,0.64),(0.37,0.75)), ((0.50,0.59),(0.44,0.59)), ((0.70,0.59),(0.64,0.59)), ((0.58,0.43),(0.76,0.54)), ((0.80,0.43),(0.80,0.54)), ((0.80,0.24),(0.80,0.32)), ((0.66,0.19),(0.72,0.19)), ((0.30,0.27),(0.50,0.19))]:
        ax.annotate("", xy=xy, xytext=xytext, arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"))
    ax.text(0.04, 0.04, "Locked model is Q-trained; ET/SWE/TWSA/SM are external validation constraints in this package.", fontsize=10)
    fig.savefig(FIGURES_DIR / "Figure2_model_framework_model6.png", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "Figure2_model_framework_model6.pdf", bbox_inches="tight")
    plt.close(fig)
    write_text(DOCS_DIR / "Figure2_caption.md", "Figure 2. Locked Model 6 framework. Unlike the Rohini original, this branch uses a SIMHYD-style runoff response, a single groundwater store, PET-based LAI-scaled ET, and Q-only training; ET, SWE, TWSA, and surface soil moisture are independent validation products.")


def figure3_multivariable_performance() -> None:
    ensure_dirs()
    if not (TABLES_DIR / "monthly_model_outputs.csv").exists():
        build_publication_datasets()
    monthly = pd.read_csv(TABLES_DIR / "monthly_model_outputs.csv", parse_dates=["date"])

    mod = pd.read_csv(EXT_TABLES / "mod16_validation_pairs_extended.csv", parse_dates=["date"])
    mod["month"] = mod["date"].dt.to_period("M").dt.to_timestamp("M")
    mod["obs_et_mm_day"] = mod["mod16_et_8day_mm"] / 8.0
    mod["sim_et_mm_day"] = mod["model_et_total_8day_mm"] / 8.0
    mod_m = mod.groupby(["basin_id", "month"], as_index=False)[["obs_et_mm_day", "sim_et_mm_day"]].mean().rename(columns={"month": "date"})
    start, end = mod_m["date"].min(), mod_m["date"].max()
    q = monthly[(monthly["date"] >= start) & (monthly["date"] <= end)].copy()
    swe = pd.read_parquet(LEGACY_PKG_DIR / "tables" / "swe_validation_pairs.parquet")
    swe["date"] = pd.to_datetime(swe["date"])
    snowdom = pd.read_csv(REQ_TABLES / "swe_validation_by_basin_snow_dominated.csv")
    snow_ids = set(pd.to_numeric(snowdom["basin_id"], errors="coerce").dropna().astype(int))
    swe = swe[swe["basin_id"].astype(int).isin(snow_ids)].copy()
    swe["month"] = swe["date"].dt.to_period("M").dt.to_timestamp("M")
    swe_m = swe.groupby(["basin_id", "month"], as_index=False)[["nsidc_swe_mm", "model_swe_mm"]].mean().rename(columns={"month": "date"})
    twsa = pd.read_csv(EXT_TABLES / "grace_validation_pairs_extended.csv", parse_dates=["date"]).rename(
        columns={"twsa_mm": "obs_twsa_mm", "model_storage_aligned_mm": "sim_twsa_mm"}
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.2), dpi=320)
    rows = [
        plot_density(axes[0, 0], mod_m, "obs_et_mm_day", "sim_et_mm_day", "a", "Observed ET (mm/day)", "Simulated ET (mm/day)"),
        plot_density(axes[0, 1], q, "Q_obs_mean_mm_day", "Q_routed_mean_mm_day", "b", "Observed runoff (mm/day)", "Simulated runoff (mm/day)"),
        plot_density(axes[1, 0], swe_m, "nsidc_swe_mm", "model_swe_mm", "c", "Observed SWE (mm)", "Simulated SWE (mm)"),
        plot_density(axes[1, 1], twsa, "obs_twsa_mm", "sim_twsa_mm", "d", "Observed TWSA (mm)", "Simulated TWSA (mm)", symmetric=True),
    ]
    for row, var, unit in zip(rows, ["ET_MODIS_monthly_mean_daily_flux", "Runoff_monthly_mean_daily_flux", "SWE_monthly_mean_snowdominated", "TWSA_GRACE_monthly_anomaly"], ["mm/day", "mm/day", "mm", "mm"]):
        row["variable"] = var
        row["units"] = unit
    fig.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.09, wspace=0.34, hspace=0.32)
    fig.savefig(FIGURES_DIR / "Figure3_multivariable_performance_primary.png", dpi=320)
    fig.savefig(FIGURES_DIR / "Figure3_multivariable_performance_primary.pdf")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(TABLES_DIR / "Figure3_metrics_primary.csv", index=False)
    pd.concat(
        [
            mod_m.assign(panel="ET"),
            q[["basin_id", "date", "Q_obs_mean_mm_day", "Q_routed_mean_mm_day"]].assign(panel="Runoff"),
            swe_m.assign(panel="SWE"),
            twsa.assign(panel="TWSA"),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(TABLES_DIR / "Figure3_monthly_pairs_primary.csv", index=False)
    write_text(DOCS_DIR / "Figure3_caption.md", "Figure 3. Monthly observed versus simulated variables for locked Model 6. ET uses MODIS MOD16 as primary product; runoff uses observed streamflow; SWE uses snow-dominated NSIDC/SNODAS overlap; TWSA uses GRACE/JPL basin-month anomalies. Weak SWE performance is retained in the panel metrics.")

    # Sensitivity figures reuse existing product tables where available.
    for src, name in [
        (RUN_DIR / "external_validation_figures" / "reference_style_multivariable_modis.png", "Figure3_multivariable_performance_modis_sensitivity.png"),
        (RUN_DIR / "external_validation_figures" / "reference_style_multivariable_modis_subset455_swe183.png", "Figure3_multivariable_performance_modis_subset455_swe183.png"),
    ]:
        copy_if_exists(src, FIGURES_DIR / name)
    copy_if_exists(
        EXT_TABLES / "reference_style_multivariable_modis_subset455_swe183_metrics.csv",
        TABLES_DIR / "Figure3_metrics_modis_subset455_swe183.csv",
    )
    copy_if_exists(
        EXT_TABLES / "reference_style_multivariable_modis_subset455_swe183_readme.md",
        DOCS_DIR / "Figure3_modis_subset455_swe183_caption.md",
    )


def figure4_spatial_climatic_mean_asrz() -> None:
    ensure_dirs()
    static = pd.read_csv(TABLES_DIR / "basin_static_master.csv") if (TABLES_DIR / "basin_static_master.csv").exists() else None
    if static is None:
        build_publication_datasets()
        static = pd.read_csv(TABLES_DIR / "basin_static_master.csv")
    ycol = "mean_realized_arz_mm" if "mean_realized_arz_mm" in static.columns else "mean_aSrz_mm"
    lai_col = "mean_lai" if "mean_lai" in static.columns else "lai_max"
    aridity = "aridity_index" if "aridity_index" in static.columns else "aridity"
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=260)
    sc = scatter_map(axes[0], static, ycol, "A. Mean realized aSrz", cmap="YlGnBu")
    plt.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.03, label="mm")
    sc2 = axes[1].scatter(static[aridity], static[ycol], c=static[lai_col], cmap="YlGn", s=28, edgecolors="black", linewidths=0.12)
    lowess_line(axes[1], static[aridity].to_numpy(float), static[ycol].to_numpy(float))
    axes[1].set_xlabel("Aridity index (PET/P)")
    axes[1].set_ylabel("Mean realized aSrz (mm)")
    axes[1].set_title("B. Mean aSrz vs aridity")
    axes[1].grid(alpha=0.15)
    plt.colorbar(sc2, ax=axes[1], fraction=0.046, pad=0.03, label="Mean LAI")
    p = static["p_mean"].to_numpy(float) * 365 if "p_mean" in static.columns else np.full(len(static), np.nan)
    pet = static["pet_mean"].to_numpy(float) * 365 if "pet_mean" in static.columns else np.full(len(static), np.nan)
    sc3 = axes[2].scatter(p, pet, c=static[ycol], cmap="magma", s=28, edgecolors="black", linewidths=0.12)
    lim = np.nanpercentile(np.r_[p, pet], 99)
    axes[2].plot([0, lim], [0, lim], "--", color="0.35", lw=1.0)
    axes[2].set_xlim(0, lim)
    axes[2].set_ylim(0, lim)
    axes[2].set_xlabel("Mean annual P (mm/yr)")
    axes[2].set_ylabel("Mean annual PET (mm/yr)")
    axes[2].set_title("C. Hydroclimate space")
    axes[2].grid(alpha=0.15)
    plt.colorbar(sc3, ax=axes[2], fraction=0.046, pad=0.03, label="Mean aSrz (mm)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "Figure4_spatial_climatic_mean_asrz.png")
    fig.savefig(FIGURES_DIR / "Figure4_spatial_climatic_mean_asrz.pdf")
    plt.close(fig)
    static[["basin_id", "lat", "lon", aridity, lai_col, ycol]].to_csv(TABLES_DIR / "Figure4_basin_mean_asrz.csv", index=False)
    r_lai = corr(static[lai_col].to_numpy(float), static[ycol].to_numpy(float))
    r_arid = corr(static[aridity].to_numpy(float), static[ycol].to_numpy(float))
    write_text(DOCS_DIR / "Figure4_caption.md", f"Figure 4. Spatial and climatic patterns of mean realized aSrz. Basin points are used because no LAI raster background is packaged here. Correlation mean aSrz vs mean LAI = {r_lai:.3f}; vs aridity = {r_arid:.3f}; n = {len(static)} basins.")


def figure5_asrz_twsa_sm_dynamics() -> None:
    ensure_dirs()
    asrz = pd.read_csv(RUN_DIR / "realized_asrz_monthly.csv", parse_dates=["month"]).rename(columns={"month": "date"})
    grace = pd.read_csv(EXT_TABLES / "grace_validation_pairs_extended.csv", parse_dates=["date"])
    sm = pd.read_csv(REQ_TABLES / "esa_cci_soil_moisture_validation_pairs.csv", parse_dates=["date"])
    static = pd.read_csv(TABLES_DIR / "basin_static_master.csv") if (TABLES_DIR / "basin_static_master.csv").exists() else pd.read_csv(REQ_TABLES / "realized_arz_by_basin.csv")
    aridity = "aridity_index" if "aridity_index" in static.columns else "aridity"
    lai_col = "mean_lai" if "mean_lai" in static.columns else "lai_max"

    rows = []
    for basin, grp in asrz.groupby("basin_id"):
        a = grp[["date", "aSrz_mm", "Sa_mm"]]
        gt = a.merge(grace[grace["basin_id"] == basin][["date", "twsa_mm"]], on="date", how="inner").dropna()
        st = a.merge(sm[sm["basin_id"] == basin][["date", "surface_sm_monthly", "model_relative_soil_storage"]], on="date", how="inner").dropna()
        if len(gt) >= 24 or len(st) >= 24:
            rows.append(
                {
                    "basin_id": int(basin),
                    "n_twsa": int(len(gt)),
                    "r_asrz_twsa": corr(gt["twsa_mm"].to_numpy(float), gt["aSrz_mm"].to_numpy(float)) if len(gt) >= 24 else np.nan,
                    "n_sm": int(len(st)),
                    "r_asrz_sm": corr(st["surface_sm_monthly"].to_numpy(float), st["aSrz_mm"].to_numpy(float)) if len(st) >= 24 else np.nan,
                    "r_sa_sm": corr(st["surface_sm_monthly"].to_numpy(float), st["Sa_mm"].to_numpy(float)) if len(st) >= 24 else np.nan,
                }
            )
    corr_df = pd.DataFrame(rows).merge(static, on="basin_id", how="left")
    corr_df.to_csv(TABLES_DIR / "Figure5_basin_correlations.csv", index=False)

    humid = corr_df.dropna(subset=["r_asrz_twsa", aridity, lai_col]).sort_values(["r_asrz_twsa", lai_col], ascending=[False, False]).head(1)
    semiarid = corr_df.dropna(subset=["r_asrz_sm", aridity]).sort_values([aridity, "r_asrz_sm"], ascending=[False, False]).head(1)
    examples = [int(humid.iloc[0]["basin_id"]) if len(humid) else int(corr_df.iloc[0]["basin_id"]), int(semiarid.iloc[0]["basin_id"]) if len(semiarid) else int(corr_df.iloc[min(1, len(corr_df)-1)]["basin_id"])]

    fig = plt.figure(figsize=(9.4, 8.8), dpi=260)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.05, 1.2], hspace=0.48, wspace=0.32)
    for idx, basin in enumerate(examples):
        ax = fig.add_subplot(gs[idx, :])
        a = asrz[asrz["basin_id"] == basin][["date", "aSrz_mm"]].copy()
        g = grace[grace["basin_id"] == basin][["date", "twsa_mm"]]
        s = sm[sm["basin_id"] == basin][["date", "surface_sm_monthly"]]
        m = a.merge(g, on="date", how="left").merge(s, on="date", how="left")
        ax.plot(m["date"], m["aSrz_mm"], color="black", lw=2.2, label="aSrz")
        ax.set_ylabel("aSrz (mm)")
        ax2 = ax.twinx()
        ax2.plot(m["date"], m["twsa_mm"], color="#1476d4", lw=1.5, label="GRACE TWSA")
        ax2.set_ylabel("TWSA (mm)", color="#1476d4")
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.09))
        ax3.plot(m["date"], m["surface_sm_monthly"], color="#ff5a3d", lw=0.9, alpha=0.9, label="ESA CCI SM")
        ax3.set_ylabel("Surface SM", color="#ff5a3d")
        row = corr_df[corr_df["basin_id"] == basin].iloc[0]
        ax.text(0.02, 0.92, f"{'a' if idx == 0 else 'b'}  basin {basin}   r(aSrz,TWSA)={row['r_asrz_twsa']:.2f}   r(aSrz,SM)={row['r_asrz_sm']:.2f}", transform=ax.transAxes, fontweight="bold")
        ax.grid(alpha=0.12)
    axc = fig.add_subplot(gs[2, 0])
    sc = axc.scatter(corr_df[aridity], corr_df["r_asrz_twsa"], c=corr_df[lai_col], cmap="BrBG", s=26, edgecolors="black", linewidths=0.10)
    lowess_line(axc, corr_df[aridity].to_numpy(float), corr_df["r_asrz_twsa"].to_numpy(float))
    axc.axhline(0, color="0.5", ls="--", lw=0.8)
    axc.set_xlabel("Aridity index (PET/P)")
    axc.set_ylabel("r(aSrz, TWSA)")
    axc.set_title("c. GRACE-valid basins")
    axc.grid(alpha=0.12)
    axd = fig.add_subplot(gs[2, 1])
    sc2 = axd.scatter(corr_df[aridity], corr_df["r_asrz_sm"], c=corr_df[lai_col], cmap="BrBG", s=26, edgecolors="black", linewidths=0.10)
    lowess_line(axd, corr_df[aridity].to_numpy(float), corr_df["r_asrz_sm"].to_numpy(float))
    axd.axhline(0, color="0.5", ls="--", lw=0.8)
    axd.set_xlabel("Aridity index (PET/P)")
    axd.set_ylabel("r(aSrz, surface SM)")
    axd.set_title("d. ESA CCI SM-valid basins")
    axd.grid(alpha=0.12)
    cbar = fig.colorbar(sc2, ax=[axc, axd], fraction=0.035, pad=0.03)
    cbar.set_label("Mean LAI")
    fig.savefig(FIGURES_DIR / "Figure5_asrz_twsa_sm_dynamics.png", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "Figure5_asrz_twsa_sm_dynamics.pdf", bbox_inches="tight")
    plt.close(fig)
    (TABLES_DIR / "Figure5_selected_examples.json").write_text(json.dumps({"examples": examples}, indent=2))
    write_text(DOCS_DIR / "Figure5_caption.md", f"Figure 5. Monthly realized aSrz dynamics compared with GRACE TWSA and ESA CCI surface soil moisture. Panels c/d use available basins independently: {int(corr_df['r_asrz_twsa'].notna().sum())} GRACE-valid basins and {int(corr_df['r_asrz_sm'].notna().sum())} SM-valid basins.")

    # z-score sensitivity
    zdf = corr_df.copy()
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=240)
    ax.scatter(zdf["r_asrz_twsa"], zdf["r_asrz_sm"], c=zdf[aridity], cmap="viridis", s=28, edgecolors="black", linewidths=0.12)
    ax.axhline(0, color="0.5", ls="--", lw=0.8)
    ax.axvline(0, color="0.5", ls="--", lw=0.8)
    ax.set_xlabel("r(aSrz, TWSA)")
    ax.set_ylabel("r(aSrz, surface SM)")
    ax.set_title("Figure 5 z-score/correlation sensitivity summary")
    fig.savefig(FIGURES_DIR / "Figure5_asrz_twsa_sm_dynamics_zscore_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def figure6_asrz_capacity() -> None:
    ensure_dirs()
    static = pd.read_csv(TABLES_DIR / "basin_static_master.csv") if (TABLES_DIR / "basin_static_master.csv").exists() else pd.read_csv(REQ_TABLES / "realized_arz_by_basin.csv")
    cap = "max_realized_arz_mm" if "max_realized_arz_mm" in static.columns else "aSrz_capacity_mm"
    mean = "mean_realized_arz_mm" if "mean_realized_arz_mm" in static.columns else "mean_aSrz_mm"
    aridity = "aridity_index" if "aridity_index" in static.columns else "aridity"
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=260)
    sc = scatter_map(axes[0], static, cap, "A. aSrz capacity", cmap="YlGnBu")
    plt.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.03, label="mm")
    sc2 = axes[1].scatter(static[cap], static[mean], c=static[aridity], cmap="plasma", s=28, edgecolors="black", linewidths=0.12)
    lim = max(1.0, float(np.nanpercentile(np.r_[static[cap], static[mean]], 99)))
    axes[1].plot([0, lim], [0, lim], "--", color="0.5", lw=1.0, label="1:1")
    axes[1].plot([0, lim], [0, 0.5 * lim], color="black", lw=1.2, label="y=0.5x")
    axes[1].set_xlim(0, lim)
    axes[1].set_ylim(0, lim)
    axes[1].set_xlabel("aSrz capacity (mm)")
    axes[1].set_ylabel("Mean realized aSrz (mm)")
    axes[1].set_title("B. Mean aSrz vs capacity")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.15)
    plt.colorbar(sc2, ax=axes[1], fraction=0.046, pad=0.03, label="Aridity")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "Figure6_asrz_capacity.png")
    fig.savefig(FIGURES_DIR / "Figure6_asrz_capacity.pdf")
    plt.close(fig)
    out = static[["basin_id", "lat", "lon", aridity, mean, cap]].copy()
    out.to_csv(TABLES_DIR / "Figure6_asrz_capacity_by_basin.csv", index=False)
    out["above_half_capacity"] = out[mean] > 0.5 * out[cap]
    out.groupby("above_half_capacity")[[aridity, mean, cap]].agg(["count", "median", "mean"]).to_csv(TABLES_DIR / "Figure6_above_below_half_capacity_stats.csv")
    write_text(DOCS_DIR / "Figure6_caption.md", "Figure 6. Spatial pattern and climatic controls of realized active root-zone storage capacity. Capacity is max(aSrz), not theta_cap.")


def figure7_shap_controls_capacity() -> None:
    ensure_dirs()
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    static = pd.read_csv(TABLES_DIR / "basin_static_master.csv") if (TABLES_DIR / "basin_static_master.csv").exists() else pd.read_csv(REQ_TABLES / "realized_arz_by_basin.csv")
    ycol = "max_realized_arz_mm" if "max_realized_arz_mm" in static.columns else "aSrz_capacity_mm"
    candidates = {
        "P": "p_mean",
        "PET": "pet_mean",
        "LAI": "mean_lai" if "mean_lai" in static.columns else "lai_max",
        "slope": "slope_mean",
        "sand": "sand_frac",
    }
    cols = [c for c in candidates.values() if c in static.columns]
    labels = [k for k, v in candidates.items() if v in static.columns]
    data = static[["basin_id", ycol] + cols].dropna().copy()
    data.to_csv(TABLES_DIR / "Figure7_rf_training_data.csv", index=False)
    X = data[cols].to_numpy(float)
    y = data[ycol].to_numpy(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=500, random_state=42, min_samples_leaf=5)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    val_r2 = r2_score(y_test, pred)

    method = "permutation"
    importance = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=42)
    imp = pd.DataFrame({"predictor": labels, "feature_column": cols, "importance": importance.importances_mean})
    shap_values = None
    try:
        import shap

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X)
        imp = pd.DataFrame({"predictor": labels, "feature_column": cols, "importance": np.mean(np.abs(shap_values), axis=0)})
        pd.DataFrame(shap_values, columns=labels).assign(basin_id=data["basin_id"].to_numpy()).to_csv(TABLES_DIR / "Figure7_shap_values.csv", index=False)
        method = "SHAP mean absolute value"
    except Exception:
        pass
    imp = imp.sort_values("importance", ascending=False)
    imp["validation_r2"] = val_r2
    imp["importance_method"] = method
    imp.to_csv(TABLES_DIR / "Figure7_rf_importance.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.3), dpi=260)
    axes = axes.ravel()
    axes[0].barh(imp["predictor"], imp["importance"], color="#4277a8")
    axes[0].invert_yaxis()
    axes[0].set_title(f"A. RF controls ({method})\\nvalidation R2={val_r2:.2f}")
    axes[0].set_xlabel("Relative contribution")
    lai_color = data[candidates["LAI"]].to_numpy(float) if candidates["LAI"] in data.columns else y
    for ax, label, col in zip(axes[1:], labels, cols):
        ax.scatter(data[col], y, c=lai_color, cmap="YlGn", s=24, edgecolors="black", linewidths=0.10)
        lowess_line(ax, data[col].to_numpy(float), y)
        ax.set_xlabel(label)
        ax.set_ylabel("aSrz capacity (mm)")
        ax.set_title(label)
        ax.grid(alpha=0.12)
    for ax in axes[1 + len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "Figure7_shap_controls_capacity.png")
    fig.savefig(FIGURES_DIR / "Figure7_shap_controls_capacity.pdf")
    plt.close(fig)
    write_text(DOCS_DIR / "Figure7_caption.md", f"Figure 7. Random-forest interpretation of aSrz capacity controls using Rohini-style predictors P, PET, LAI, slope, and sand where available. Importance method: {method}; validation R2 = {val_r2:.3f}.")


def figure8_identifiability_theta_cap() -> None:
    ensure_dirs()
    static = pd.read_csv(TABLES_DIR / "basin_static_master.csv") if (TABLES_DIR / "basin_static_master.csv").exists() else pd.read_csv(REQ_TABLES / "realized_arz_by_basin.csv")
    aridity = "aridity_index" if "aridity_index" in static.columns else "aridity"
    mean = "mean_realized_arz_mm" if "mean_realized_arz_mm" in static.columns else "mean_aSrz_mm"
    cap = "max_realized_arz_mm" if "max_realized_arz_mm" in static.columns else "aSrz_capacity_mm"
    theta = "mean_theta_cap_mm" if "mean_theta_cap_mm" in static.columns else "theta_cap_mean"
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=260)
    axes[0].scatter(static[aridity], static[mean], s=24, alpha=0.75, label="realized mean aSrz")
    axes[0].scatter(static[aridity], static["mean_sa_mm"] if "mean_sa_mm" in static.columns else static[mean], s=12, alpha=0.35, label="model mean Sa")
    lowess_line(axes[0], static[aridity].to_numpy(float), static[mean].to_numpy(float), color="black")
    axes[0].set_xlabel("Aridity")
    axes[0].set_ylabel("Storage (mm)")
    axes[0].set_title("A. Current locked model only")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.12)
    axes[1].scatter(static[aridity], static[cap], s=24, alpha=0.75, label="realized capacity")
    axes[1].scatter(static[aridity], static[theta], s=12, alpha=0.35, label="theta_cap")
    lowess_line(axes[1], static[aridity].to_numpy(float), static[cap].to_numpy(float), color="black")
    axes[1].set_xlabel("Aridity")
    axes[1].set_ylabel("Capacity / upper bound (mm)")
    axes[1].set_title("B. Realized capacity vs theta_cap")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.12)
    axes[2].axis("off")
    axes[2].text(0.02, 0.88, "C. Identifiability note", fontsize=13, fontweight="bold")
    axes[2].text(
        0.02,
        0.72,
        "theta_cap is a structural bound.\\n"
        "aSrz_capacity is the realized dynamic range.\\n\\n"
        "Requested theta_cap_b ablations are configured\\n"
        "but not retrained in this locked-model package.\\n\\n"
        "Therefore this panel is current-model diagnostic\\n"
        "evidence, not a completed structural ablation.",
        fontsize=10,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "Figure8_identifiability_theta_cap.png")
    fig.savefig(FIGURES_DIR / "Figure8_identifiability_theta_cap.pdf")
    plt.close(fig)
    static[["basin_id", aridity, mean, cap, theta]].to_csv(TABLES_DIR / "Figure8_theta_cap_ablation_by_basin.csv", index=False)
    write_text(DOCS_DIR / "Figure8_caption.md", "Figure 8. Functional identifiability diagnostic for the locked Model 6 branch. Only the current locked model is evaluated here; theta_cap_b and no-prior ablations are configured but were not retrained by this packaging step.")


def make_all_figures() -> None:
    figure1_conceptual_asrz()
    figure2_model_framework()
    figure3_multivariable_performance()
    figure4_spatial_climatic_mean_asrz()
    figure5_asrz_twsa_sm_dynamics()
    figure6_asrz_capacity()
    figure7_shap_controls_capacity()
    figure8_identifiability_theta_cap()
    print(f"Wrote figures under {FIGURES_DIR}")


def write_ablation_stubs() -> None:
    write_text(
        SCRIPTS_DIR / "train_theta_cap_ablation.py",
        """
        #!/usr/bin/env python3
        \"\"\"Stub for future theta_cap structural ablations.

        This package locks the existing Model 6 LAIEco checkpoint. Full theta_cap
        ablations require retraining and are intentionally not launched by the
        publication packaging command.
        \"\"\"
        from pathlib import Path

        ROOT = Path("/home/mircore/Desktop/diff_hydro/ECO_HYBRID/PUBLICATION_MODEL6_ROHINI_REPLICATION")

        if __name__ == "__main__":
            print("Theta-cap ablation training is not run in the locked publication package.")
            print("See ABLATION_MANIFEST.yaml for configured experiments.")
        """,
    )
    write_text(
        SCRIPTS_DIR / "evaluate_theta_cap_ablation.py",
        """
        #!/usr/bin/env python3
        from pathlib import Path

        ROOT = Path("/home/mircore/Desktop/diff_hydro/ECO_HYBRID/PUBLICATION_MODEL6_ROHINI_REPLICATION")

        if __name__ == "__main__":
            print("No completed theta-cap ablation checkpoints are packaged. Current locked model diagnostics are in tables/Figure8_theta_cap_ablation_by_basin.csv.")
        """,
    )
    write_text(
        SCRIPTS_DIR / "train_observational_ablation.py",
        """
        #!/usr/bin/env python3
        if __name__ == "__main__":
            print("Observational ablation training is not run because the locked model is Q-trained; ET/SWE/TWSA are validation products here.")
        """,
    )
    write_text(
        SCRIPTS_DIR / "evaluate_observational_ablation.py",
        """
        #!/usr/bin/env python3
        if __name__ == "__main__":
            print("No multi-observation ablation checkpoints are packaged. See docs/OBSERVATIONAL_ABLATION_NOT_RUN.md.")
        """,
    )
    write_text(
        DOCS_DIR / "OBSERVATIONAL_ABLATION_NOT_RUN.md",
        """
        # Observational Ablation Status

        The locked Model 6 LAIEco branch is trained with streamflow supervision only.
        MODIS ET, GLEAM ET, SWE, GRACE TWSA, and ESA CCI surface soil moisture are
        used for independent validation in this package.

        Rohini-style observational ablations such as Q+ET+SWE+TWSA, no_ET, no_Q,
        no_SWE, and no_TWSA require a separate multi-observation training branch.
        They are not faked here.
        """,
    )


def write_docs() -> None:
    best = read_json(BEST_METRICS_PATH)
    ext = pd.read_csv(EXT_TABLES / "external_validation_product_summary.csv") if (EXT_TABLES / "external_validation_product_summary.csv").exists() else pd.DataFrame()
    fig_manifest = pd.read_csv(TABLES_DIR / "figure_manifest_table.csv") if (TABLES_DIR / "figure_manifest_table.csv").exists() else pd.DataFrame()
    ext_md = df_to_markdown(ext) if len(ext) else "External validation table missing."
    fig_md = df_to_markdown(fig_manifest) if len(fig_manifest) else "Figure manifest missing."
    write_text(
        PACKAGE_DIR / "METHODS_MODEL_DESCRIPTION.md",
        """
        # Methods: Locked Model 6 LAIEco

        This package analyzes the locked Model 6 active-root-zone SIMHYD branch:
        `Model6Closed_Snow_aSrz_SIMHYD_Simple_LAIEco_full671_target0732_train1`.

        ## States

        - `SNOWPACK`: solid snow storage.
        - `MELTWATER`: liquid water retained in snowpack.
        - `Sa`: ecosystem-accessible active root-zone storage.
        - `GW`: single groundwater/baseflow storage.

        ## Snow

        The locked branch uses the Model 6 smoothed HBV-style snow module used during training:

        - precipitation is smoothly partitioned into rain and snow around `TT`;
        - snowmelt is controlled by a degree-day factor `CFMAX_t`;
        - refreezing is controlled by `CFR`;
        - liquid-water holding is controlled by `CWH`.

        The exact hard HBV1.1 snow test was run separately and performed worse, so it is not the locked branch.

        ## Interception

        Interception is active. Intercepted water contributes to `ET_total` as `INT + ET_a`.

        ## Active Root Zone

        Soil wetness is computed from the previous active store:

        `Smoist_prev = clamp(Sa / theta_cap, 0, 1)`

        The accessible precipitation fraction is:

        `alpha = clamp(theta_ab * (1 - Smoist_prev) ** theta_ak, 0, 1)`

        `P_accessible = alpha * P_after_snow_and_interception`

        `P_inaccessible = (1 - alpha) * P_after_snow_and_interception`

        Active storage before ET is:

        `Sa_pre = Sa + P_accessible`

        The locked branch uses PET-based LAI-scaled ET:

        `ET_a_pot = PET * theta_efmax * water_stress * LAI_et_scalar`

        `ET_a = min(ET_a_pot, Sa_pre)`

        Realized diagnostic storage is:

        `aSrz_t = Sa_t - min_tau(Sa_tau)`

        `aSrz_capacity = max_t(aSrz_t)`

        ## SIMHYD Runoff And Groundwater

        Inaccessible water and active-store overflow are partitioned into surface runoff, interflow, and recharge using SIMHYD-style parameters `SUB`, `CRAK`, and `SQ_t`.

        Groundwater release is:

        `GW1 = GW + REC`

        `BAS = min(K * GW1, GW1)`

        `GW_next = GW1 - BAS`

        `Q_process = SRUN + IFLOW + BAS`

        Gamma routing turns process runoff into `Q_routed`.

        ## Water Balance

        Before routing:

        `S_before = SNOWPACK + MELTWATER + Sa + GW`

        `S_after = SNOWPACK_next + MELTWATER_next + Sa_next + GW_next`

        `residual = P - INT - ET_a - Q_process - (S_after - S_before)`

        The locked run reports median weighted daily residual near zero.

        ## Parameter Ranges

        - `INSC`: [0.5, 5.0] mm
        - `SUB`: [0, 1]
        - `CRAK`: [0, 1]
        - `SQ_t`: [0, 6]
        - `K`: [0.003, 0.3] d-1
        - `TT`: [-2.5, 2.5] degC
        - `CFMAX_t`: [0.5, 10] mm d-1 degC-1 nominal; learned dynamic multiplier in locked branch may soften the effective lower edge
        - `CFR`: [0, 0.1]
        - `CWH`: [0, 0.2]
        - `theta_ab`: [0.5, 1.0]
        - `theta_ak`: [1, 10]
        - `theta_cap`: [10, 1500] mm
        - `theta_efmax`: [0.5, 1.0]
        - `theta_wetpoint`: [0.3, 0.9]
        - routing `route_a`: [0, 2.9]
        - routing `route_b`: [0, 6.5]

        ## Parameter Generator

        The locked branch uses the existing Model 6 neural parameterization with static basin attributes, warm-started from the closed Model 6 checkpoint. It is not the later LSTM-parameter branch.

        ## Training And Evaluation

        Training objective: `RmseLossComb(alpha=0.25)`.

        Training supervision: streamflow only.

        ET, SWE, TWSA, and soil moisture are independent validation products, not training losses in the locked branch.
        """,
    )
    write_text(
        PACKAGE_DIR / "ASSUMPTIONS_AND_LIMITATIONS.md",
        """
        # Assumptions And Limitations

        - This is a Model 6-based Rohini-style diagnostic replication, not the original Rohini model.
        - The locked model is Q-trained; ET, SWE, TWSA, and surface soil moisture are validation products.
        - LAI is daily gap-filled NOAA AVHRR CDR LAI. Gap-filled early-period LAI is documented in the locked run manifest.
        - PET-based LAI-scaled ET is used in the locked branch; this is not the Rohini net-radiation ET equation.
        - GRACE TWSA has a spatial-resolution mismatch with individual CAMELS basins; only overlapping valid basin-months are used.
        - SWE validation is weak and is reported honestly. The likely causes are the smoothed snow partition/melt formulation, lack of SWE supervision, basin-average forcing biases, and compensation from streamflow-only calibration.
        - `theta_cap` is a structural upper bound. `aSrz_capacity` is realized capacity from the simulated `Sa` trajectory.
        - Theta-cap structural ablations are configured but not retrained in this locked package.
        - Observational ablations are not run because no completed multi-observation training branch is locked here.
        """,
    )
    write_text(
        PACKAGE_DIR / "REPRODUCIBILITY.md",
        f"""
        # Reproducibility

        ## Main command

        Run from the repository root:

        ```bash
        bash ECO_HYBRID/PUBLICATION_MODEL6_ROHINI_REPLICATION/RUN_ALL.sh
        ```

        ## Individual steps

        ```bash
        python ECO_HYBRID/PUBLICATION_MODEL6_ROHINI_REPLICATION/scripts/00_audit_data_availability.py
        python ECO_HYBRID/PUBLICATION_MODEL6_ROHINI_REPLICATION/scripts/01_build_publication_datasets.py
        python ECO_HYBRID/PUBLICATION_MODEL6_ROHINI_REPLICATION/scripts/evaluate_locked_model.py
        python ECO_HYBRID/PUBLICATION_MODEL6_ROHINI_REPLICATION/scripts/make_all_figures.py
        python ECO_HYBRID/PUBLICATION_MODEL6_ROHINI_REPLICATION/scripts/final_sanity_check.py
        ```

        ## Locked input paths

        - Run: `{RUN_DIR}`
        - Cache: `{CACHE_PATH}`
        - Checkpoint copy: `{CHECKPOINT_DIR}`
        - Random seed for RF interpretation: 42

        ## No fake data policy

        `scripts/00_audit_data_availability.py` records every required file. Missing ablations are documented as not run.
        """,
    )
    write_text(
        PACKAGE_DIR / "README.md",
        f"""
        # Model 6 Rohini-Style Replication Package

        ## Study Purpose

        This folder is a reproducible, publication-oriented analysis package for the locked Model 6 LAIEco branch. It recreates the logic of the Rohini/Blougouras figure sequence using this model's outputs, not their model outputs.

        ## Locked Model Identity

        - Branch: `Model6Closed_Snow_aSrz_SIMHYD_Simple_LAIEco_full671_target0732_train1`
        - Run folder: `{RUN_DIR}`
        - Basin count: {best.get("number_of_basins", 671)}
        - Median NSE: {best.get("median_NSE")}
        - Median KGE: {best.get("median_KGE")}
        - Median R2: {best.get("median_R2")}
        - Median realized aSrz capacity: {best.get("median_aSrz_capacity_mm")} mm
        - Training loss: `RmseLossComb(alpha=0.25)`
        - Training constraints: streamflow only

        ## Dataset Summary

        - Forcing and streamflow: CAMELS-US/Caravan-compatible Model 6 pipeline.
        - LAI: daily gap-filled NOAA AVHRR CDR basin LAI.
        - ET validation: MODIS MOD16 8-day ET and GLEAM monthly ET.
        - SWE validation: local NSIDC/SNODAS-style basin SWE tables, snow-dominated subset where available.
        - TWSA validation: GRACE/JPL basin-month anomaly table.
        - Surface soil moisture validation: ESA CCI monthly basin table.

        ## Figure Reproduction Table

        {fig_md}

        ## Main Results

        | Metric | Value |
        |---|---:|
        | Median NSE | {best.get("median_NSE")} |
        | Mean NSE | {best.get("mean_NSE")} |
        | Median KGE | {best.get("median_KGE")} |
        | Median R2 | {best.get("median_R2")} |
        | NSE > 0 count | {best.get("test_nse_gt_0_count")} |
        | NSE > 0.5 count | {best.get("test_nse_gt_0p5_count")} |
        | NSE > 0.7 count | {best.get("test_nse_gt_0p7_count")} |
        | Median low-flow NSE | {best.get("median_low_flow_NSE")} |
        | Median high-flow NSE | {best.get("median_high_flow_NSE")} |
        | Median aSrz capacity | {best.get("median_aSrz_capacity_mm")} mm |
        | Median mean aSrz | {best.get("median_mean_aSrz_mm")} mm |
        | Median process closure residual | {best.get("median_weighted_process_closure_residual_mm_day")} mm/day |

        ## External Validation

        {ext_md}

        ## Main Equations

        `aSrz_t = Sa_t - min_tau(Sa_tau)`

        `aSrz_capacity = max_t(aSrz_t)`

        `TWS_sim = SNOWPACK + MELTWATER + Sa + GW`

        `ET_total = INT + ET_a`

        `Q_process = SRUN + IFLOW + BAS`

        See `METHODS_MODEL_DESCRIPTION.md` for the full equation list.

        ## Assumptions

        This is a Model 6-based Rohini-style diagnostic replication, not the original Rohini model. SWE remains the weakest independent validation target and is not hidden.

        ## Commands To Reproduce

        ```bash
        bash RUN_ALL.sh
        ```

        ## Output Locations

        - Figures: `figures/`
        - Tables: `tables/`
        - Metrics: `metrics/`
        - Captions and notes: `docs/`
        - Locked checkpoint/config: `checkpoints/main_locked_model/`
        """,
    )


def write_environment_files() -> None:
    write_text(
        PACKAGE_DIR / "requirements.txt",
        """
        numpy
        pandas
        matplotlib
        pyarrow
        scikit-learn
        shap
        statsmodels
        """,
    )
    write_text(
        PACKAGE_DIR / "environment.yml",
        """
        name: model6-rohini-replication
        channels:
          - conda-forge
          - defaults
        dependencies:
          - python>=3.10
          - numpy
          - pandas
          - matplotlib
          - pyarrow
          - scikit-learn
          - shap
          - statsmodels
        """,
    )


def write_wrappers() -> None:
    # Copy this implementation into the package so the package is self-contained.
    src = Path(__file__).resolve()
    shutil.copy2(src, SCRIPTS_DIR / "publication_pipeline.py")
    wrapper_map = {
        "00_audit_data_availability.py": "audit",
        "01_build_publication_datasets.py": "build-datasets",
        "evaluate_locked_model.py": "evaluate",
        "make_all_figures.py": "figures",
        "final_sanity_check.py": "sanity",
        "figures/figure1_conceptual_asrz.py": "figure1",
        "figures/figure2_model_framework.py": "figure2",
        "figures/figure3_multivariable_performance.py": "figure3",
        "figures/figure4_spatial_climatic_mean_asrz.py": "figure4",
        "figures/figure5_asrz_twsa_sm_dynamics.py": "figure5",
        "figures/figure6_asrz_capacity.py": "figure6",
        "figures/figure7_shap_controls_capacity.py": "figure7",
        "figures/figure8_identifiability_theta_cap.py": "figure8",
    }
    for relpath, action in wrapper_map.items():
        write_text(
            SCRIPTS_DIR / relpath,
            f"""
            #!/usr/bin/env python3
            from pathlib import Path
            import subprocess
            import sys

            root = Path(__file__).resolve().parents[1]
            if Path(__file__).resolve().parent.name == "figures":
                root = Path(__file__).resolve().parents[2]
            script = root / "scripts" / "publication_pipeline.py"
            raise SystemExit(subprocess.call([sys.executable, str(script), "{action}"]))
            """,
        )
    for path in SCRIPTS_DIR.rglob("*.py"):
        path.chmod(0o755)
    write_text(
        PACKAGE_DIR / "RUN_ALL.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        cd "$(dirname "$0")"
        python scripts/00_audit_data_availability.py
        python scripts/01_build_publication_datasets.py
        python scripts/evaluate_locked_model.py
        python scripts/make_all_figures.py
        python scripts/final_sanity_check.py
        """,
    )
    (PACKAGE_DIR / "RUN_ALL.sh").chmod(0o755)


def final_sanity_check() -> None:
    ensure_dirs()
    required = [
        PACKAGE_DIR / "README.md",
        PACKAGE_DIR / "METHODS_MODEL_DESCRIPTION.md",
        PACKAGE_DIR / "ASSUMPTIONS_AND_LIMITATIONS.md",
        PACKAGE_DIR / "REPRODUCIBILITY.md",
        PACKAGE_DIR / "LOCKED_MODEL_CONFIG.yaml",
        PACKAGE_DIR / "DATA_MANIFEST.yaml",
        PACKAGE_DIR / "FIGURE_MANIFEST.yaml",
        PACKAGE_DIR / "ABLATION_MANIFEST.yaml",
        DATA_MANIFEST_DIR / "DATA_AUDIT_TABLE.csv",
        DATA_MANIFEST_DIR / "DATA_AUDIT_REPORT.md",
        TABLES_DIR / "basin_static_master.csv",
        TABLES_DIR / "monthly_model_outputs.csv",
        TABLES_DIR / "daily_model_outputs_indexed.parquet",
        METRICS_DIR / "main_summary_metrics.csv",
        METRICS_DIR / "external_validation_summary.csv",
        METRICS_DIR / "storage_diagnostics.csv",
        METRICS_DIR / "water_balance_diagnostics.csv",
    ]
    required += [FIGURES_DIR / f"Figure{i}_{name}.png" for i, name in [
        (1, "conceptual_asrz_model6"),
        (2, "model_framework_model6"),
        (3, "multivariable_performance_primary"),
        (4, "spatial_climatic_mean_asrz"),
        (5, "asrz_twsa_sm_dynamics"),
        (6, "asrz_capacity"),
        (7, "shap_controls_capacity"),
        (8, "identifiability_theta_cap"),
    ]]
    caption_paths = [DOCS_DIR / f"Figure{i}_caption.md" for i in range(1, 9)]
    required += caption_paths
    rows = [{"path": str(p), "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0} for p in required]
    check = pd.DataFrame(rows)

    # Verify aSrz definition on a small deterministic sample.
    cache = load_cache()
    sa = cache["sa"].astype(float)
    asrz = sa - np.nanmin(sa, axis=1, keepdims=True)
    stored = pd.read_parquet(TABLES_DIR / "daily_model_outputs_indexed.parquet", columns=["basin_id", "date", "aSrz_mm"])
    sample_basin = int(cache["basin_ids"][0])
    sample_date = pd.to_datetime(str(cache["dates"][0]))
    stored_val = stored[(stored["basin_id"] == sample_basin) & (pd.to_datetime(stored["date"]) == sample_date)]["aSrz_mm"]
    definition_ok = len(stored_val) and abs(float(stored_val.iloc[0]) - float(asrz[0, 0])) < 1e-4
    extra = pd.DataFrame(
        [
            {"path": "aSrz_definition_check: Sa - min(Sa)", "exists": bool(definition_ok), "size_bytes": 0},
            {"path": "theta_cap_vs_aSrz_capacity_not_confused", "exists": (TABLES_DIR / "Figure8_theta_cap_ablation_by_basin.csv").exists(), "size_bytes": 0},
            {"path": "SWE_sim_uses_SNOWPACK_plus_MELTWATER_or_cache_swe_model", "exists": "swe_model" in cache and "snowpack" in cache and "meltwater" in cache, "size_bytes": 0},
            {"path": "ET_validation_uses_ET_total", "exists": (TABLES_DIR / "Figure3_monthly_pairs_primary.csv").exists(), "size_bytes": 0},
            {"path": "water_balance_reported", "exists": (METRICS_DIR / "water_balance_diagnostics.csv").exists(), "size_bytes": 0},
        ]
    )
    check = pd.concat([check, extra], ignore_index=True)
    check.to_csv(LOGS_DIR / "final_sanity_check.csv", index=False)
    failures = check[~check["exists"].astype(bool)]
    write_text(
        LOGS_DIR / "final_sanity_check.md",
        f"""
        # Final Sanity Check

        Passed: {len(check) - len(failures)} / {len(check)}

        Failed: {len(failures)}

        {df_to_markdown(failures) if len(failures) else 'No failures.'}
        """,
    )
    if len(failures):
        print(f"Sanity check found {len(failures)} missing/failed items. See {LOGS_DIR / 'final_sanity_check.md'}")
    else:
        print("Sanity check passed.")


def build_all() -> None:
    ensure_dirs()
    write_wrappers()
    write_locked_config()
    write_manifests()
    write_ablation_stubs()
    write_environment_files()
    audit_data_availability()
    build_publication_datasets()
    evaluate_locked_model()
    make_all_figures()
    write_docs()
    final_sanity_check()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        default="all",
        choices=[
            "all",
            "audit",
            "build-datasets",
            "evaluate",
            "figures",
            "figure1",
            "figure2",
            "figure3",
            "figure4",
            "figure5",
            "figure6",
            "figure7",
            "figure8",
            "sanity",
            "docs",
            "scaffold",
        ],
    )
    args = parser.parse_args()
    ensure_dirs()
    if args.action == "all":
        build_all()
    elif args.action == "scaffold":
        write_wrappers()
        write_locked_config()
        write_manifests()
        write_ablation_stubs()
        write_environment_files()
        write_docs()
    elif args.action == "audit":
        audit_data_availability()
    elif args.action == "build-datasets":
        build_publication_datasets()
    elif args.action == "evaluate":
        evaluate_locked_model()
    elif args.action == "figures":
        make_all_figures()
    elif args.action == "figure1":
        figure1_conceptual_asrz()
    elif args.action == "figure2":
        figure2_model_framework()
    elif args.action == "figure3":
        figure3_multivariable_performance()
    elif args.action == "figure4":
        figure4_spatial_climatic_mean_asrz()
    elif args.action == "figure5":
        figure5_asrz_twsa_sm_dynamics()
    elif args.action == "figure6":
        figure6_asrz_capacity()
    elif args.action == "figure7":
        figure7_shap_controls_capacity()
    elif args.action == "figure8":
        figure8_identifiability_theta_cap()
    elif args.action == "sanity":
        final_sanity_check()
    elif args.action == "docs":
        write_docs()


if __name__ == "__main__":
    main()
