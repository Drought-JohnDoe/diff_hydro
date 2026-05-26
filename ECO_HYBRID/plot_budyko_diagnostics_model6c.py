import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_DIR = Path(
    "/home/mircore/Desktop/diff_hydro/ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep30"
)
FIG_DIR = RUN_DIR / "figures" / "budyko_diagnostics"
OUT_CSV = RUN_DIR / "budyko_diagnostics_671.csv"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def find_first_column(df: pd.DataFrame, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        lc = cand.lower()
        if lc in lower_map:
            return lower_map[lc]
    return None


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").astype(float)


def rankdata_average(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def pearson_corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    xx = x[mask]
    yy = y[mask]
    if np.std(xx) == 0 or np.std(yy) == 0:
        return np.nan
    return float(np.corrcoef(xx, yy)[0, 1])


def spearman_corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    rx = rankdata_average(x[mask])
    ry = rankdata_average(y[mask])
    return pearson_corr(rx, ry)


def fu_budyko(phi, omega):
    phi = np.asarray(phi, dtype=float)
    phi = np.clip(phi, 0.0, None)
    omega = max(float(omega), 1.0001)
    return 1.0 + phi - np.power(1.0 + np.power(phi, omega), 1.0 / omega)


def fit_fu_omega(phi, et_over_p):
    mask = np.isfinite(phi) & np.isfinite(et_over_p) & (phi >= 0)
    x = phi[mask]
    y = et_over_p[mask]
    omega_grid = np.concatenate([
        np.linspace(1.01, 3.0, 800),
        np.linspace(3.01, 8.0, 500),
    ])
    best_omega = np.nan
    best_rmse = np.inf
    for omega in omega_grid:
        yhat = fu_budyko(x, omega)
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_omega = float(omega)
    return best_omega, best_rmse


def main():
    ensure_dir(FIG_DIR)

    per_basin_path = RUN_DIR / "per_basin_metrics_compare.csv"
    meta_path = RUN_DIR / "basin_metadata.csv"

    df = pd.read_csv(per_basin_path)
    if "model" in df.columns:
        model_mask = df["model"].astype(str).str.contains("Model6Closed_Snow_aSrz_SIMHYD_Simple", na=False)
        df = df.loc[model_mask].copy()
    if df.empty:
        raise RuntimeError("No rows for Model6Closed_Snow_aSrz_SIMHYD_Simple found in per_basin_metrics_compare.csv")

    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        if "basin_id" in meta.columns and "basin_id" in df.columns:
            df = df.merge(meta, on="basin_id", how="left", suffixes=("", "_meta"))
            if "lat_meta" in df.columns:
                df["lat"] = df["lat"].fillna(df["lat_meta"])
            if "lon_meta" in df.columns:
                df["lon"] = df["lon"].fillna(df["lon_meta"])

    et_col = find_first_column(df, ["ET/P", "ET_P", "et_p", "ET_over_P"])
    q_col = find_first_column(df, ["Q/P", "Q_P", "q_p", "Q_over_P"])
    nse_col = find_first_column(df, ["NSE", "nse"])
    kge_col = find_first_column(df, ["KGE", "kge"])
    aridity_col = find_first_column(df, ["aridity", "PET/P", "pet_p", "aridity_index"])
    pmean_col = find_first_column(df, ["p_mean"])
    petmean_col = find_first_column(df, ["pet_mean"])

    if et_col is None or q_col is None:
        raise RuntimeError("Could not find ET/P and Q/P columns.")

    if aridity_col is None:
        if pmean_col is not None and petmean_col is not None:
            df["aridity_derived"] = safe_numeric(df[petmean_col]) / np.clip(safe_numeric(df[pmean_col]), 1e-6, None)
            aridity_col = "aridity_derived"
        else:
            raise RuntimeError("Could not find aridity column or p_mean/pet_mean to derive it.")

    df["aridity_index"] = safe_numeric(df[aridity_col])
    df["ET_over_P_clean"] = safe_numeric(df[et_col])
    df["Q_over_P_clean"] = safe_numeric(df[q_col])
    df["NSE_clean"] = safe_numeric(df[nse_col]) if nse_col else np.nan
    df["KGE_clean"] = safe_numeric(df[kge_col]) if kge_col else np.nan

    plot_df = df[np.isfinite(df["aridity_index"]) & np.isfinite(df["ET_over_P_clean"]) & np.isfinite(df["Q_over_P_clean"])].copy()
    if plot_df.empty:
        raise RuntimeError("No valid rows remain after filtering aridity, ET/P, and Q/P.")

    phi = plot_df["aridity_index"].to_numpy(dtype=float)
    etp = plot_df["ET_over_P_clean"].to_numpy(dtype=float)
    qp = plot_df["Q_over_P_clean"].to_numpy(dtype=float)

    omega_default = 2.0
    budyko_default = fu_budyko(phi, omega_default)
    rmse_default = float(np.sqrt(np.mean((etp - budyko_default) ** 2)))
    omega_fit, rmse_fit = fit_fu_omega(phi, etp)
    budyko_fit = fu_budyko(phi, omega_fit)
    budyko_residual = etp - budyko_fit

    plot_df["budyko_default_et_over_p"] = budyko_default
    plot_df["budyko_fitted_et_over_p"] = budyko_fit
    plot_df["budyko_residual"] = budyko_residual
    plot_df.to_csv(OUT_CSV, index=False)

    xline = np.linspace(0.0, max(4.0, float(np.nanmax(phi) * 1.05)), 400)
    default_line = fu_budyko(xline, omega_default)
    fit_line = fu_budyko(xline, omega_fit)

    median_et = float(np.nanmedian(etp))
    median_q = float(np.nanmedian(qp))

    color_metric = plot_df["NSE_clean"].to_numpy(dtype=float) if nse_col else plot_df["KGE_clean"].to_numpy(dtype=float)
    color_label = "NSE" if nse_col else "KGE"

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    sc = ax.scatter(phi, etp, c=color_metric, cmap="viridis", s=34, edgecolors="black", linewidths=0.25, alpha=0.9)
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(color_label)
    ax.plot(xline, np.ones_like(xline), color="gray", linestyle="--", linewidth=1.4, label="Water limit: ET/P = 1")
    ax.plot(xline, xline, color="dimgray", linestyle=":", linewidth=1.4, label="Energy limit: ET/P = PET/P")
    ax.plot(xline, default_line, color="#d95f02", linewidth=2.0, label="Fu Budyko (omega = 2.0)")
    ax.plot(xline, fit_line, color="#1b9e77", linewidth=2.2, label=f"Fu fit (omega = {omega_fit:.3f})")
    ax.set_xlabel("Aridity index (PET/P)")
    ax.set_ylabel("Evaporative ratio (ET/P)")
    ax.set_title("Model6Closed_Snow_aSrz_SIMHYD_Simple Budyko diagnostic")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=9, loc="lower right")
    ax.text(0.02, 0.98, f"Median ET/P = {median_et:.3f}\nMedian Q/P = {median_q:.3f}\nBest-fit omega = {omega_fit:.3f}\nBudyko RMSE = {rmse_fit:.4f}", transform=ax.transAxes, va="top", ha="left", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "budyko_scatter_model6c.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.hist(budyko_residual, bins=30, color="#3182bd", edgecolor="black", alpha=0.85)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Budyko residual = model ET/P - fitted Fu ET/P")
    ax.set_ylabel("Basins")
    ax.set_title("Budyko residual histogram")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "budyko_residual_histogram.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.scatter(phi, budyko_residual, c=color_metric, cmap="viridis", s=32, edgecolors="black", linewidths=0.2, alpha=0.9)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    ax.set_xlabel("Aridity index (PET/P)")
    ax.set_ylabel("Budyko residual")
    ax.set_title("Budyko residual vs aridity")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "budyko_residual_vs_aridity.png", dpi=220)
    plt.close(fig)

    lat_col = find_first_column(plot_df, ["lat"])
    lon_col = find_first_column(plot_df, ["lon"])
    map_warning = None
    if lat_col is not None and lon_col is not None and np.isfinite(plot_df[lat_col]).any() and np.isfinite(plot_df[lon_col]).any():
        fig, ax = plt.subplots(figsize=(10.2, 5.8))
        vmax = float(np.nanpercentile(np.abs(budyko_residual), 95))
        vmax = max(vmax, 0.05)
        sc = ax.scatter(plot_df[lon_col], plot_df[lat_col], c=budyko_residual, cmap="coolwarm", vmin=-vmax, vmax=vmax, s=30, edgecolors="black", linewidths=0.2)
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Budyko residual")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("CONUS Budyko residual map")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "budyko_residual_map_671.png", dpi=220)
        plt.close(fig)
    else:
        map_warning = "Latitude/longitude unavailable; skipped Budyko residual map."

    summary_stats = {
        "median_ET_over_P": median_et,
        "median_Q_over_P": median_q,
        "pearson_aridity_vs_ET_over_P": pearson_corr(phi, etp),
        "spearman_aridity_vs_ET_over_P": spearman_corr(phi, etp),
        "pearson_aridity_vs_Q_over_P": pearson_corr(phi, qp),
        "spearman_aridity_vs_Q_over_P": spearman_corr(phi, qp),
        "rmse_fu_omega_2": rmse_default,
        "rmse_fu_fitted": rmse_fit,
        "best_fit_omega": omega_fit,
        "n_ET_over_P_gt_1": int(np.sum(etp > 1.0)),
        "n_ET_over_P_gt_PET_over_P": int(np.sum(etp > phi)),
        "n_basins": int(len(plot_df)),
    }
    (FIG_DIR / "budyko_summary_stats.json").write_text(json.dumps(summary_stats, indent=2))

    print("Budyko diagnostics for Model6Closed_Snow_aSrz_SIMHYD_Simple (671 basins)")
    for key, val in summary_stats.items():
        print(f"- {key}: {val}")
    if map_warning:
        print(f"- warning: {map_warning}")


if __name__ == "__main__":
    main()
