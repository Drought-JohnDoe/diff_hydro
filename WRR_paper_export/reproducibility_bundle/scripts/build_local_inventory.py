from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from paper_common import load_config


def describe_table(path: Path) -> dict:
    row = {
        "path": str(path),
        "file_type": path.suffix.lower().lstrip("."),
        "variables": "",
        "units_if_known": "",
        "date_range": "",
        "spatial_resolution": "",
        "temporal_resolution": "",
        "count": "",
        "missingness": "",
        "role": "",
        "notes": "",
    }
    if not path.exists():
        row["notes"] = "missing"
        return row
    try:
        if path.suffix.lower() == ".csv":
            full = pd.read_csv(path)
            df = full.head(50)
            if len(full) == 0:
                row["notes"] = "empty_table"
        elif path.suffix.lower() == ".parquet":
            full = pd.read_parquet(path)
            df = full.head(50)
            if len(full) == 0:
                row["notes"] = "empty_table"
        else:
            return row
        row["variables"] = ", ".join(df.columns.tolist())
        if "date" in df.columns:
            try:
                dates = pd.to_datetime(df["date"], errors="coerce")
                if dates.notna().any():
                    row["date_range"] = f"{dates.min().date()} to {dates.max().date()}"
            except Exception:
                pass
        if "year_month" in df.columns:
            dates = pd.to_datetime(df["year_month"], errors="coerce")
            if dates.notna().any():
                row["date_range"] = f"{dates.min().date()} to {dates.max().date()}"
        if "basin_id" in df.columns:
            row["count"] = str(df["basin_id"].nunique()) + " basins (sampled)"
        elif "gauge_id" in df.columns:
            row["count"] = str(df["gauge_id"].nunique()) + " basins (sampled)"
        miss = float(df.isna().mean().mean()) if len(df.columns) > 0 else 0.0
        row["missingness"] = f"{miss:.3f}"
    except Exception as exc:
        row["notes"] = f"read_error: {exc}"
    return row


def main():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "wrr_model6_config.yaml")
    inventory_items = [
        ("model_training", Path(cfg["model"]["main_run_dir"]), "run dir", "main model benchmark"),
        ("model_output", Path(cfg["model"]["ep100_metrics_csv"]), "evaluation only", "streamflow + learned state summary"),
        ("model_output", Path(cfg["model"]["ep100_water_balance_csv"]), "evaluation only", "water-balance summary"),
        ("model_output", Path(cfg["model"]["ep60_aux_daily_archive"]), "evaluation only", "state-rich auxiliary daily archive"),
        ("basin_data", Path(cfg["basins"]["attrs_dir"]) / "camels_clim.txt", "static attrs", "CAMELS climate"),
        ("basin_data", Path(cfg["basins"]["attrs_dir"]) / "camels_geol.txt", "static attrs", "CAMELS geology"),
        ("basin_data", Path(cfg["basins"]["attrs_dir"]) / "camels_hydro.txt", "static attrs", "CAMELS hydro"),
        ("basin_data", Path(cfg["basins"]["attrs_dir"]) / "camels_soil.txt", "static attrs", "CAMELS soil"),
        ("basin_data", Path(cfg["basins"]["attrs_dir"]) / "camels_topo.txt", "static attrs", "CAMELS topo"),
        ("basin_data", Path(cfg["basins"]["attrs_dir"]) / "camels_vege.txt", "static attrs", "CAMELS vegetation"),
        ("basin_shapes", Path(cfg["basins"]["shapefile"]), "evaluation only", "CAMELS basin polygons"),
        ("independent_et", Path(cfg["independent_products"]["gleam_daily_csv"]), "evaluation only", "GLEAM ET"),
        ("independent_et", Path(cfg["independent_products"]["gleam_monthly_csv"]), "evaluation only", "GLEAM ET monthly"),
        ("independent_et", Path(cfg["independent_products"]["fluxcom_monthly_parquet"]), "evaluation only", "FLUXCOM ET basin-month"),
        ("independent_et", Path(cfg["independent_products"]["mod16_annual_csv"]), "evaluation only", "MOD16 annual ET cross-check"),
        ("independent_twsa", Path(cfg["independent_products"]["grace_jpl_nc"]), "evaluation only", "JPL GRACE mascon"),
        ("independent_twsa", Path(cfg["independent_products"]["grace_jpl_basin_monthly_parquet"]), "evaluation only", "JPL basin-month exploratory extraction"),
        ("independent_twsa", Path(cfg["independent_products"]["grace_jpl_metrics_csv"]), "evaluation only", "JPL basin metrics"),
        ("independent_twsa", Path(cfg["independent_products"]["esa_cci_sm_dir"]), "evaluation only", "ESA CCI soil moisture"),
        ("independent_swe", Path(cfg["independent_products"]["swe_validation_dir"]), "evaluation only", "NSIDC SWE validation"),
    ]
    optional_missing = [
        ("independent_et", cfg["independent_products"].get("gleam_daily_csv"), "evaluation only", "GLEAM daily ET placeholder"),
        ("independent_et", cfg["independent_products"].get("gleam_monthly_csv"), "evaluation only", "GLEAM monthly ET placeholder"),
        ("independent_twsa", cfg["independent_products"].get("grace_csr_path"), "evaluation only", "GRACE CSR placeholder"),
        ("independent_twsa", cfg["independent_products"].get("grace_gsfc_path"), "evaluation only", "GRACE GSFC placeholder"),
    ]
    rows = []
    for category, path, role, note in inventory_items:
        desc = describe_table(path)
        desc["category"] = category
        desc["role"] = role
        desc["notes"] = (desc["notes"] + "; " if desc["notes"] else "") + note
        rows.append(desc)
    for category, path, role, note in optional_missing:
        if path is None:
            rows.append(
                {
                    "category": category,
                    "path": "null",
                    "file_type": "",
                    "variables": "",
                    "units_if_known": "",
                    "date_range": "",
                    "spatial_resolution": "",
                    "temporal_resolution": "",
                    "count": "",
                    "missingness": "",
                    "role": role,
                    "notes": f"missing_configured_path; {note}",
                }
            )
    out_csv = Path(cfg["outputs"]["inventory_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    out_md = Path(cfg["outputs"]["inventory_md"])
    lines = ["# Local Data Inventory", "", f"Generated from `{cfg['project']['name']}` local paths.", ""]
    for row in rows:
        lines.append(f"## {row['category']}: `{Path(row['path']).name}`")
        for k in ["path", "file_type", "variables", "date_range", "missingness", "role", "notes"]:
            lines.append(f"- {k}: {row[k]}")
        lines.append("")
    out_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
