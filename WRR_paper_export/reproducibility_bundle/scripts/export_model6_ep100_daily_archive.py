from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from paper_common import ensure_outputs, load_camels_attributes, load_config, write_json


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "wrr_model6_config.yaml")
    ensure_outputs(cfg)

    aux_daily = pd.read_parquet(cfg["model"]["ep60_aux_daily_archive"])
    aux_daily["date"] = pd.to_datetime(aux_daily["date"])

    validation_daily = pd.read_csv(
        "/home/mircore/Desktop/diff_hydro/outputs/Model6_ExternalValidation/model6_validation_products_daily.csv"
    )
    validation_daily["date"] = pd.to_datetime(validation_daily["date"])
    validation_daily["basin_id"] = validation_daily["basin_id"].astype(int)

    daily = aux_daily.merge(
        validation_daily.rename(
            columns={
                "total_ET_model": "ET_model_total",
                "actual_ET": "ET_independent_aux",
                "interception_evaporation": "INT_validation",
                "snowpack": "SNOWPACK",
                "meltwater": "MELTWATER",
                "soil_moisture": "soil_moisture_validation",
                "soil_moisture_relative": "soil_moisture_relative_validation",
                "total_discharge": "total_discharge_validation",
                "precipitation": "precipitation_validation",
            }
        ),
        on=["basin_id", "date"],
        how="left",
    )

    daily["INT_effective"] = daily["INT"].where(daily["INT"].notna(), daily["INT_validation"])
    daily["ET_model"] = daily["ET_model_total"].where(daily["ET_model_total"].notna(), daily["INT_effective"] + daily["ET_a"])
    daily["TWS_model_proxy"] = daily["SNOWPACK"] + daily["MELTWATER"] + daily["Sa"] + daily["GW"]
    daily["storage_proxy_no_snow"] = daily["Sa"] + daily["GW"]
    daily["archive_source_main"] = "ep100_main_benchmark"
    daily["archive_source_states"] = "ep60_auxiliary_daily_archive"
    daily["archive_source_snow_et"] = "model6_external_validation_daily_products"

    out_daily = Path(cfg["outputs"]["diagnostics_dir"]) / "model6_ep100_daily_basin_day.parquet"
    daily.to_parquet(out_daily, index=False)

    monthly_sum_cols = [
        "Q_obs",
        "Q_process",
        "P",
        "INT_effective",
        "INT",
        "ET_a",
        "ET_model",
        "REC",
        "BAS",
        "SRUN",
        "IFLOW",
        "total_discharge_validation",
        "precipitation_validation",
    ]
    monthly_mean_cols = [
        "Sa",
        "GW",
        "SNOWPACK",
        "MELTWATER",
        "theta_cap",
        "theta_wetpoint",
        "alpha",
        "soil_moisture_validation",
        "soil_moisture_relative_validation",
        "Smoist",
        "x_norm",
        "TWS_model_proxy",
        "storage_proxy_no_snow",
    ]
    tmp = daily.copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    tmp["date_month"] = tmp["date"].dt.to_period("M").dt.to_timestamp("M")
    monthly_sum = tmp.groupby(["basin_id", "date_month"], dropna=False)[monthly_sum_cols].sum(min_count=1).reset_index()
    monthly_mean = tmp.groupby(["basin_id", "date_month"], dropna=False)[monthly_mean_cols].mean().reset_index()
    monthly = monthly_sum.merge(monthly_mean, on=["basin_id", "date_month"], how="outer").rename(columns={"date_month": "date"})
    monthly.to_parquet(Path(cfg["outputs"]["diagnostics_dir"]) / "model6_ep100_monthly_basin.parquet", index=False)

    ep100 = pd.read_csv(cfg["model"]["ep100_metrics_csv"])
    ep100 = ep100.loc[ep100["model"] == "Model6Closed_Snow_aSrz_SIMHYD_Simple"].copy()
    theta = pd.read_csv(cfg["model"]["ep60_theta_diagnostics_csv"])
    theta = theta.rename(
        columns={
            "theta_wetpoint_weighted": "theta_wetpoint_weighted_ep60",
            "theta_wetpoint_mean": "theta_wetpoint_mean_ep60",
            "theta_cap_mean": "theta_cap_mean_ep60",
        }
    )
    ppt = pd.read_csv(
        "/home/mircore/Desktop/diff_hydro/ECO_HYBRID/PPT/tables/model6closed_ep60_ppt_diagnostics.csv"
    )
    keep_ppt = [
        "basin_id",
        "K_weighted",
        "K_mean",
        "theta_ab_weighted",
        "theta_ak_weighted",
        "theta_efmax_weighted",
        "theta_cap_weighted",
        "theta_wetpoint_weighted",
        "component_weight_1",
        "component_weight_2",
        "component_weight_3",
        "component_weight_4",
    ]
    ppt = ppt[keep_ppt].copy()
    attrs = load_camels_attributes(cfg)
    params = ep100.merge(
        theta[
            [
                "basin_id",
                "theta_wetpoint_weighted_ep60",
                "theta_wetpoint_mean_ep60",
                "component_weight_sum",
                "theta_wetpoint_comp_1",
                "theta_wetpoint_comp_2",
                "theta_wetpoint_comp_3",
                "theta_wetpoint_comp_4",
                "theta_cap_comp_1",
                "theta_cap_comp_2",
                "theta_cap_comp_3",
                "theta_cap_comp_4",
            ]
        ],
        on="basin_id",
        how="left",
    ).merge(ppt, on="basin_id", how="left").merge(attrs, on="basin_id", how="left")
    params["component_entropy_ep60"] = 0.0
    for c in ["component_weight_1", "component_weight_2", "component_weight_3", "component_weight_4"]:
        arr = np.clip(params[c].fillna(0.0), 1e-12, 1.0)
        params["component_entropy_ep60"] += -(arr * np.log(arr))
    params["parameter_source_note"] = "Ep100 benchmark metrics merged with Ep60 auxiliary parameter diagnostics"
    params.to_csv(Path(cfg["outputs"]["diagnostics_dir"]) / "model6_ep100_learned_parameters.csv", index=False)

    write_json(
        Path(cfg["outputs"]["diagnostics_dir"]) / "model6_ep100_archive_manifest.json",
        {
            "daily_rows": int(len(daily)),
            "monthly_rows": int(len(monthly)),
            "n_basins_daily": int(daily["basin_id"].nunique()),
            "date_start_daily": str(daily["date"].min().date()),
            "date_end_daily": str(daily["date"].max().date()),
            "state_source": "Ep60 auxiliary state archive",
            "snow_et_source": "local Model6 external validation daily products",
            "main_benchmark_source": "Ep100 saved benchmark outputs",
        },
    )


if __name__ == "__main__":
    main()

