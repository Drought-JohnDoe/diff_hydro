# Local Data Inventory

Generated from `WRR_Model6_EndToEnd_Paper` local paths.

## model_training: `Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep100_iter200`
- path: /home/mircore/Desktop/diff_hydro/ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep100_iter200
- file_type: 
- variables: 
- date_range: 
- missingness: 
- role: run dir
- notes: main model benchmark

## model_output: `per_basin_metrics_compare.csv`
- path: /home/mircore/Desktop/diff_hydro/ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep100_iter200/per_basin_metrics_compare.csv
- file_type: csv
- variables: model, basin_id, lat, lon, NSE, KGE, R2, FLV, FHV, low_flow_NSE, high_flow_NSE, ET_over_P, INT_over_P, Q_over_P, SRUN_over_P, IFLOW_over_P, BAS_over_P, REC_over_P, external_loss_over_P, mean_abs_daily_wb_residual_mm_day, cumulative_relative_wb_error, SNOWPACK_drift_mm, MELTWATER_drift_mm, Sa_drift_mm, GW_drift_mm, alpha_mean, theta_cap_mean, aSrz_capacity_mm, mean_aSrz_mm, aridity_index
- date_range: 
- missingness: 0.167
- role: evaluation only
- notes: streamflow + learned state summary

## model_output: `water_balance_compare.csv`
- path: /home/mircore/Desktop/diff_hydro/ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep100_iter200/water_balance_compare.csv
- file_type: csv
- variables: model, basin_id, cumulative_precipitation_mm, mean_abs_daily_wb_residual_mm_day, max_abs_daily_wb_residual_mm_day, cumulative_relative_wb_error, external_loss_over_P
- date_range: 
- missingness: 0.000
- role: evaluation only
- notes: water-balance summary

## model_output: `drought_tipping_daily_basin_day.parquet`
- path: /home/mircore/Desktop/diff_hydro/ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep60/drought_tipping_theta_wetpoint/drought_tipping_daily_basin_day.parquet
- file_type: parquet
- variables: basin_id, date, Q_obs, Q_process, dQ_dt, Sa, theta_cap, Smoist, theta_wetpoint, x_norm, alpha, ET_a, REC, BAS, GW, gw_balance, P, INT, SRUN, IFLOW, drought_day, onset_window
- date_range: 1995-10-01 to 1995-11-19
- missingness: 0.001
- role: evaluation only
- notes: state-rich auxiliary daily archive

## basin_data: `camels_clim.txt`
- path: /home/mircore/Desktop/diff_hydro/ECO_HYBRID/ECO_INPUTS_2020/camels_us/camels_attributes_v2/camels_clim.txt
- file_type: txt
- variables: 
- date_range: 
- missingness: 
- role: static attrs
- notes: CAMELS climate

## basin_data: `camels_geol.txt`
- path: /home/mircore/Desktop/diff_hydro/ECO_HYBRID/ECO_INPUTS_2020/camels_us/camels_attributes_v2/camels_geol.txt
- file_type: txt
- variables: 
- date_range: 
- missingness: 
- role: static attrs
- notes: CAMELS geology

## basin_data: `camels_hydro.txt`
- path: /home/mircore/Desktop/diff_hydro/ECO_HYBRID/ECO_INPUTS_2020/camels_us/camels_attributes_v2/camels_hydro.txt
- file_type: txt
- variables: 
- date_range: 
- missingness: 
- role: static attrs
- notes: CAMELS hydro

## basin_data: `camels_soil.txt`
- path: /home/mircore/Desktop/diff_hydro/ECO_HYBRID/ECO_INPUTS_2020/camels_us/camels_attributes_v2/camels_soil.txt
- file_type: txt
- variables: 
- date_range: 
- missingness: 
- role: static attrs
- notes: CAMELS soil

## basin_data: `camels_topo.txt`
- path: /home/mircore/Desktop/diff_hydro/ECO_HYBRID/ECO_INPUTS_2020/camels_us/camels_attributes_v2/camels_topo.txt
- file_type: txt
- variables: 
- date_range: 
- missingness: 
- role: static attrs
- notes: CAMELS topo

## basin_data: `camels_vege.txt`
- path: /home/mircore/Desktop/diff_hydro/ECO_HYBRID/ECO_INPUTS_2020/camels_us/camels_attributes_v2/camels_vege.txt
- file_type: txt
- variables: 
- date_range: 
- missingness: 
- role: static attrs
- notes: CAMELS vegetation

## basin_shapes: `basinset_gf_nhru.shp`
- path: /home/mircore/Desktop/diff_hydro/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/shapefiles/merge/basinset_gf_nhru.shp
- file_type: shp
- variables: 
- date_range: 
- missingness: 
- role: evaluation only
- notes: CAMELS basin polygons

## independent_et: `gleam_basin_daily.csv`
- path: /home/mircore/Desktop/diff_hydro/outputs/Model6_ExternalValidation/gleam_basin_daily.csv
- file_type: csv
- variables: basin_id, date, product, variable, value
- date_range: 
- missingness: nan
- role: evaluation only
- notes: empty_table; GLEAM ET

## independent_et: `gleam_basin_monthly.csv`
- path: /home/mircore/Desktop/diff_hydro/outputs/Model6_ExternalValidation/gleam_basin_monthly.csv
- file_type: csv
- variables: basin_id, product, variable, year_month, value, temporal_aggregation
- date_range: 
- missingness: nan
- role: evaluation only
- notes: empty_table; GLEAM ET monthly

## independent_et: `fluxcom_et_basin_monthly.parquet`
- path: /home/mircore/Desktop/diff_hydro/ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep100_iter200/independent_validation_grace_fluxcom/fluxcom_et_basin_monthly.parquet
- file_type: parquet
- variables: basin_id, fluxcom_et_mm_month, valid_pixel_fraction, date
- date_range: 2001-01-31 to 2001-01-31
- missingness: 0.380
- role: evaluation only
- notes: FLUXCOM ET basin-month

## independent_et: `model6_vs_mod16a3gf_annual_et_by_basin.csv`
- path: /home/mircore/Desktop/diff_hydro/outputs/Model6_ExternalValidation/MOD16A3GF_Model6_ET/model6_vs_mod16a3gf_annual_et_by_basin.csv
- file_type: csv
- variables: basin_id, gage_name, lat, lon, NSE, KGE, n_years, et_corr, et_rmse_mm_yr, et_bias_mm_yr, et_pbias_pct, et_kge, model6_et_mean_mm_yr, mod16_et_mean_mm_yr
- date_range: 
- missingness: 0.000
- role: evaluation only
- notes: MOD16 annual ET cross-check

## independent_twsa: `GRCTellus.JPL.200204_202603.GLO.RL06.3M.MSCNv04CRI.nc`
- path: /home/mircore/Desktop/diff_hydro/ECO_HYBRID/ECO_INPUTS_2020/grace_jpl/raw/GRCTellus.JPL.200204_202603.GLO.RL06.3M.MSCNv04CRI.nc
- file_type: nc
- variables: 
- date_range: 
- missingness: 
- role: evaluation only
- notes: JPL GRACE mascon

## independent_twsa: `grace_twsa_basin_monthly.parquet`
- path: /home/mircore/Desktop/diff_hydro/ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep100_iter200/independent_validation_grace_fluxcom/grace_twsa_basin_monthly.parquet
- file_type: parquet
- variables: basin_id, grace_twsa_mm, valid_pixel_fraction, date
- date_range: 2002-04-30 to 2002-04-30
- missingness: 0.380
- role: evaluation only
- notes: JPL basin-month exploratory extraction

## independent_twsa: `grace_twsa_per_basin_metrics.csv`
- path: /home/mircore/Desktop/diff_hydro/ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep100_iter200/independent_validation_grace_fluxcom/grace_twsa_per_basin_metrics.csv
- file_type: csv
- variables: basin_id, n_obs, R2, NSE, KGE, lat, lon, area_km2
- date_range: 
- missingness: 0.285
- role: evaluation only
- notes: JPL basin metrics

## independent_twsa: `esacci_sm`
- path: /home/mircore/Desktop/diff_hydro/results/validation/raw/esacci_sm
- file_type: 
- variables: 
- date_range: 
- missingness: 
- role: evaluation only
- notes: ESA CCI soil moisture

## independent_swe: `NSIDC0719_Model6_SWE`
- path: /home/mircore/Desktop/diff_hydro/outputs/Model6_ExternalValidation/NSIDC0719_Model6_SWE
- file_type: 
- variables: 
- date_range: 
- missingness: 
- role: evaluation only
- notes: NSIDC SWE validation

## independent_twsa: `null`
- path: null
- file_type: 
- variables: 
- date_range: 
- missingness: 
- role: evaluation only
- notes: missing_configured_path; GRACE CSR placeholder

## independent_twsa: `null`
- path: null
- file_type: 
- variables: 
- date_range: 
- missingness: 
- role: evaluation only
- notes: missing_configured_path; GRACE GSFC placeholder

