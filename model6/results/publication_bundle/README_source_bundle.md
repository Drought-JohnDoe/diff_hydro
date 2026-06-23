# Model 6 Rohini-Style Replication Package

        ## Study Purpose

        This folder is a reproducible, publication-oriented analysis package for the locked Model 6 LAIEco branch. It recreates the logic of the Rohini/Blougouras figure sequence using this model's outputs, not their model outputs.

        ## Locked Model Identity

        - Branch: `Model6Closed_Snow_aSrz_SIMHYD_Simple_LAIEco_full671_target0732_train1`
        - Run folder: `/home/mircore/Desktop/diff_hydro/ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_LAIEco_full671_target0732_train1`
        - Basin count: 671
        - Median NSE: 0.6913885772228241
        - Median KGE: 0.6584972999366892
        - Median R2: 0.7309911185485979
        - Median realized aSrz capacity: 171.4255828857422 mm
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

        | Figure   | Script                                                | Output file                                           | Required data                           | Status                                     |
| -------- | ----------------------------------------------------- | ----------------------------------------------------- | --------------------------------------- | ------------------------------------------ |
| Figure 1 | scripts/figures/figure1_conceptual_asrz.py            | figures/Figure1_conceptual_asrz_model6.png            | none                                    | complete                                   |
| Figure 2 | scripts/figures/figure2_model_framework.py            | figures/Figure2_model_framework_model6.png            | locked model config                     | complete                                   |
| Figure 3 | scripts/figures/figure3_multivariable_performance.py  | figures/Figure3_multivariable_performance_primary.png | Q, MODIS ET, SWE, GRACE                 | complete if audit finds products           |
| Figure 4 | scripts/figures/figure4_spatial_climatic_mean_asrz.py | figures/Figure4_spatial_climatic_mean_asrz.png        | aSrz, attributes                        | complete                                   |
| Figure 5 | scripts/figures/figure5_asrz_twsa_sm_dynamics.py      | figures/Figure5_asrz_twsa_sm_dynamics.png             | aSrz, GRACE, ESA CCI SM                 | complete                                   |
| Figure 6 | scripts/figures/figure6_asrz_capacity.py              | figures/Figure6_asrz_capacity.png                     | aSrz capacity, attributes               | complete                                   |
| Figure 7 | scripts/figures/figure7_shap_controls_capacity.py     | figures/Figure7_shap_controls_capacity.png            | aSrz capacity, P, PET, LAI, slope, sand | complete with SHAP or permutation fallback |
| Figure 8 | scripts/figures/figure8_identifiability_theta_cap.py  | figures/Figure8_identifiability_theta_cap.png         | current theta_cap; ablation configs     | current-only unless ablations are trained  |

        ## Main Results

        | Metric | Value |
        |---|---:|
        | Median NSE | 0.6913885772228241 |
        | Mean NSE | 0.43750624448038783 |
        | Median KGE | 0.6584972999366892 |
        | Median R2 | 0.7309911185485979 |
        | NSE > 0 count | 630 |
        | NSE > 0.5 count | 515 |
        | NSE > 0.7 count | 314 |
        | Median low-flow NSE | -33.48791313171387 |
        | Median high-flow NSE | 0.6081240177154541 |
        | Median aSrz capacity | 171.4255828857422 mm |
        | Median mean aSrz | 73.16185760498047 mm |
        | Median process closure residual | 0.0001823231141315773 mm/day |

        ## External Validation

        | product            | valid_basins | valid_pairs | median_corr        | median_R2          | median_NSE         | median_KGE         | median_RMSE        | median_RMSE_mm_per_8day | median_zRMSE       | median_bias             | median_pbias_pct   | pooled_corr        | pooled_r2          | pooled_nse         | pooled_kge         | pooled_rmse       | pooled_zrmse       | pooled_bias            | pooled_pbias          | median_RMSE_mm_per_month | median_RMSE_mm     |
| ------------------ | ------------ | ----------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ----------------------- | ------------------ | ----------------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ----------------- | ------------------ | ---------------------- | --------------------- | ------------------------ | ------------------ |
| MODIS 8-day ET     | 671          | 269742      | 0.807833636883675  | 0.6525951848807053 | 0.5420064087298602 | 0.6692771724650368 | 6.114788412524411  | 6.114788412524411       | 0.6767522377282101 | 0.2337813017512923      | 1.3707130238039544 | 0.7622851017648606 | 0.5810785763726639 | 0.5363153714716921 | 0.76072942023579   | 6.490933727056787 | 0.6809439246577562 | 0.0903575724676683     | 0.6891618453153645    |                          |                    |
| GLEAM monthly ET   | 457          | 47985       | 0.8883255818828043 | 0.7891223394274228 | 0.6768004180625578 | 0.7662889934479731 | 15.86395324110984  |                         | 0.5685064484572204 | 0.5177851793982267      | 0.9535951282333436 | 0.8551693127595342 | 0.731314553485614  | 0.6593420965564503 | 0.8097254052606984 | 17.8585916848223  | 0.5836590643890915 | 0.3052898750611825     | 0.5944169003936569    | 15.86395324110984        |                    |
| GRACE monthly TWSA | 164          | 16236       | 0.7073789319832189 | 0.502698772724924  | 0.347644896744578  | 0.4670937246008294 | 56.413342232566144 |                         | 0.8076849956218715 | -1.3816108750890837e-15 |                    | 0.65101819035082   | 0.4238246841676564 | 0.3303821373993036 | 0.6483424583442715 | 70.89485611393933 | 0.8183018163249404 | 6.8515890164356995e-09 | 7.039278564753324e-07 |                          | 56.413342232566144 |

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
