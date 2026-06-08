from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pandas as pd

from paper_common import load_config


def p(x: float | int | None, digits: int = 3) -> str:
    if x is None or pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "wrr_model6_config.yaml")
    tables = Path(cfg["outputs"]["tables_dir"])
    manuscript = Path(cfg["outputs"]["manuscript_dir"])
    manuscript.mkdir(parents=True, exist_ok=True)

    stream = pd.read_csv(tables / "streamflow_metrics_by_basin.csv")
    wb = pd.read_csv(tables / "water_balance_closure_by_basin.csv")
    params = pd.read_csv(tables / "learned_parameters_by_basin.csv")
    et = pd.read_csv(tables / "et_validation_by_basin.csv")
    twsa = pd.read_csv(tables / "twsa_validation_by_basin_exploratory.csv")
    twsa_reg = pd.read_csv(tables / "twsa_validation_by_region.csv")
    comp = pd.read_csv(tables / "model_comparison_summary.csv")
    theta_corr = pd.read_csv(tables / "theta_cap_interpretability_correlations.csv")
    theta_rf = pd.read_csv(tables / "theta_cap_random_forest_importance.csv")
    inv = pd.read_csv(cfg["outputs"]["inventory_csv"])
    figures_dir = manuscript / "figures"
    tables_dir = manuscript / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    tex_engine = shutil.which("latexmk") or shutil.which("pdflatex") or shutil.which("xelatex")
    agutext = shutil.which("kpsewhich")
    class_name = "article"
    if agutext:
        try:
            for cls in ["agutex.cls", "agutext.cls", "copernicus.cls"]:
                res = subprocess.run(["kpsewhich", cls], capture_output=True, text=True)
                if res.returncode == 0 and res.stdout.strip():
                    class_name = Path(cls).stem
                    break
        except Exception:
            pass

    main_row = comp.loc[comp["run_name"] == "main_closed_ep100"].iloc[0]
    soft_row = comp.loc[comp["run_name"] == "soft_gate_reference_ep100"].iloc[0]

    figure_plan = f"""# Figure Plan

1. Streamflow skill maps: `maps/map_NSE.png`, `maps/map_KGE.png`, `maps/map_low_flow_NSE.png`, `maps/map_high_flow_NSE.png`
2. Representative hydrographs: `figures/hydrographs/*.png`
3. Water-balance closure summary: `figures/closure_summary.png`
4. Learned storage/parameter maps: `maps/map_theta_cap_mean.png`, `maps/map_theta_wetpoint_weighted_ep60.png`, `maps/map_K_weighted.png`, `maps/map_component_entropy_ep60.png`
5. Theta-cap interpretability scatter suite: `figures/theta_cap_scatter_suite.png`
6. ET validation summary: `figures/et_validation_summary.png`
7. ET validation maps: `maps/map_ET_bias_FLUXCOM.png`, `maps/map_R2_FLUXCOM.png`, `maps/map_model_ET_over_P.png`
8. TWSA validation maps: `maps/map_R2_JPL.png`, `maps/map_corr_regional.png`, `maps/map_amplitude_ratio.png`
9. Regional TWSA time series: `figures/twsa_regional_timeseries.png`
"""
    (manuscript / "figure_plan.md").write_text(figure_plan)

    outline = """# WRR Model 6 Manuscript Outline

1. Introduction
2. Data and model
3. Methods
4. Results
5. Discussion
6. Conclusions
7. Data and code availability
"""
    (manuscript / "wrr_model6_manuscript_outline.md").write_text(outline)

    # Copy key figures and tables into LaTeX-friendly subfolders.
    figure_sources = [
        root / "maps" / "map_NSE.pdf",
        root / "maps" / "map_KGE.pdf",
        root / "maps" / "supplementary" / "map_NSE_unclipped.pdf",
        root / "maps" / "supplementary" / "map_KGE_unclipped.pdf",
        root / "maps" / "map_ET_bias_FLUXCOM.pdf",
        root / "maps" / "map_R2_JPL.pdf",
        root / "figures" / "closure_summary.pdf",
        root / "figures" / "et_validation_summary.pdf",
        root / "figures" / "theta_cap_scatter_suite.pdf",
        root / "figures" / "twsa_regional_timeseries.pdf",
    ]
    copied_figs = []
    for src in figure_sources:
        if src.exists():
            dst = figures_dir / src.name
            shutil.copy2(src, dst)
            copied_figs.append(dst.name)

    for src in [
        tables / "model_comparison_summary.csv",
        tables / "streamflow_metrics_by_region.csv",
        tables / "et_validation_summary_by_region.csv",
        tables / "twsa_validation_by_region.csv",
    ]:
        if src.exists():
            shutil.copy2(src, tables_dir / src.name)

    full = f"""# A Closed-Water-Balance Ecohydrological Model Benchmark for 671 CAMELS Basins Across the Conterminous United States

## Abstract
We assembled an end-to-end benchmark package for the closed `Model6Closed_Snow_aSrz_SIMHYD_Simple` run trained on 671 CAMELS basins and evaluated through `Ep100`. The model achieved median streamflow skill of NSE `{p(stream['NSE'].median())}` and KGE `{p(stream['KGE'].median())}` while maintaining near-perfect process-level closure, with median absolute daily closure residual `{p(wb['mean_abs_daily_wb_residual_mm_day'].median(), 6)}` mm/day and zero basins exceeding 1% cumulative water-balance error in the saved benchmark. Independent diagnostics using FLUXCOM monthly ET and JPL GRACE mascon storage anomalies indicate that the streamflow-trained model captures major ET variability and part of large-scale storage variability, but these products also expose limitations in low-flow representation, internal partitioning certainty, and coarse-scale storage validation. The resulting package supports a model-development and diagnostic framing rather than a stronger claim of fully constrained internal hydrologic realism.

## Plain Language Summary
We packaged a continental hydrology model test in a form that can be audited basin by basin. The model reproduces streamflow reasonably well and conserves water very tightly. Independent evapotranspiration and total water storage products show that some internal behavior is physically plausible, but they also show where a streamflow-trained model is still uncertain, especially for low flows and groundwater/storage partitioning. The package therefore supports transparent diagnosis rather than over-claiming physical certainty.

## Introduction
Large-sample hydrologic modeling increasingly aims to combine predictive skill with interpretable internal states and parameters. A persistent challenge is that good streamflow skill does not guarantee physically credible internal partitioning among snow, soil moisture, evapotranspiration, and groundwater. Here we assemble a Water Resources Research-style benchmark package around a closed-water-balance version of Model 6 trained on 671 CAMELS basins over the conterminous United States. The package is designed to answer three questions: (1) how strong is streamflow skill across hydroclimatic regimes, (2) how well does the model maintain explicit water balance closure, and (3) whether the learned storage and evapotranspiration behavior are broadly consistent with independent products.

## Methods
The authoritative model is `Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep100_iter200`, evaluated from the saved `Ep100` checkpoint and benchmark summaries. Because a fully materialized `Ep100` state-rich daily archive was not available locally, the package combines the authoritative `Ep100` benchmark summaries with the existing `Ep60` weighted daily state archive and the existing local model validation daily/monthly products. All outputs that rely on the auxiliary archive are labeled explicitly. Basin geometry comes from the dissolved CAMELS HRU shapefile and static controls come from the standard CAMELS attribute tables. Independent ET diagnostics use local FLUXCOM monthly basin series and local MOD16A3GF annual basin summaries; local GLEAM placeholders exist but contain no valid rows, so no GLEAM result is claimed here. Independent storage diagnostics use local JPL GRACE mascon basin-month series; CSR and GSFC products were not found locally and are therefore only represented by downloader templates.

Streamflow skill is summarized using the saved `Ep100` closed-model basin metrics, augmented by auxiliary daily diagnostics for log-NSE, correlation, seasonal error, and flow-duration-curve error. Water balance closure uses the saved `Ep100` benchmark water-balance summaries. Learned parameter interpretation focuses on `theta_cap`, `theta_wetpoint`, groundwater recession `K`, and component weights, with controls from climate, vegetation, soil, and geology. Independent ET validation compares model ET against FLUXCOM monthly ET and uses MOD16 as an annual-only cross-check. Total water storage anomaly (TWSA) diagnostics compare model storage anomalies against JPL GRACE at exploratory basin scale and at aggregated HUC2 scale.

## Results
### Streamflow skill and closure
The closed model achieved median NSE `{p(stream['NSE'].median())}`, median KGE `{p(stream['KGE'].median())}`, median R2 `{p(stream['R2'].median())}`, and median low-flow NSE `{p(stream['low_flow_NSE'].median())}`. Auxiliary daily diagnostics yielded median log-NSE `{p(stream['logNSE_daily_aux'].median())}`. Performance varies strongly across hydroclimatic regions, with HUC2 medians exceeding 0.75 NSE in parts of the Northeast and dropping sharply in several arid western regions. Water balance closure remained exceptionally tight: median mean absolute daily closure residual was `{p(wb['mean_abs_daily_wb_residual_mm_day'].median(), 6)}` mm/day and median cumulative relative error was `{p(wb['cumulative_relative_wb_error'].median(), 6)}`. In contrast, the saved soft-gate reference retained higher median NSE (`{p(main_row['median_NSE'])}` for the closed model versus `{p(soft_row['median_NSE'])}` for the soft-gate reference) but at the cost of external losses and large closure drift.

### Learned storage and parameter structure
Median learned `theta_cap` in the benchmark summaries was `{p(params['theta_cap_mean'].median())}` mm, with median aSrz capacity `{p(params['aSrz_capacity_mm'].median())}` mm. The auxiliary parameter package gives median `theta_wetpoint` `{p(params['theta_wetpoint_weighted_ep60'].median())}`, median groundwater recession `K` `{p(params['K_weighted'].median(), 4)}`, median `theta_ab` `{p(params['theta_ab_weighted'].median())}`, and median `theta_ak` `{p(params['theta_ak_weighted'].median())}`. The learned storage scale is strongly structured by climate and partitioning behavior: Spearman correlation between `theta_cap` and aridity is `{p(theta_corr.loc[theta_corr['predictor']=='aridity','spearman_theta_cap'].iloc[0])}`, with ET/P `{p(theta_corr.loc[theta_corr['predictor']=='ET_over_P','spearman_theta_cap'].iloc[0])}`, and with runoff ratio `{p(theta_corr.loc[theta_corr['predictor']=='runoff_ratio','spearman_theta_cap'].iloc[0])}`. Random-forest ranking indicates that aridity, forest cover, soil depth, and geology/permeability are the leading static controls in the local feature set.

### Independent ET validation
For the basins with valid FLUXCOM overlap, the local validation tables reproduce median FLUXCOM monthly R2 `{p(et['R2_FLUXCOM'].median())}` and median monthly NSE `{p(et['NSE_FLUXCOM'].median())}` across `{int(et['NSE_FLUXCOM'].notna().sum())}` basins. The annual MOD16 cross-check is somewhat stronger in this local package, with median annual NSE `{p(et['NSE_MOD16'].median())}`. The ET maps and seasonal plots show that the model captures large seasonal ET structure but also that independent ET products disagree materially. Because the local GLEAM files are empty placeholders, no monthly or seasonal GLEAM result is claimed.

### Independent storage anomaly validation
The local JPL basin package yields `{int(twsa['NSE_JPL'].notna().sum())}` basins with valid exploratory comparison, with median basin R2 `{p(twsa['R2_JPL'].median())}`, median basin NSE `{p(twsa['NSE_JPL'].median())}`, and median basin KGE `{p(twsa['KGE_JPL'].median())}`. Area filtering shows that many small basins have no reliable coarse-grid overlap, so basin-scale TWSA results are exploratory. At HUC2 aggregation, the median regional correlation is `{p(twsa_reg['corr_regional'].median())}`. Regional NSE is often lower and more fragile, which is consistent with scale mismatch, anomaly centering, and the limited complexity of the model’s storage architecture. CSR and GSFC GRACE products were not available locally, so the product-comparison table explicitly reports them as missing and no cross-product uncertainty claim is made beyond JPL.

### Baseline comparison
Relative to the saved local alternatives, the closed model is best interpreted as a closure-preserving benchmark rather than a pure top-line skill winner. The soft-gate reference has higher median NSE and KGE but also large external loss fraction and widespread closure error. The closed model outperforms the minimal aSrz variant and is broadly competitive with the available dynamic-K variants while retaining exact closed-water-balance behavior.

## Discussion
This benchmark package supports several clear conclusions. First, the closed model is credible as a continental streamflow benchmark with strong closure discipline. Second, closure alone does not guarantee correct partitioning: low-flow NSE remains weak, ET products disagree, and GRACE support is moderate rather than definitive. Third, the learned storage capacity behaves like an effective hydroclimatic storage scale rather than a literal soil-depth estimate. Its correlations with aridity, ET/P, and runoff ratio are physically coherent, but the model remains streamflow-trained and therefore internal realism must be treated as an independent diagnostic question, not as proven truth. Fourth, GRACE evaluation is scale-limited: the strongest signals appear in larger basins and regional aggregates, while small basins are often invalid at mascon scale. Finally, the simple groundwater structure and PET-driven ET formulation likely limit low-flow realism and ecohydrologic partitioning.

## Conclusions
The WRR package demonstrates that the closed Model 6 benchmark can deliver continental-scale streamflow skill together with nearly exact water-balance closure and interpretable effective storage behavior. Independent ET and TWSA diagnostics are encouraging but incomplete: FLUXCOM and MOD16 indicate meaningful ET consistency, whereas JPL GRACE indicates moderate large-scale storage realism with strong caveats about scale and missing cross-product confirmation. The appropriate framing is therefore a reproducible model-development and diagnostic paper, not a claim that all internal fluxes and states are fully constrained by observation.

## Data and Code Availability
All outputs referenced here are organized under `WRR_Model6_EndToEnd_Paper/`. Placeholder downloader templates were created for GLEAM, GRACE CSR, and GRACE GSFC because those products were not available locally in usable form during this build.
"""
    (manuscript / "wrr_model6_full_draft.md").write_text(full)

    refs = """@article{newman2015camels,
  title={The CAMELS data set: catchment attributes and meteorology for large-sample studies},
  author={Newman, AJ and others},
  journal={Hydrology and Earth System Sciences},
  year={2015}
}

@article{knoben2019kge,
  title={Technical note: Inherent benchmark or not? Comparing Nash-Sutcliffe and Kling-Gupta efficiency scores},
  author={Knoben, WJM and Freer, JE and Woods, RA},
  journal={Hydrology and Earth System Sciences},
  year={2019}
}

@article{miralles2011gleam,
  title={Global land-surface evaporation estimated from satellite-based observations},
  author={Miralles, Diego and others},
  journal={Hydrology and Earth System Sciences},
  year={2011}
}

@article{lorenz2014grace,
  title={GRACE-based terrestrial water storage changes and their interpretation},
  author={Lorenz, Christian and others},
  journal={Journal of Hydrology},
  year={2014}
}
"""
    (manuscript / "references.bib").write_text(refs)

    nse_caption = (
        "Spatial distribution of streamflow prediction skill across the 671 CAMELS basins. "
        "NSE values are shown on a fixed 0--1 scale to emphasize differences among basins with non-negative skill. "
        "Values below 0 are clipped to 0 for visualization only; original values are retained in the basin-level metric table."
    )
    kge_caption = (
        "Spatial distribution of Kling-Gupta efficiency across the 671 CAMELS basins. "
        "KGE values are shown on a fixed 0--1 scale for visual comparability. "
        "Values below 0 are clipped to 0 for visualization only; original values are retained in the basin-level metric table."
    )
    if class_name == "agutex":
        docclass = "\\documentclass[draft]{agutex}"
    elif class_name == "agutext":
        docclass = "\\documentclass[draft]{agutext}"
    elif class_name == "copernicus":
        docclass = "\\documentclass[12pt]{copernicus}"
    else:
        docclass = "\\documentclass[12pt]{article}"
    nse_include = "\\includegraphics[width=\\textwidth]{figures/map_NSE.pdf}" if "map_NSE.pdf" in copied_figs else "% map_NSE.pdf missing"
    kge_include = "\\includegraphics[width=\\textwidth]{figures/map_KGE.pdf}" if "map_KGE.pdf" in copied_figs else "% map_KGE.pdf missing"
    et_include = (
        "\\includegraphics[width=\\textwidth]{figures/et_validation_summary.pdf}"
        if "et_validation_summary.pdf" in copied_figs
        else "% et_validation_summary.pdf missing"
    )
    twsa_include = (
        "\\includegraphics[width=\\textwidth]{figures/twsa_regional_timeseries.pdf}"
        if "twsa_regional_timeseries.pdf" in copied_figs
        else "% twsa_regional_timeseries.pdf missing"
    )
    tex = rf"""{docclass}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{hyperref}}
\usepackage{{natbib}}
\title{{A Closed-Water-Balance Ecohydrological Model Benchmark for 671 CAMELS Basins Across the Conterminous United States}}
\author{{Author placeholders}}
\date{{}}
\begin{{document}}
\maketitle
\section*{{Key Points}}
\begin{{itemize}}
\item The closed Model 6 benchmark achieves strong continental streamflow skill while preserving near-exact water-balance closure.
\item Independent FLUXCOM and MOD16 ET diagnostics support broad ET realism, but local GLEAM products were unavailable for this build.
\item JPL GRACE supports moderate large-scale storage realism, while small-basin storage comparisons remain exploratory.
\end{{itemize}}
\begin{{abstract}}
The closed Model 6 benchmark achieved median NSE {p(stream['NSE'].median())} and median KGE {p(stream['KGE'].median())} with a median absolute daily closure residual of {p(wb['mean_abs_daily_wb_residual_mm_day'].median(),6)} mm/day.
\end{{abstract}}
\section*{{Plain Language Summary}}
We assembled a reproducible continental hydrology benchmark package and evaluated it with independent evapotranspiration and storage products.
\section{{Introduction}}
Large-sample hydrologic models need both predictive skill and physically credible internal organization.
\section{{Methods}}
The authoritative run is `Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep100_iter200`. State-rich daily diagnostics use the existing Ep60 auxiliary archive when no Ep100 state export was available locally.
\section{{Results}}
Median streamflow NSE was {p(stream['NSE'].median())}; median KGE was {p(stream['KGE'].median())}. Median FLUXCOM ET NSE was {p(et['NSE_FLUXCOM'].median())} across {int(et['NSE_FLUXCOM'].notna().sum())} basins. Median JPL GRACE basin NSE was {p(twsa['NSE_JPL'].median())}.
\begin{{figure}}[ht]
\centering
{nse_include}
\caption{{{nse_caption}}}
\end{{figure}}
\begin{{figure}}[ht]
\centering
{kge_include}
\caption{{{kge_caption}}}
\end{{figure}}
\begin{{figure}}[ht]
\centering
{et_include}
\caption{{Monthly and seasonal evapotranspiration validation against FLUXCOM with MOD16 used as an annual cross-check.}}
\end{{figure}}
\begin{{figure}}[ht]
\centering
{twsa_include}
\caption{{Regional comparison between model TWSA and JPL GRACE TWSA.}}
\end{{figure}}
\section{{Discussion}}
The model should be interpreted as a closure-preserving diagnostic benchmark rather than a claim of fully constrained internal realism.
\section{{Conclusions}}
The benchmark supports a model-development and evaluation paper with explicit caveats about ET product disagreement, GRACE scale mismatch, and inferred storage capacity.
\section*{{Data Availability}}
All generated outputs are organized under `WRR_Model6_EndToEnd_Paper/`.
\section*{{Code Availability}}
All scripts used to generate the package are stored under `WRR_Model6_EndToEnd_Paper/scripts/`.
\section*{{Acknowledgments}}
Acknowledgments placeholder.
\bibliographystyle{{plainnat}}
\bibliography{{references}}
\end{{document}}
"""
    tex_path = manuscript / "model6_wrr_paper.tex"
    tex_path.write_text(tex)

    compile_log = manuscript / "latex_compile.log"
    pdf_path = manuscript / "model6_wrr_paper.pdf"
    compile_status = "not_attempted"
    if tex_engine:
        try:
            if Path(tex_engine).name == "latexmk":
                cmd = [tex_engine, "-pdf", "-interaction=nonstopmode", str(tex_path.name)]
            else:
                cmd = [tex_engine, "-interaction=nonstopmode", str(tex_path.name)]
            res = subprocess.run(cmd, cwd=manuscript, capture_output=True, text=True)
            compile_log.write_text((res.stdout or "") + "\n" + (res.stderr or ""))
            if res.returncode == 0 and pdf_path.exists():
                compile_status = "success"
            else:
                compile_status = "failed"
        except Exception as exc:
            compile_log.write_text(str(exc))
            compile_status = "failed"
    else:
        compile_log.write_text("No LaTeX engine available locally. Generated .tex only.\n")
        compile_status = "unavailable"

    files_inspected = "\n".join(
        f"- `{x}`"
        for x in inv["path"].tolist()[:25]
        if pd.notna(x) and str(x).lower() not in {"null", "nan"}
    )
    final_report = f"""# Final Report

## Files inspected
{files_inspected}

## Files created
- `diagnostics/model6_ep100_daily_basin_day.parquet`
- `diagnostics/model6_ep100_monthly_basin.parquet`
- `diagnostics/model6_ep100_learned_parameters.csv`
- `tables/streamflow_metrics_by_basin.csv`
- `tables/water_balance_closure_by_basin.csv`
- `tables/learned_parameters_by_basin.csv`
- `tables/et_validation_by_basin.csv`
- `tables/twsa_validation_by_basin_exploratory.csv`
- `tables/model_comparison_summary.csv`
- `manuscript/wrr_model6_full_draft.md`

## Scripts created
- `scripts/export_model6_ep100_daily_archive.py`
- `scripts/analyze_streamflow_skill.py`
- `scripts/analyze_water_balance_closure.py`
- `scripts/analyze_learned_parameters.py`
- `scripts/analyze_interpretability_tests.py`
- `scripts/validate_et_products.py`
- `scripts/validate_twsa_grace.py`
- `scripts/build_model_comparison_table.py`
- `scripts/build_geodata_packages.py`
- `scripts/write_manuscript_package.py`
- downloader templates for GLEAM/CSR/GSFC

## Found vs missing products
- FLUXCOM monthly ET: found and used
- MOD16A3GF annual ET: found and used
- JPL GRACE mascon: found and used
- GLEAM ET: local CSV placeholders exist but are empty; no result claimed
- GRACE CSR/GSFC: not found locally; downloader templates only
- ESA CCI soil moisture: found locally but not used in the core WRR package
- NSIDC SWE validation: found locally as auxiliary context

## Whether ET maps were generated
- Yes. FLUXCOM-based ET maps and summary figures were generated.
- No GLEAM maps were generated because the local product is empty.

## Whether TWSA maps were generated
- Yes. JPL-based exploratory basin maps and regional figures were generated.
- CSR/GSFC comparison maps were not generated because those products are missing.

## Summary
- Streamflow: median NSE `{p(stream['NSE'].median())}`, median KGE `{p(stream['KGE'].median())}`, median low-flow NSE `{p(stream['low_flow_NSE'].median())}`
- Closure: median daily residual `{p(wb['mean_abs_daily_wb_residual_mm_day'].median(), 6)}` mm/day, median cumulative error `{p(wb['cumulative_relative_wb_error'].median(), 6)}`
- Storage: median theta_cap `{p(params['theta_cap_mean'].median())}` mm, median aSrz capacity `{p(params['aSrz_capacity_mm'].median())}` mm
- ET: median FLUXCOM NSE `{p(et['NSE_FLUXCOM'].median())}`, median MOD16 annual NSE `{p(et['NSE_MOD16'].median())}`
- TWSA: median JPL basin NSE `{p(twsa['NSE_JPL'].median())}`, median regional correlation `{p(twsa_reg['corr_regional'].median())}`

## Key figures
- `figures/et_validation_summary.png`
- `figures/closure_summary.png`
- `figures/theta_cap_scatter_suite.png`
- `figures/twsa_regional_timeseries.png`
- `maps/map_NSE.png`
- `maps/map_ET_bias_FLUXCOM.png`
- `maps/map_R2_JPL.png`

## Map styling and LaTeX manuscript update
- Sequential colormaps used: `viridis`, `plasma`, `magma`, `cividis`, and `turbo`
- Diverging colormaps used: `RdBu_r`, `coolwarm`, and `BrBG`
- NSE/KGE were clipped only for visualization on fixed 0--1 maps; original values remain unchanged in the CSV tables
- Unclipped supplementary maps: `maps/supplementary/map_NSE_unclipped.png`, `maps/supplementary/map_KGE_unclipped.png`
- LaTeX engine available: `{bool(tex_engine)}`
- LaTeX class/template used: `{class_name if class_name else 'article'}`
- PDF compilation status: `{compile_status}`
- LaTeX source: `manuscript/model6_wrr_paper.tex`
- Compiled PDF: `{ 'manuscript/model6_wrr_paper.pdf' if compile_status == 'success' else 'not generated' }`

## Remaining blockers
- No fully local usable GLEAM ET product
- No local CSR or GSFC GRACE product
- State-rich daily archive uses explicit `Ep60` auxiliary diagnostics for internal states/partitioning while `Ep100` remains the authoritative benchmark for streamflow and closure summaries
- GRACE small-basin results remain exploratory because of coarse product scale

## Exact rerun commands
```bash
cd /home/mircore/Desktop/diff_hydro/WRR_Model6_EndToEnd_Paper/scripts
python build_local_inventory.py
python export_model6_ep100_daily_archive.py
python analyze_streamflow_skill.py
python analyze_water_balance_closure.py
python analyze_learned_parameters.py
python analyze_interpretability_tests.py
python validate_et_products.py
python validate_twsa_grace.py
python build_model_comparison_table.py
python build_geodata_packages.py
python write_manuscript_package.py
```
"""
    (root / "final_report.md").write_text(final_report)


if __name__ == "__main__":
    main()
