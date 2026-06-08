# Omitted Large Files

The following files were intentionally excluded from the GitHub-friendly export bundle because they are large and not required for direct Overleaf compilation.

## Omitted

- `/home/mircore/Desktop/diff_hydro/WRR_Model6_EndToEnd_Paper/diagnostics/model6_ep100_daily_basin_day.parquet`
  - Reason: large daily state/flux archive
  - Status: retained locally
- `/home/mircore/Desktop/diff_hydro/WRR_paper_reproducibility_bundle.zip`
  - Reason: export zip exceeds GitHub's 100 MB single-file limit
  - Status: retained locally; reproducibility folder pushed instead

## Included substitutes

- `reproducibility_bundle/diagnostics/model6_ep100_monthly_basin.parquet`
- `reproducibility_bundle/diagnostics/model6_ep100_learned_parameters.csv`
- `reproducibility_bundle/diagnostics/model6_ep100_archive_manifest.json`

## Rationale

The export is designed so that:

- the Overleaf package stays light and uploadable
- the GitHub package remains practical to clone
- the full original local workspace still preserves complete reproducibility
