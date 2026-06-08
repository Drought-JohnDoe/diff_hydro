# WRR Paper Export

This export is split into two parts so the paper is easy to use in Overleaf and still reproducible from GitHub.

## Bundles

- `overleaf_bundle/`
  - Lean manuscript package for direct upload to Overleaf.
  - Contains `main.tex`, `references.bib`, paper figures as PDF, summary tables, and manuscript notes.
- `reproducibility_bundle/`
  - Larger local/GitHub package containing the scripts, configs, tables, maps, inventory, and selected diagnostics used to build the paper.
  - Keeps the paper reproducible without requiring the entire original workspace.

## Zip archives

- `../WRR_paper_overleaf_bundle.zip`
- `../WRR_paper_reproducibility_bundle.zip`

GitHub note:

- The Overleaf zip is small enough to push directly.
- The reproducibility zip is retained locally but is too large for normal GitHub upload.
- The full reproducibility *folder* is included in the GitHub export instead.

## Important note on omitted large files

The full local paper workspace is much larger than the GitHub-friendly export. The main omitted file is:

- `WRR_Model6_EndToEnd_Paper/diagnostics/model6_ep100_daily_basin_day.parquet`

This daily archive is retained locally for full reproducibility but is intentionally excluded from the GitHub export because of its size.

See `manifests/omitted_large_files.md` for details.
