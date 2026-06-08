# Overleaf Upload Guide

This folder is the direct manuscript bundle for Overleaf.

## Upload steps

1. Create a new Overleaf project.
2. Upload all files in this folder.
3. Set `main.tex` as the main file if Overleaf does not pick it automatically.
4. Compile with `pdfLaTeX` or `latexmk`.

## Included content

- `main.tex`
- `references.bib`
- paper figures in `figures/`
- paper summary tables in `tables/`
- markdown draft and notes for reference

## Known local limitation

This bundle was not compiled to PDF locally because no LaTeX engine was installed in the local environment. The source is prepared for Overleaf compilation.

## Figure note

For the main manuscript maps:

- `NSE` and `KGE` are visually clipped to `0-1` for presentation.
- Original unclipped values remain in the exported CSV tables.
- Supplementary unclipped versions are also included.
