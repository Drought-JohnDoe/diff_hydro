from __future__ import annotations

from pathlib import Path


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "logs" / "download_gleam_template.txt"
    out.write_text(
        "GLEAM ET downloader template only. Local GLEAM basin CSVs are empty.\n"
        "Expected action: obtain licensed/public GLEAM files externally, then aggregate to 671 basins.\n"
        "Suggested target products: monthly ETa and daily ETa covering 1995-10-01 to 2010-09-30.\n"
    )


if __name__ == "__main__":
    main()

