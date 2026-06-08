from __future__ import annotations

from pathlib import Path


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "logs" / "download_grace_gsfc_template.txt"
    out.write_text(
        "GRACE GSFC downloader template only.\n"
        "No local GSFC product was found during inventory.\n"
        "After obtaining a monthly mascon/anomaly product, aggregate to basin-month and update config.grace_gsfc_path.\n"
    )


if __name__ == "__main__":
    main()
