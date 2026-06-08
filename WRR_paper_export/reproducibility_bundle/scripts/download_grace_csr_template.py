from __future__ import annotations

from pathlib import Path


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "logs" / "download_grace_csr_template.txt"
    out.write_text(
        "GRACE CSR downloader template only.\n"
        "Known public source used elsewhere in workspace:\n"
        "https://download.csr.utexas.edu/outgoing/grace/RL06_mascons/CSR_GRACE_GRACE-FO_RL0602_Mascons_all-corrections_v02.nc\n"
        "After download, aggregate to basin-month and update config.grace_csr_path.\n"
    )


if __name__ == "__main__":
    main()

