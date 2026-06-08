from __future__ import annotations

from pathlib import Path

import pandas as pd

from paper_common import load_config


def load_summary(path: str, run_name: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame([{"run_name": run_name, "status": "missing"}])
    df = pd.read_csv(p)
    df["run_name"] = run_name
    df["status"] = "available"
    return df


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "wrr_model6_config.yaml")
    runs = [
        ("main_closed_ep100", cfg["model"]["ep100_summary_csv"], "Model6Closed_Snow_aSrz_SIMHYD_Simple"),
        ("soft_gate_reference_ep100", cfg["model"]["ep100_summary_csv"], "Model6PhysicalFix_B_soft_gate"),
        ("asrz_minimal_ep10", "/home/mircore/Desktop/diff_hydro/ECO_HYBRID/Model6Physical_aSrz_Minimal_full671_ep10/summary_compare.csv", "Model6Physical_aSrz_Minimal"),
        ("dynamicK_ep50", "/home/mircore/Desktop/diff_hydro/ECO_HYBRID/Model6C_dynamicK_full671_ep50/summary_compare.csv", "Model6C_dynamicK"),
        ("dynamicK_powergw_ep50", "/home/mircore/Desktop/diff_hydro/ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_DynamicKPowerGW_full671_ep50/summary_compare.csv", "DynamicKPowerGW_full671"),
    ]
    frames = []
    for name, path, model_name in runs:
        df = load_summary(path, name)
        if "model" in df.columns:
            df = df.loc[df["model"] == model_name].copy()
        frames.append(df)
    out = pd.concat(frames, ignore_index=True, sort=False)
    out.to_csv(Path(cfg["outputs"]["tables_dir"]) / "model_comparison_summary.csv", index=False)

    missing = pd.DataFrame(
        {
            "baseline_name": ["HBV_reference", "LSTM_reference"],
            "status": ["missing_local_run", "missing_local_run"],
            "action": ["use template runner", "use template runner"],
        }
    )
    missing.to_csv(Path(cfg["outputs"]["tables_dir"]) / "missing_baseline_templates.csv", index=False)


if __name__ == "__main__":
    main()
