from __future__ import annotations

from pathlib import Path

import argparse
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Placeholder wrapper for gridded-to-basin aggregation.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "status": ["not_implemented_here"],
            "message": ["Use the existing local basin-aggregated products or project-specific aggregators."],
            "input": [args.input],
        }
    ).to_csv(out, index=False)


if __name__ == "__main__":
    main()

