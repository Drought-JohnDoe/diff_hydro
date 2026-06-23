#!/usr/bin/env python3
"""Stub for future theta_cap structural ablations.

This package locks the existing Model 6 LAIEco checkpoint. Full theta_cap
ablations require retraining and are intentionally not launched by the
publication packaging command.
"""
from pathlib import Path

ROOT = Path("/home/mircore/Desktop/diff_hydro/ECO_HYBRID/PUBLICATION_MODEL6_ROHINI_REPLICATION")

if __name__ == "__main__":
    print("Theta-cap ablation training is not run in the locked publication package.")
    print("See ABLATION_MANIFEST.yaml for configured experiments.")
