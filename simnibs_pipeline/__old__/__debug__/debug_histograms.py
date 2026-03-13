"""
Debug script — appel minimal de efields_histograms pour la région extra.

Usage:
    cd simnibs-pipeline/simnibs_pipeline
    python test/debug_histograms.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from _4_viz import Visualizer

# ── Chemins ─────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("/Users/hippolyte.dreyfus/Desktop/hemianotACS/Data/derivatives/simnibs-config-healthyV2/results-V2")
DEBUG_PNG = RESULTS_DIR / "preprocess" / "debug_test.png"

BASE = Path("/Users/hippolyte.dreyfus/Desktop/hemianotACS/Data/derivatives/simnibs-config-healthyV2")
SUBJECT = "0001"

SIM_DIR = BASE / SUBJECT / "simulations/simulation_simulation_fef_hemianotacs_bc8ae6ee/mni_volumes"
STEM = f"{SUBJECT}_TDCS_1_scalar_MNI_magnE"

extra_masked  = SIM_DIR / f"{STEM}_extra_roi_masked.nii.gz"
extra_cleaned = SIM_DIR / f"{STEM}_extra_roi_cleaned.nii.gz"

# ── Appel minimal ────────────────────────────────────────────────────────────
viz = Visualizer(output_dir=RESULTS_DIR)

extra_data = {
    SUBJECT: [
        ("fef", "simulation", extra_masked, extra_cleaned),
    ]
}

viz.efields_histograms(extra_data, region="extra")
print(f"Done — check {RESULTS_DIR / 'preprocess'}")
