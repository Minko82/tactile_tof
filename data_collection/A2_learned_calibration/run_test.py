"""
run_test.py — UR5 filtertest for the two-stage LEARNED architecture:

    stage 1: learned memoryless calibration  (calibration.json, zero lag)
    stage 2: near-pass-through Kalman        (LiveFilter, high process_accel_psd,
             kept for velocity estimation / outlier gating / dropout coasting)

Runs the SAME four-phase motion (10s linear, 10s static, ~10s slow random,
~10s fast random) via A3_proximity/robot.py, so results are directly comparable
with the Kalman-only runs in A3/filtertest/. Results -> A2_learned_calibration/results/<n>/.

    python3 run_test.py            run
    python3 run_test.py viz        run with the live plot
    python3 run_test.py 7          fixed random seed
    python3 run_test.py print      print the URScript (no motion)

Fit the calibration first:  python3 calibration.py fit
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "A3_proximity"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "A2_data_filter"))

import robot
from calibration import load_for_surface
from live_filter import LiveFilter

# Near-pass-through stage-2 tuning, picked by offline replay sweep on held-out
# run 11 (see README): fixed R (adaptive R shrinks toward the floor on this
# ultra-clean sensor and the outlier gate then strangles real fast motion),
# high q, and a loosened 8-sigma gate that still catches genuine spikes.
PROCESS_ACCEL_PSD  = 50000.0
MANEUVER_ACCEL_PSD = 100000.0    # must stay >= PROCESS_ACCEL_PSD
REJECT_SIGMA       = 8.0


def main():
    a = sys.argv[1:]
    nums = [int(x) for x in a if x.isdigit()]
    words = [x for x in a if not x.isdigit() and x not in ("viz", "print")]
    surf = words[0] if words else robot.DEFAULT_SURFACE   # any material name works

    cal = load_for_surface(surf)          # calibration_<surf>.json (or warns)
    print(f"surface '{surf}': calibration degree {cal.meta.get('degree')}, "
          f"fit RMSE {cal.meta.get('rmse_mm', float('nan')):.2f} mm, "
          f"span [{cal.raw_min:.0f}, {cal.raw_max:.0f}] mm  "
          f"| filter: process_accel_psd={PROCESS_ACCEL_PSD:.0f}")

    robot.cmd_filtertest(
        seed=nums[0] if nums else None,
        show_script="print" in a,
        viz="viz" in a,
        surface=surf,
        transform=cal.apply,
        make_filter=lambda: LiveFilter(process_accel_psd=PROCESS_ACCEL_PSD,
                                       maneuver_accel_psd=MANEUVER_ACCEL_PSD,
                                       reject_sigma=REJECT_SIGMA),
        # every material pools its runs under A2/<surface>/<n>/
        out_base=os.path.join(HERE, surf),
    )


if __name__ == "__main__":
    main()
