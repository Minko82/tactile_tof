"""
compare_test.py — ONE UR5 run, BOTH pipelines side by side on the same frames:

    kalman  : the A3 method — constant mount offset + LiveFilter(FT_ACCEL_PSD,
              adaptive R)  (exactly what robot.py filtertest runs)
    learned : the A2 method — polynomial calibration + near-pass-through
              LiveFilter  (exactly what run_test.py runs)

Both see the identical sensor stream and robot motion, so the comparison has no
run-to-run variance. The live plot shows one filtered/velocity/error trace per
pipeline; the summary prints one per-phase table each. Results (wide-format
filter_log.csv) -> results_compare/<n>/.

    python3 compare_test.py viz        run with the live plot
    python3 compare_test.py [seed]     run headless / fixed seed
    python3 compare_test.py print      print the URScript (no motion)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "A3_proximity"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "A2_data_filter"))

import robot
from calibration import load_for_surface, GenericCalibration
from live_filter import LiveFilter
from run_test import PROCESS_ACCEL_PSD, MANEUVER_ACCEL_PSD, REJECT_SIGMA


def main():
    a = sys.argv[1:]
    nums = [int(x) for x in a if x.isdigit()]
    words = [x for x in a if not x.isdigit() and x not in ("viz", "print")]
    surf = words[0] if words else robot.DEFAULT_SURFACE   # any material name works

    cal = load_for_surface(surf)          # calibration_<surf>.json (or warns)
    print(f"surface '{surf}'  ->  runs pool {surf}/<n>/")
    print(f"kalman : offset {robot.SURFACE_OFFSETS_MM.get(surf, robot.MOUNT_OFFSET_MM)} mm "
          f"+ LiveFilter(q={robot.FT_ACCEL_PSD:.0f}, adaptive R)")
    print(f"learned: poly calibration (fit RMSE {cal.meta.get('rmse_mm', float('nan')):.2f} mm) "
          f"+ LiveFilter(q={PROCESS_ACCEL_PSD:.0f}, R=4, {REJECT_SIGMA:.0f}σ gate)")

    # say which model variant is running IN THE PLOT LEGEND, not just the
    # terminal — a fallback shows up as a bias and is easy to misread
    if os.path.exists(os.path.join(HERE, f"calibration_{surf}.json")):
        learned_name = "learned"
    elif isinstance(cal, GenericCalibration):
        learned_name = "learned (generic model)"
    else:
        learned_name = f"learned UNFITTED (using {cal.meta.get('surface', '?')})"
    pipelines = [
        ("kalman", None, None),        # robot.py defaults = the A3 method
        (learned_name, cal.apply,
         lambda: LiveFilter(process_accel_psd=PROCESS_ACCEL_PSD,
                            maneuver_accel_psd=MANEUVER_ACCEL_PSD,
                            reject_sigma=REJECT_SIGMA)),
    ]
    robot.cmd_filtertest(
        seed=nums[0] if nums else None,
        show_script="print" in a,
        viz="viz" in a,
        pipelines=pipelines,
        # every material pools its runs under A2/<surface>/<n>/
        out_base=os.path.join(HERE, surf),
        surface=surf,
    )


if __name__ == "__main__":
    main()
