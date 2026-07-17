"""
angle.py — incidence-angle runs: the arm holds the sensor tilted off
perpendicular and runs the standard four-phase round at each angle.

    python3 angle.py <surface> [deg ...] [noviz]      default angles: 5 10 15

Examples:
    python3 angle.py wood               wood at 5, 10 and 15 degrees (3 runs)
    python3 angle.py white 10           white at 10 degrees only

Data -> A2_learned_calibration/<surface>_angle<deg>/<n>/ per angle.
Angles are capped at 20 deg (mount clearance above the table).
The tilt is baked into the robot program — no manual repositioning needed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import compare_test


def parse(a):
    degs = [int(x) for x in a if x.isdigit()] or [5, 10, 15]
    words = [x for x in a if not x.isdigit() and x not in ("viz", "noviz")]
    if not words:
        raise SystemExit(__doc__)
    return words[0], degs, "noviz" not in a


if __name__ == "__main__":
    surf, degs, viz = parse(sys.argv[1:])
    for i, deg in enumerate(degs):
        print(f"\n========== {surf} @ {deg} deg tilt  ({i + 1}/{len(degs)}) ==========")
        compare_test.run(surface=surf, viz=viz, tilt_deg=float(deg),
                         out_pool=f"{surf}_angle{deg}")
    print(f"\nangle sweep done: {', '.join(f'{surf}_angle{d}/' for d in degs)}")
