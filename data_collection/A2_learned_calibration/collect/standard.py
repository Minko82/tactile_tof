"""
standard.py — one standard four-phase comparison run (the original test).

    python3 standard.py <surface> [seed] [noviz]

Examples:
    python3 standard.py wood            wood run, live plot
    python3 standard.py felt 7 noviz    new material, fixed seed, headless

Data -> A2_learned_calibration/<surface>/<n>/.  After 2-3 runs on a surface:
    python3 ../calibration.py fit <surface>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import compare_test


def parse(a):
    nums = [int(x) for x in a if x.isdigit()]
    words = [x for x in a if not x.isdigit() and x not in ("viz", "noviz")]
    if not words:
        raise SystemExit(__doc__)
    return words[0], (nums[0] if nums else None), "noviz" not in a


if __name__ == "__main__":
    surf, seed, viz = parse(sys.argv[1:])
    compare_test.run(surface=surf, seed=seed, viz=viz)
