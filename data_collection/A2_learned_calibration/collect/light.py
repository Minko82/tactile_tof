"""
light.py — lighting-condition run: identical motion, you control the lighting.

    python3 light.py <surface> <condition> [noviz]

Examples:
    python3 light.py white lamp         desk lamp aimed at the surface
    python3 light.py white sunlight     table in direct sunlight
    python3 light.py white dark         lights off / blinds closed

Data -> A2_learned_calibration/<surface>_light_<condition>/<n>/.
The per-zone ambient level (a0..a63 columns) is what makes these runs
informative — indoors-baseline runs all sit at ~5 kcps/SPAD.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import compare_test


if __name__ == "__main__":
    a = sys.argv[1:]
    words = [x for x in a if x not in ("viz", "noviz")]
    if len(words) < 2:
        raise SystemExit(__doc__)
    surf, cond = words[0], words[1]
    input(f"Set up the lighting for condition '{cond}' now, then press Enter ... ")
    compare_test.run(surface=surf, viz="noviz" not in a,
                     out_pool=f"{surf}_light_{cond}")
