"""Analytic validation of the escape-solid-angle quadrature (justifies n = 200).

The capture criterion estimates the evader's escape solid angle by counting how many Fibonacci
directions fall inside the escape region -- measuring the area of a spherical region from
lattice-point membership. Gonzalez (Math. Geosci. 42:49-64, 2010) established the Fibonacci
lattice as the right instrument for that measurement; what remains is to fix n.

A spherical cap of half-angle alpha has the closed form Omega = 2*pi*(1 - cos alpha), so the
quadrature can be checked against ground truth with no simulation and no trained model. This
script reports the error as a fraction of the full sphere (4*pi), which is the scale the
criterion is compared against.

Run:  python examples/sampling_analytic_check.py
"""
import os
import sys
import math
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "xuance", "environment", "multi_agent_env"))
from geometry3d import fibonacci_sphere, solid_angle_per_point   # noqa: E402

FULL = 4.0 * math.pi


def cap_error(n, alpha_rad):
    """|estimated - exact| for a spherical cap, as a fraction of the full sphere."""
    exact = 2.0 * math.pi * (1.0 - math.cos(alpha_rad))
    dirs = fibonacci_sphere(n)
    est = float(np.sum(dirs[:, 2] > math.cos(alpha_rad)) * solid_angle_per_point(n))
    return est, exact, abs(est - exact) / FULL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="50,100,200,400,800,1600")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",") if x.strip()]

    # Sweep cap sizes densely so the reported error is a worst case over region sizes, not a
    # lucky single cap: a Fibonacci lattice can hit one particular cap almost exactly.
    alphas = np.arange(5, 176, 5)
    rows = []
    for n in levels:
        errs = [cap_error(n, math.radians(a))[2] for a in alphas]
        rows.append({
            "n": n,
            "angular_resolution_deg": math.degrees(math.sqrt(FULL / n)),
            "mean_err_frac_sphere": float(np.mean(errs)),
            "max_err_frac_sphere": float(np.max(errs)),
            "solid_angle_per_point_sr": solid_angle_per_point(n),
        })

    print(f"Spherical-cap quadrature error over {len(alphas)} cap sizes (5deg..175deg)")
    print(f"{'n':>6} {'ang.res':>9} {'mean err':>10} {'max err':>10}   (fraction of 4pi)")
    for r in rows:
        print(f"{r['n']:>6} {r['angular_resolution_deg']:>8.1f}deg "
              f"{r['mean_err_frac_sphere'] * 100:>9.3f}% {r['max_err_frac_sphere'] * 100:>9.3f}%")

    ref = {r["n"]: r for r in rows}
    if 200 in ref and 400 in ref:
        gain = ref[200]["max_err_frac_sphere"] / max(ref[400]["max_err_frac_sphere"], 1e-12)
        print(f"\nn=200 worst-case error: {ref[200]['max_err_frac_sphere'] * 100:.3f}% of the "
              f"full sphere; doubling to n=400 improves it by {gain:.1f}x at 2x the per-step cost.")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"cap_sweep_deg": alphas.tolist(), "levels": rows}, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
