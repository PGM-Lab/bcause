"""Independent brute-force check of the exact LP solver.

--------------------------------------------------------------------------------
HOW IT WORKS
--------------------------------------------------------------------------------
It reuses the *problem* from the solver -- the same feasible polytope
``a_eq @ theta = b_eq, theta >= 0`` and the same objective vectors -- via
``ExactCredalCausalInference.program(...)``. It then optimises by VERTEX SAMPLING
instead of by an LP on the objective:

    repeat many times:
        pick a random cost direction  c ~ N(0, I)
        theta = argmin over the polytope of  c @ theta      (a linprog -> a VERTEX)
        evaluate the query VALUE at theta literally:
            PNS (linear)     : value = num @ theta
            PN/PS (fractional): value = (num @ theta) / (den @ theta)   [if den > eps]
        track running min and max of value

Why this converges to the truth: a linear (PNS) or linear-fractional (PN/PS)
objective attains its optimum over a polytope at a VERTEX; a random cost vector
makes ``linprog`` return a random vertex; enough random directions eventually hit
the optimal vertex. So the brute max/min bracket the true optimum from the
inside -- a sound sanity check (it can only *under*-estimate the max), not a
proof by itself. The proof of optimality is the LP's own dual certificate; brute
force independently corroborates it and rules out looser values.

Efficiency notes:
  * Each iteration is one warm, tiny LP (n = number of confounder states, ~<=16).
  * We keep the WITNESS (the theta achieving the best value) so a human can
    verify it satisfies every constraint and reproduces the bound by hand.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exact_credal_solver import (  # noqa: E402
    CredalProgram, ExactCredalCausalInference, _load,
)

_DENOM_EPS = 1e-12


def _value(prog: CredalProgram, theta: np.ndarray):
    """The query's raw value at one feasible ``theta`` (no transform)."""
    if prog.kind == "linear":
        return float(prog.num @ theta)
    den = float(prog.den @ theta)
    return None if den < _DENOM_EPS else float(prog.num @ theta) / den


def brute_force_bounds(prog: CredalProgram, n_samples: int = 40000, seed: int = 0):
    """Bracket ``[low, upp]`` for one program by random-vertex sampling.

    Returns ``(low, upp, witness_low, witness_high)`` where each witness is a
    feasible ``theta`` (a distribution over the confounder states) that attains
    the corresponding bracket value -- kept for hand-verification.
    """
    n = prog.num.size
    bounds = [(0.0, 1.0)] * n
    rng = np.random.default_rng(seed)
    lo, hi = np.inf, -np.inf
    wl = wh = None
    # Always probe the two axis-aligned optima of the raw numerator first; the
    # random directions then explore the rest of the vertex set.
    directions = [prog.num, -prog.num] + list(rng.standard_normal((n_samples, n)))
    for c in directions:
        res = linprog(c, A_eq=prog.a_eq, b_eq=prog.b_eq, bounds=bounds, method="highs")
        if not res.success:
            continue
        val = _value(prog, res.x)
        if val is None:
            continue
        if val < lo:
            lo, wl = val, res.x
        if val > hi:
            hi, wh = val, res.x
    return (float(np.clip(lo, 0, 1)), float(np.clip(hi, 0, 1)), wl, wh)


def check_witness(prog: CredalProgram, theta: np.ndarray) -> dict:
    """Residuals for a human check: constraint error and the reproduced value."""
    return {
        "sum": float(theta.sum()),
        "max_constraint_error": float(np.max(np.abs(prog.a_eq @ theta - prog.b_eq))),
        "reproduced_value": _value(prog, theta),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Brute-force verification of the exact LP bounds.")
    ap.add_argument("base", nargs="?",
                    default="/Users/antoniogonzalezalves/Documents/s23/"
                            "simple_nparents2_nzr08_zdr10_37",
                    help="Model path WITHOUT extension.")
    ap.add_argument("queries", nargs="*", default=["PS(V1,V0)", "PN(V1,V0)", "PNS(V1,V0)"])
    ap.add_argument("--samples", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, data = _load(args.base)
    inf = ExactCredalCausalInference(model, data)

    # Optional: credici's golden bounds, if the model ships a *_query.csv.
    gold = {}
    qcsv = Path(args.base + "_query.csv")
    if qcsv.exists():
        g = pd.read_csv(qcsv)
        gold = {(r["query"], r["cause"], r["effect"]): (r["low"], r["upp"])
                for _, r in g.iterrows()}

    print(f"model: {Path(args.base).name}   samples/query={args.samples}\n")
    header = f"{'query':13} {'LP (exact)':22} {'brute force':22} {'credici golden':22} agree?"
    print(header + "\n" + "-" * len(header))
    for q in args.queries:
        kind, cause, effect = re.match(r"([A-Za-z]+)\((\w+),(\w+)\)", q).groups()
        kind = kind.upper()
        prog = inf.program(kind, cause, effect)
        lp_lo, lp_up = inf.bounds(kind, cause, effect)
        bf_lo, bf_up, w_lo, w_hi = brute_force_bounds(prog, args.samples, args.seed)
        gl = gold.get((kind, cause, effect))
        agree = abs(lp_lo - bf_lo) < 1e-4 and abs(lp_up - bf_up) < 1e-4
        gtxt = f"[{gl[0]:.5f}, {gl[1]:.5f}]" if gl else "(n/a)"
        print(f"{kind}({cause},{effect}) ".ljust(13)
              + f"[{lp_lo:.5f}, {lp_up:.5f}] ".ljust(22)
              + f"[{bf_lo:.5f}, {bf_up:.5f}] ".ljust(22)
              + f"{gtxt} ".ljust(22)
              + ("YES" if agree else "NO"))
        # Show the upper-bound witness so the result is checkable by hand.
        if w_hi is not None:
            chk = check_witness(prog, w_hi)
            print(f"    upper witness theta*=P(U): {np.round(w_hi, 5).tolist()}")
            print(f"      sum={chk['sum']:.6f}  max|A.theta-b|={chk['max_constraint_error']:.2e}"
                  f"  value={chk['reproduced_value']:.5f}")


if __name__ == "__main__":
    main()
