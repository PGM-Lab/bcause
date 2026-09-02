"""Exact credal counterfactual solver -- PN / PS / PNS interval bounds via LP.

A native, in-process replacement for the exact solver in ``credici.jar``
(``runCVE.java`` -> ``CredalCausalVE``). credici answers the same queries with
vertex/extreme-point credal variable elimination, which is exponential and
yields a *valid but sometimes loose outer* approximation. This module computes
the **exact tight** bounds instead, as a small linear program.

--------------------------------------------------------------------------------
THE IDEA (why this is an LP)
--------------------------------------------------------------------------------
The structural equations are deterministic, so once the effect's exogenous
confounder ``U`` is fixed to a state ``j`` the effect is a fixed response
``g(pa, j)`` of its parents. The *only* quantity the data leaves undetermined is
that confounder's distribution

        theta_j = P(U = j),        theta_j >= 0,   sum_j theta_j = 1.

Everything else (the response table ``g`` and the parent marginals) is known.
Therefore, for a fixed query:

  * the data-matching constraints  sum_j theta_j [g(a,j)=t] = P(Y=t | Pa=a)
    are LINEAR in theta  -> they carve out the "credal set" (a polytope);
  * PNS is a LINEAR functional of theta                       -> two LPs (min/max);
  * PN and PS are LINEAR-FRACTIONAL (ratio of two linear
    functionals) -> made linear by the Charnes-Cooper transform -> two LPs each.

So each bound is one call to ``scipy.optimize.linprog`` (HiGHS), i.e. an exact,
deterministic optimum -- no sampling.

--------------------------------------------------------------------------------
THE THREE INGREDIENTS
--------------------------------------------------------------------------------
1. Response table ``g``    -- the effect's OWN structural equation, read from the
                              model (fixed-mechanism setting, matching credici).
2. Credal-set constraints  -- match the empirical conditional P(Y | Pa) of the
                              effect given its parents (rounded to 5 decimals,
                              as credici's ``FactorUtil.DEFAULT_DECIMALS``).
3. Query objective         -- indicator sums over ``g`` weighted by the empirical
                              parent law, factorised over c-components (each
                              independent exogenous group contributes a product
                              term), which is how credici treats the graph.

Query semantics (T = true state, F = false state; defaults to (1, 0)):
    PNS(X,Y) = P(Y_{X=T}=T, Y_{X=F}=F)                 (unconditional)
    PN (X,Y) = P(Y_{X=F}=F | X=T, Y=T)
    PS (X,Y) = P(Y_{X=T}=T | X=F, Y=F)

Only the ``bcause`` package is used as a library; nothing here is imported by it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linprog

# Make the bcause package importable when this file is run from dev/exact_credici/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bcause.util.domainutils as dutils  # noqa: E402
from bcause.models.cmodel import StructuralCausalModel  # noqa: E402

# credici rounds every empirical factor to this many decimals
# (FactorUtil.DEFAULT_DECIMALS) before solving; matched here for parity.
_DECIMALS = 5
# Denominator below this makes a conditional query (PN/PS) undefined, e.g. PS
# conditions on P(X=false, Y=false), which some datasets never realise.
_DENOM_EPS = 1e-12


class NoFeasibleSolution(Exception):
    """No P(U) reproduces the empirical law (credici's NoFeasibleSolutionException)."""


@dataclass
class CredalProgram:
    """The optimisation problem for one query, exposed for reuse and inspection.

    Feasible set (the credal set of the confounder distribution):
        ``a_eq @ theta == b_eq``,  ``theta >= 0``     (last row of a_eq is the simplex).

    Objective, depending on ``kind``:
        - ``"linear"`` (PNS): maximise / minimise ``num @ theta``  (``den`` is None);
        - ``"fractional"`` (PN, PS): maximise / minimise ``(num @ theta)/(den @ theta)``.

    ``brute_force.py`` consumes exactly this object, so the LP solver and the
    independent verifier optimise provably the same problem.
    """

    kind: str                    # "linear" | "fractional"
    a_eq: np.ndarray             # (n_constraints, n_states)
    b_eq: np.ndarray             # (n_constraints,)
    num: np.ndarray              # objective (PNS) or numerator (PN/PS), length n_states
    den: Optional[np.ndarray]    # None for PNS; denominator vector for PN/PS


class ExactCredalCausalInference:
    """Exact PN/PS/PNS interval bounds for an SCM constrained by observed data.

    Args:
        model: A :class:`StructuralCausalModel`; only its graph and deterministic
            structural equations are used -- not its stored parameters.
        data: Endogenous dataset whose columns are the endogenous variable names
            (``V0, V1, ...``); supplies the empirical law.
    """

    def __init__(self, model: StructuralCausalModel, data: pd.DataFrame):
        self._model = model.fix_numeric_domains()
        self._domains = self._model.domains
        self._data = data

    # ------------------------------------------------------------------ public API

    def bounds(self, query: str, cause: Hashable, effect: Hashable,
               true_false_cause: Optional[tuple] = None,
               true_false_effect: Optional[tuple] = None) -> list:
        """Return ``[low, upp]`` for ``query`` in {"PNS", "PN", "PS"}."""
        prog = self.program(query, cause, effect, true_false_cause, true_false_effect)
        return self._solve(prog)

    # Thin convenience wrappers (same names/signature as bcause's causal inference).
    def prob_necessity_sufficiency(self, cause, effect, tf_cause=None, tf_effect=None):
        return self.bounds("PNS", cause, effect, tf_cause, tf_effect)

    def prob_necessity(self, cause, effect, tf_cause=None, tf_effect=None):
        return self.bounds("PN", cause, effect, tf_cause, tf_effect)

    def prob_sufficiency(self, cause, effect, tf_cause=None, tf_effect=None):
        return self.bounds("PS", cause, effect, tf_cause, tf_effect)

    def program(self, query: str, cause: Hashable, effect: Hashable,
                true_false_cause: Optional[tuple] = None,
                true_false_effect: Optional[tuple] = None) -> CredalProgram:
        """Assemble the LP for one query (constraints + objective), without solving.

        Kept separate from :meth:`bounds` so the brute-force verifier can optimise
        the identical problem by a different method.
        """
        m = self._model
        endo_parents = m.get_edogenous_parents(effect)
        if cause not in endo_parents:
            raise ValueError(f"{cause} must be a direct endogenous parent of {effect}")

        Tc, Fc = true_false_cause or _true_false(cause, self._domains[cause])
        Te, Fe = true_false_effect or _true_false(effect, self._domains[effect])

        # (1) response table g[parent_config, exo_state] and (2) credal constraints.
        response, n_states = self._response_table(effect, endo_parents)
        parent_space = dutils.assingment_space(
            dutils.subdomain(self._domains, *endo_parents))
        row_of = {tuple(a[v] for v in endo_parents): i
                  for i, a in enumerate(parent_space)}
        a_eq, b_eq = self._constraints(response, parent_space, endo_parents,
                                       effect, Te, n_states)

        # (3) objective vectors, summed over the model's response functions and
        # weighted by the empirical parent law (factorised over c-components).
        marginals = self._marginals(endo_parents)

        def g_when(a, cause_value):
            """Response vector (over exo states) for config ``a`` with the cause set."""
            key = tuple(cause_value if v == cause else a[v] for v in endo_parents)
            return response[row_of[key]]

        if query == "PNS":
            others = [p for p in endo_parents if p != cause]
            c = np.zeros(n_states)
            for a in parent_space:
                if a[cause] != Tc:          # enumerate each {others} once, via X=T rows
                    continue
                w = _config_weight(marginals, a, others)
                if w:
                    c += w * (g_when(a, Tc) == Te) * (g_when(a, Fc) == Fe)
            return CredalProgram("linear", a_eq, b_eq, c, None)

        # PN / PS: numerator and denominator of the conditional counterfactual.
        # PN conditions on (X=T, Y=T) and asks Y_{X=F}=F; PS mirrors it.
        obs_c, obs_e = (Tc, Te) if query == "PN" else (Fc, Fe)
        cf_c, cf_e = (Fc, Fe) if query == "PN" else (Tc, Te)
        num = np.zeros(n_states)
        den = np.zeros(n_states)
        for a in parent_space:
            if a[cause] != obs_c:
                continue
            w = _config_weight(marginals, a, endo_parents)
            if not w:
                continue
            base = w * (response[row_of[tuple(a[v] for v in endo_parents)]] == obs_e)
            den += base
            num += base * (g_when(a, cf_c) == cf_e)
        return CredalProgram("fractional", a_eq, b_eq, num, den)

    # --------------------------------------------------------------- model pieces

    def _response_table(self, effect, endo_parents):
        """Effect response over the model's own exogenous states (fixed mechanism).

        ``response[i, j]`` = value the effect takes for the ``i``-th
        endogenous-parent configuration (assingment_space order) under the
        ``j``-th joint exogenous state. Using the model's own response functions
        -- not the full canonical space -- is what makes the result the exact
        tight refinement of credici.
        """
        exo_parents = self._model.get_exogenous_parents(effect)
        factor = self._model.factors[effect]
        y_states = np.asarray(self._domains[effect])
        n_states = int(np.prod([len(self._domains[u]) for u in exo_parents])) or 1
        table = [
            y_states[factor.restrict(**a)
                     .values_array(var_order=[effect] + exo_parents)
                     .argmax(axis=0)].reshape(n_states)
            for a in dutils.assingment_space(
                dutils.subdomain(self._domains, *endo_parents))
        ]
        return np.asarray(table), n_states

    def _constraints(self, response, parent_space, endo_parents, effect, Te, n_states):
        """Credal-set equalities ``sum_j theta_j [g(a,j)=s] = P(Y=s|Pa=a)`` + simplex.

        For a binary effect one equality per parent config suffices (the other
        state follows from the simplex). For a k-state effect (the ``ysize3``
        models, |Y|=3) we pin k-1 states per config; the last is implied by the
        simplex, so the full conditional P(Y|Pa) is matched.
        """
        cond = self._conditional(effect, endo_parents)   # rounded empirical conditionals
        effect_states = list(self._domains[effect])
        rows, rhs = [], []
        for i, a in enumerate(parent_space):
            key = tuple(a.values())
            if key not in cond:            # config never observed -> no information
                continue
            dist = cond[key]
            for s in effect_states[:-1]:   # pin all but one state; last is implied
                rows.append((response[i] == s).astype(float))
                rhs.append(dist.get(s, 0.0))
        rows.append(np.ones(n_states))     # probabilities sum to one
        rhs.append(1.0)
        return np.asarray(rows), np.asarray(rhs)

    def _conditional(self, effect, endo_parents):
        """Empirical, 5-dp-rounded ``P(Y | Pa=a)`` keyed by parent-value tuple."""
        joint = self._data.groupby(list(endo_parents) + [effect]).size()
        table: dict = {}
        for idx, n in joint.items():
            table.setdefault(idx[:-1], {})[idx[-1]] = n
        for key, dist in table.items():
            states = list(dist)
            probs = _round_normalize(np.array([dist[s] for s in states], float))
            table[key] = dict(zip(states, probs))
        return table

    def _marginals(self, endo_parents):
        """Empirical parent law factorised over c-components (as credici does).

        Parents are grouped by the c-component they belong to; each group's
        empirical joint is estimated (and 5-dp rounded) independently. For the
        Markovian ``s23`` models every group is a single root, so this is a
        product of per-variable marginals -- which differs from the empirical
        JOINT by finite-sample noise, and that difference is what makes the
        numbers match credici exactly.

        Returns a list of ``(group_vars, {value_tuple: prob})``.
        """
        parents = set(endo_parents)
        marginals = []
        for comp in self._model.endo_ccomponents:
            group = [p for p in endo_parents if p in comp and p in parents]
            if not group:
                continue
            counts = self._data.groupby(group).size()
            probs = _round_normalize(counts.to_numpy(dtype=float))
            keys = [k if len(group) > 1 else (k,) for k in counts.index]
            marginals.append((group, dict(zip(keys, probs))))
        return marginals

    # ------------------------------------------------------------------ LP solving

    def _solve(self, prog: CredalProgram) -> list:
        if prog.kind == "linear":
            return self._linear_bounds(prog.num, prog.a_eq, prog.b_eq)
        return self._fractional_bounds(prog.num, prog.den, prog.a_eq, prog.b_eq)

    def _linear_bounds(self, c, a_eq, b_eq):
        """min/max of a linear objective over the credal polytope (PNS)."""
        bounds = [(0.0, 1.0)] * c.size
        low = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
        upp = linprog(-c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not (low.success and upp.success):
            raise NoFeasibleSolution("Empirical law not reproducible by any P(U)")
        return [float(np.clip(low.fun, 0, 1)), float(np.clip(-upp.fun, 0, 1))]

    def _fractional_bounds(self, num, den, a_eq, b_eq):
        """Exact min/max of ``(num.theta)/(den.theta)`` via the Charnes-Cooper LP.

        Substitute ``y = theta/(den.theta)`` and ``t = 1/(den.theta)``: the ratio
        becomes linear -- optimise ``num.y`` s.t. ``a_eq y - b_eq t = 0``,
        ``den.y = 1``, ``y, t >= 0`` -- and the optimum is the ratio itself.
        """
        n = num.size
        a_cc = np.vstack([np.hstack([a_eq, -b_eq.reshape(-1, 1)]),   # a_eq y - b_eq t = 0
                          np.hstack([den, [0.0]])])                  # den.y = 1
        b_cc = np.concatenate([np.zeros(a_eq.shape[0]), [1.0]])
        c = np.concatenate([num, [0.0]])
        bounds = [(0.0, None)] * (n + 1)
        low = linprog(c, A_eq=a_cc, b_eq=b_cc, bounds=bounds, method="highs")
        upp = linprog(-c, A_eq=a_cc, b_eq=b_cc, bounds=bounds, method="highs")
        if not (low.success and upp.success):
            # Infeasible transform => denominator event has probability 0 for every
            # feasible P(U): the conditional query is undefined.
            if self._max_linear(den, a_eq, b_eq) < _DENOM_EPS:
                return [float("nan"), float("nan")]
            raise NoFeasibleSolution("Empirical law not reproducible by any P(U)")
        return [float(np.clip(low.fun, 0, 1)), float(np.clip(-upp.fun, 0, 1))]

    def _max_linear(self, c, a_eq, b_eq):
        res = linprog(-c, A_eq=a_eq, b_eq=b_eq, bounds=[(0, 1)] * c.size, method="highs")
        return -res.fun if res.success else 0.0


# --------------------------------------------------------------------- helpers

def _true_false(var, domain):
    """(true, false) states for ``var``: binary via bcause's rule, else (1, 0).

    credici's queries default to states 1 (true) and 0 (false); for the 3-state
    ``ysize3`` effects it uses the same 1/0, treating the query as "effect is 1"
    vs "effect is 0". Matched here.
    """
    if len(domain) == 2:
        return dutils.identify_true_false(var, domain)
    if 1 in domain and 0 in domain:
        return 1, 0
    raise ValueError(f"Cannot pick true/false states for {var} with domain {domain}")


def _round_normalize(probs: np.ndarray) -> np.ndarray:
    """Round to ``_DECIMALS`` and renormalise, like credici's fixEmpiricalMap."""
    rounded = np.round(probs, _DECIMALS)
    total = rounded.sum()
    return rounded / total if total > 0 else rounded


def _config_weight(marginals, assignment, variables) -> float:
    """Empirical ``P(variables = assignment)``, factorised over c-component groups.

    Groups outside ``variables`` contribute 1; partially covered groups are
    marginalised over their excluded members.
    """
    varset = set(variables)
    weight = 1.0
    for group, dist in marginals:
        sub = [i for i, v in enumerate(group) if v in varset]
        if not sub:
            continue
        if len(sub) == len(group):
            weight *= dist.get(tuple(assignment[v] for v in group), 0.0)
        else:
            target = tuple(assignment[group[i]] for i in sub)
            weight *= sum(p for key, p in dist.items()
                          if tuple(key[i] for i in sub) == target)
    return weight


# --------------------------------------------------------------------- demo / CLI

def _load(base: str):
    """Load ``<base>.uai`` + ``<base>.csv`` into (model, endogenous DataFrame)."""
    model = StructuralCausalModel.read(base + ".uai")
    data = pd.read_csv(base + ".csv", index_col=0).add_prefix("V")
    return model, data[model.endogenous]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Exact PN/PS/PNS bounds for one SCM.")
    ap.add_argument("base", nargs="?",
                    default="/Users/antoniogonzalezalves/Documents/s23/"
                            "simple_nparents2_nzr08_zdr10_37",
                    help="Model path WITHOUT extension (expects <base>.uai and <base>.csv).")
    ap.add_argument("queries", nargs="*", default=["PS(V1,V0)", "PN(V1,V0)", "PNS(V1,V0)"],
                    help='e.g. "PS(V1,V0)" "PNS(V2,V0)"  (case-insensitive).')
    args = ap.parse_args()

    model, data = _load(args.base)
    inf = ExactCredalCausalInference(model, data)
    print(f"model: {Path(args.base).name}   endogenous={model.endogenous}")
    import re
    for q in args.queries:
        kind, cause, effect = re.match(r"([A-Za-z]+)\((\w+),(\w+)\)", q).groups()
        lo, up = inf.bounds(kind.upper(), cause, effect)
        print(f"  {kind.upper()}({cause},{effect}) = [{lo:.5f}, {up:.5f}]")
