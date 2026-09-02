"""General exact credal solver for (semi-)Markovian SCMs -- PN / PS / PNS bounds.

Generalises ``exact_credal_solver`` from "one confounder directly under the
effect" to "the whole c-component(s) spanned by the cause and effect", so it also
covers the SEMI-Markovian models obtained by merging exogenous variables (where a
confounder is shared between endogenous variables).

--------------------------------------------------------------------------------
METHOD
--------------------------------------------------------------------------------
Free confounders (the underdetermined latent variables) are optimised; every
other quantity is fixed by the data.

  * FREE confounders  U  = the exogenous parents of the endogenous c-component(s)
    that contain the cause and the effect.  theta = P(U) over the JOINT of their
    states is the LP variable.
  * EXTERNAL parents  E  = endogenous parents of those c-component nodes that lie
    OUTSIDE the component (point-identified roots); marginalised with their
    empirical law, factorised over c-components.
  * For a joint exogenous state ``u`` and an external configuration ``e`` the
    whole component is deterministic: propagate the structural equations in
    topological order to read every relevant endogenous value, and likewise the
    two intervened worlds ``do(X = true)`` and ``do(X = false)``.

Then, exactly as in the Markovian case:
  * credal-set constraints  sum_u theta_u [response(u,e) = r] = P_emp(component=r | E=e)
    are LINEAR in theta;
  * PNS is LINEAR  -> two LPs;  PN / PS are LINEAR-FRACTIONAL -> Charnes-Cooper LP.

Exactness: when the cause and effect share a SINGLE free confounder (the usual
semi-Markovian merge that confounds them) this is the exact tight bound. When a
merge leaves the cause and effect under two INDEPENDENT free confounders, taking
their joint as free relaxes that independence, so the result is a valid OUTER
bound (it still contains the truth) rather than tight -- credici enforces the
independence and can be tighter there. The Markovian case is the special case
"component = {effect}", and this reproduces ``exact_credal_solver`` exactly.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bcause.util.domainutils as dutils  # noqa: E402
from bcause.models.cmodel import StructuralCausalModel  # noqa: E402

_DECIMALS = 5
_DENOM_EPS = 1e-12


class NoFeasibleSolution(Exception):
    """No P(U) reproduces the empirical law (credici's NoFeasibleSolutionException)."""


@dataclass
class Program:
    kind: str                 # "linear" (PNS) | "fractional" (PN/PS)
    a_eq: np.ndarray
    b_eq: np.ndarray
    num: np.ndarray
    den: Optional[np.ndarray]


class ExactCredalSCM:
    """Exact/outer PN-PS-PNS bounds for a (semi-)Markovian SCM constrained by data."""

    def __init__(self, model: StructuralCausalModel, data: pd.DataFrame):
        self._m = model.fix_numeric_domains()
        self._dom = self._m.domains
        self._data = data
        self._detcache: dict = {}   # (var, parents) -> deterministic output tensor
        # Domains are contiguous ints 0..k-1 (bcause UAI loader), so a variable's
        # value equals its index -- used for the vectorised tensor gather below.

    # ------------------------------------------------------------------ public

    def bounds(self, query, cause, effect, tf_cause=None, tf_effect=None):
        return self._solve(self.program(query, cause, effect, tf_cause, tf_effect))

    def prob_sufficiency(self, cause, effect, tf_cause=None, tf_effect=None):
        return self.bounds("PS", cause, effect, tf_cause, tf_effect)

    def prob_necessity(self, cause, effect, tf_cause=None, tf_effect=None):
        return self.bounds("PN", cause, effect, tf_cause, tf_effect)

    def prob_necessity_sufficiency(self, cause, effect, tf_cause=None, tf_effect=None):
        return self.bounds("PNS", cause, effect, tf_cause, tf_effect)

    # --------------------------------------------------------------- assembly

    def is_feasible(self) -> bool:
        """Whether SOME P(exogenous) reproduces the data (credici's global check).

        Every endogenous c-component's confounder must be able to match that
        component's empirical law; if any component's credal set is empty the
        merged model is infeasible (credici raises NoFeasibleSolutionException).
        Used by the semi-Markovian generator to trigger dataset resampling.
        """
        for comp in self._m.endo_ccomponents:
            order, _, ext_space, obs, cond, n, _, _ = self._setup(set(comp))
            a_eq, b_eq = self._constraints(obs, ext_space, order, cond, n)
            res = linprog(np.zeros(n), A_eq=a_eq, b_eq=b_eq,
                          bounds=[(0.0, 1.0)] * n, method="highs")
            if not res.success:
                return False
        return True

    def _setup(self, component):
        """Structure + factual responses for one c-component (vectorised).

        Returns ``(order, external, ext_space, obs, cond, n, exo_arr, free_exo)``
        where ``obs[ei]`` is the (n_states, |component|) factual component response
        for the ei-th external configuration, and ``exo_arr`` holds the joint
        confounder states (n_states, |free_exo|).
        """
        m = self._m
        order = [v for v in nx.topological_sort(m.graph) if v in component]
        free_exo = sorted({u for v in component for u in m.get_exogenous_parents(v)}, key=str)
        external = sorted({p for v in component for p in m.get_edogenous_parents(v)
                           if p not in component}, key=str)
        # Confounder states as INDICES (0..card-1) per free exogenous. The tensor
        # gather needs the domain position, and merged confounders carry
        # non-contiguous string labels (e.g. '3_1'), so we never use their values.
        free_cards = [len(self._dom[u]) for u in free_exo]
        exo_arr = np.array(list(itertools.product(*[range(c) for c in free_cards])),
                           dtype=np.int64) if free_exo else np.zeros((1, 0), dtype=np.int64)
        ext_space = dutils.assingment_space(dutils.subdomain(self._dom, *external)) \
            if external else [{}]
        obs = {ei: self._world(order, free_exo, exo_arr, e, {})
               for ei, e in enumerate(ext_space)}
        cond = self._conditional(order, external)
        return order, external, ext_space, obs, cond, len(exo_arr), exo_arr, free_exo

    def _world(self, order, free_exo, exo_arr, ext_assign, do):
        """Vectorised deterministic values of ``order`` over all confounder states.

        Propagates the structural equations in topological order by numpy tensor
        gather: each endogenous is its factor's arg-max output looked up at the
        (already-computed) parent values. Returns an (n_states, |order|) array.
        """
        vals = {u: exo_arr[:, i] for i, u in enumerate(free_exo)}
        n = exo_arr.shape[0]
        for v, x in {**ext_assign, **do}.items():
            vals[v] = np.full(n, x, dtype=np.int64)
        cols = []
        for v in order:
            if v in do:
                cols.append(vals[v])
                continue
            f = self._m.factors[v]
            parents = tuple(f.right_vars)
            tensor = self._detcache.get((v, parents))
            if tensor is None:
                tensor = _deterministic_tensor(f, parents)   # arg-max output per parent config
                self._detcache[(v, parents)] = tensor
            flat = np.ravel_multi_index([vals[p] for p in parents], tensor.shape)
            vals[v] = tensor.reshape(-1)[flat].astype(np.int64)
            cols.append(vals[v])
        return np.stack(cols, axis=1)

    def program(self, query, cause, effect, tf_cause=None, tf_effect=None) -> Program:
        m = self._m
        Tc, Fc = tf_cause or _true_false(cause, self._dom[cause])
        Te, Fe = tf_effect or _true_false(effect, self._dom[effect])

        # --- structure: the EFFECT's c-component and its free confounder(s) ---
        # The cause is handled jointly if it shares the effect's confounder
        # (semi-Markovian merge), otherwise as an external covariate (Markovian).
        component = next((set(c) for c in m.endo_ccomponents if effect in c), {effect})
        order, external, ext_space, obs, cond, n_states, exo_arr, free_exo = self._setup(component)
        cause_inside = cause in component
        if not cause_inside and cause not in external:
            raise ValueError(f"{cause} is neither in {effect}'s component nor a parent of it")
        idx = {v: i for i, v in enumerate(order)}
        eff_col = idx[effect]

        def eff_under(e, do):
            """Effect value over all confounder states for external ``e`` and interventions ``do``."""
            return self._world(order, free_exo, exo_arr, e, do)[:, eff_col]

        def cause_set(e, value):
            """(external e, do) that force the cause to ``value``, inside or outside."""
            if cause_inside:
                return e, {cause: value}
            return {**e, cause: value}, {}

        # `obs`/`cond` come from _setup; build the credal constraints + weights.
        weights = self._marginals(external)
        a_eq, b_eq = self._constraints(obs, ext_space, order, cond, n_states)

        ce = idx[effect]
        # `others` = externals whose distribution we still average over (all but an
        # intervened external cause). For PNS the cause is always intervened.
        others = [v for v in external if v != cause] if not cause_inside else external

        if query == "PNS":
            c = np.zeros(n_states)
            for e in ext_space:
                w = _config_weight(weights, e, others)
                if not w:
                    continue
                eT, doT = cause_set(e, Tc)
                eF, doF = cause_set(e, Fc)
                c += w * (eff_under(eT, doT) == Te) * (eff_under(eF, doF) == Fe)
            return Program("linear", a_eq, b_eq, c, None)

        # PN / PS: condition on (cause = obs_c, effect = obs_e); counterfactual do(cause=cf_c)
        obs_c, obs_e = (Tc, Te) if query == "PN" else (Fc, Fe)
        cf_c, cf_e = (Fc, Fe) if query == "PN" else (Tc, Te)
        num = np.zeros(n_states)
        den = np.zeros(n_states)
        for ei, e in enumerate(ext_space):
            w = _config_weight(weights, e, external)
            if not w:
                continue
            cause_fact = obs[ei][:, idx[cause]] if cause_inside \
                else np.full(n_states, e[cause])
            factual = w * (cause_fact == obs_c) * (obs[ei][:, ce] == obs_e)
            if not factual.any():
                continue
            e_cf, do_cf = cause_set(e, cf_c)
            den += factual
            num += factual * (eff_under(e_cf, do_cf) == cf_e)
        return Program("fractional", a_eq, b_eq, num, den)

    def _constraints(self, obs, ext_space, order, cond, n_states):
        """``sum_u theta_u [component(u,e)=r] = P_emp(component=r | E=e)`` + simplex.

        The equalities are pinned against the FULL component configuration space
        (all but one config per external context; the last follows from the
        simplex), so unobserved configs are correctly forced to probability 0 --
        not left free, which would loosen the bound.
        """
        component_configs = list(itertools.product(*[self._dom[v] for v in order]))
        rows, rhs = [], []
        for ei, e in enumerate(ext_space):
            table = cond.get(tuple(e.values()), None)
            if table is None:            # external context never observed -> no info
                continue
            for r in component_configs[:-1]:      # pin all but one; last is implied
                mask = np.all(obs[ei] == np.array(r), axis=1).astype(float)
                rows.append(mask)
                rhs.append(table.get(r, 0.0))
        rows.append(np.ones(n_states))            # probabilities sum to one
        rhs.append(1.0)
        return np.asarray(rows), np.asarray(rhs)

    def _conditional(self, order, external):
        """Empirical, 5-dp-rounded P(component=r | E=e), keyed by external tuple."""
        cols = external + order
        counts = self._data.groupby(cols).size()
        table: dict = {}
        for idx, n in counts.items():
            idx = idx if isinstance(idx, tuple) else (idx,)
            e_key = idx[:len(external)]
            r_key = idx[len(external):]
            table.setdefault(e_key, {})[r_key] = n
        for e_key, dist in table.items():
            keys = list(dist)
            probs = _round_normalize(np.array([dist[k] for k in keys], float))
            table[e_key] = dict(zip(keys, probs))
        return table

    def _marginals(self, external):
        """Empirical law of the external parents, factorised over c-components."""
        parents = set(external)
        out = []
        for comp in self._m.endo_ccomponents:
            group = [p for p in external if p in comp and p in parents]
            if not group:
                continue
            counts = self._data.groupby(group).size()
            probs = _round_normalize(counts.to_numpy(dtype=float))
            keys = [k if len(group) > 1 else (k,) for k in counts.index]
            out.append((group, dict(zip(keys, probs))))
        return out

    # ------------------------------------------------------------------ solving

    def _solve(self, prog: Program):
        if prog.kind == "linear":
            bnds = [(0.0, 1.0)] * prog.num.size
            lo = linprog(prog.num, A_eq=prog.a_eq, b_eq=prog.b_eq, bounds=bnds, method="highs")
            hi = linprog(-prog.num, A_eq=prog.a_eq, b_eq=prog.b_eq, bounds=bnds, method="highs")
            if not (lo.success and hi.success):
                raise NoFeasibleSolution("empirical law not reproducible")
            return [float(np.clip(lo.fun, 0, 1)), float(np.clip(-hi.fun, 0, 1))]

        num, den, a_eq, b_eq = prog.num, prog.den, prog.a_eq, prog.b_eq
        n = num.size
        a_cc = np.vstack([np.hstack([a_eq, -b_eq.reshape(-1, 1)]),
                          np.hstack([den, [0.0]])])
        b_cc = np.concatenate([np.zeros(a_eq.shape[0]), [1.0]])
        c = np.concatenate([num, [0.0]])
        bnds = [(0.0, None)] * (n + 1)
        lo = linprog(c, A_eq=a_cc, b_eq=b_cc, bounds=bnds, method="highs")
        hi = linprog(-c, A_eq=a_cc, b_eq=b_cc, bounds=bnds, method="highs")
        if not (lo.success and hi.success):
            m = linprog(-den, A_eq=a_eq, b_eq=b_eq, bounds=[(0, 1)] * n, method="highs")
            if m.success and -m.fun < _DENOM_EPS:
                return [float("nan"), float("nan")]
            raise NoFeasibleSolution("empirical law not reproducible")
        return [float(np.clip(lo.fun, 0, 1)), float(np.clip(-hi.fun, 0, 1))]


# --------------------------------------------------------------------- helpers

def _deterministic_tensor(factor, parents) -> np.ndarray:
    """Arg-max output index of a (degenerate) factor per parent configuration.

    Reads the factor's raw numpy store and takes the arg-max over the effect axis
    directly -- avoiding bcause's cell-by-cell ``values_array``, which is orders
    of magnitude slower on the large merged confounders.
    """
    data = np.asarray(factor.store.data)
    svars = list(factor.store.variables)
    left = factor.left_vars[0]
    out = data.argmax(axis=svars.index(left))                 # over right vars, store order
    right_order = [v for v in svars if v != left]
    return np.transpose(out, [right_order.index(p) for p in parents])


def _true_false(var, domain):
    if len(domain) == 2:
        return dutils.identify_true_false(var, domain)
    if 1 in domain and 0 in domain:
        return 1, 0
    raise ValueError(f"cannot pick true/false for {var} domain {domain}")


def _round_normalize(probs):
    rounded = np.round(probs, _DECIMALS)
    total = rounded.sum()
    return rounded / total if total > 0 else rounded


def _config_weight(marginals, assignment, variables):
    if not variables:
        return 1.0
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
            weight *= sum(p for k, p in dist.items()
                          if tuple(k[i] for i in sub) == target)
    return weight


def _load(base):
    model = StructuralCausalModel.read(base + ".uai")
    data = pd.read_csv(base + ".csv", index_col=0).add_prefix("V")
    return model, data[model.endogenous]
