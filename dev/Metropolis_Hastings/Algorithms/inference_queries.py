"""
Fast, vectorized probability-of-sufficiency / necessity queries over an ensemble
of structural causal models that share the same structural equations.

The samplers in this project only update the exogenous (latent) distributions
P(U); the endogenous mechanisms are fixed deterministic equations. For such an
ensemble a counterfactual query need not build a twin network and run variable
elimination per model. Because every endogenous variable is a deterministic
function of the latent variables, the probability of sufficiency/necessity of a
single model is

    PS(theta) = sum_u num(u) P(u; theta) / sum_u den(u) P(u; theta)

where ``num``/``den`` are 0/1 masks that depend only on the (shared) structural
equations and ``P(u; theta) = prod_i theta_{U_i}[u_i]`` is the latent joint of
that model. The masks are precomputed once; the ensemble estimate is then a
couple of matrix products over all models and every relevant latent state at
once, and the interval is the (optionally IQR-trimmed) min/max across models.

This mirrors ``CausalMultiInference.prob_sufficiency`` /
``prob_necessity`` but is orders of magnitude faster for large ensembles.
"""

from itertools import product
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

import bcause.util.domainutils as dutils
from bcause.models.cmodel import StructuralCausalModel
from bcause.util.arrayutils import min_max_iqr


class InferenceQueries:
    """
    Vectorized PS/PN estimator for a list of SCMs sharing structural equations.

    Args:
        models: SCMs that differ only in their exogenous (latent) factors.
        outliers_removal: if True (default) the returned interval is the IQR
            trimmed min/max across models, matching CausalMultiInference.

    The heavy per-ensemble work (extracting the deterministic equations and
    stacking the latent tables) happens once in the constructor; each query is
    then a set of array operations, cached per (cause, effect) pair.
    """

    def __init__(self, models: List[StructuralCausalModel], outliers_removal: bool = True):
        if not models:
            raise ValueError("At least one model is required")
        self._models = models
        self._outliers_removal = outliers_removal

        reference = models[0]
        self._graph = reference.graph
        self._endogenous = set(reference.endogenous)
        self._exogenous = list(reference.exogenous)
        self._domains = reference.domains

        # Deterministic response of every endogenous variable, as a lookup table
        # func[parent_value_tuple] -> child index, plus its ordered parent list.
        self._det_func: Dict[str, np.ndarray] = {}
        self._det_parents: Dict[str, List[str]] = {}
        for v in self._endogenous:
            factor = reference.factors[v]
            axes = list(factor.variables)
            child_axis = axes.index(v)
            # Collapsing the one-hot child axis with argmax yields the equation.
            self._det_func[v] = np.asarray(factor.values_array()).argmax(axis=child_axis)
            self._det_parents[v] = [x for x in axes if x != v]

        # Stack each latent table across models once: {U: (n_models, card_U)}.
        self._theta = {
            u: np.stack([np.asarray(m.factors[u].values_array(), dtype=float) for m in models])
            for u in self._exogenous
        }
        # Cache of the precomputed structure per (cause, effect) pair.
        self._plan_cache: Dict[Tuple[str, str], dict] = {}

    def prob_sufficiency(self, cause, effect, true_false_cause: tuple = None,
                         true_false_effect: tuple = None) -> list:
        """PS = P(effect_{do(cause=T)} = T | cause=F, effect=F), as a [low, upp] interval."""
        tc, fc, te, fe = self._states(cause, effect, true_false_cause, true_false_effect)
        # Evidence cause=F, effect=F; intervene cause=T; count effect_cf=T.
        return self._interval(cause, effect, factual_cause=fc, factual_effect=fe,
                              do_cause=tc, cf_effect=te)

    def prob_necessity(self, cause, effect, true_false_cause: tuple = None,
                       true_false_effect: tuple = None) -> list:
        """PN = P(effect_{do(cause=F)} = F | cause=T, effect=T), as a [low, upp] interval."""
        tc, fc, te, fe = self._states(cause, effect, true_false_cause, true_false_effect)
        # Evidence cause=T, effect=T; intervene cause=F; count effect_cf=F.
        return self._interval(cause, effect, factual_cause=tc, factual_effect=te,
                              do_cause=fc, cf_effect=fe)

    def _states(self, cause, effect, true_false_cause, true_false_effect) -> tuple:
        """Resolve true/false state values to domain indices for cause and effect."""
        tcause, fcause = true_false_cause or dutils.identify_true_false(cause, self._domains[cause])
        teffect, feffect = true_false_effect or dutils.identify_true_false(effect, self._domains[effect])
        return (self._domains[cause].index(tcause), self._domains[cause].index(fcause),
                self._domains[effect].index(teffect), self._domains[effect].index(feffect))

    def _plan(self, cause: str, effect: str) -> dict:
        """
        Precompute (and cache) everything about a (cause, effect) pair that does
        not depend on the latent tables: the relevant latent variables, their
        enumerated joint states, and the per-state factual/counterfactual values
        of cause and effect under both possible interventions on the cause.
        """
        key = (cause, effect)
        if key in self._plan_cache:
            return self._plan_cache[key]

        # Only the ancestors of cause/effect matter; other latents cancel out.
        relevant = nx.ancestors(self._graph, cause) | nx.ancestors(self._graph, effect) | {cause, effect}
        exo = [u for u in self._exogenous if u in relevant]
        endo_order = [v for v in nx.topological_sort(self._graph.subgraph(relevant))
                      if v in self._endogenous]

        # Enumerate the joint latent state space of the relevant exogenous vars.
        cards = [len(self._domains[u]) for u in exo]
        assign = np.array(list(product(*[range(c) for c in cards])), dtype=np.intp).reshape(-1, len(exo))

        def evaluate(forced: dict) -> dict:
            # Propagate the deterministic equations over all latent states at once.
            values = {u: assign[:, j] for j, u in enumerate(exo)}
            for v in endo_order:
                if v in forced:
                    values[v] = np.full(assign.shape[0], forced[v], dtype=np.intp)
                else:
                    parents = self._det_parents[v]
                    values[v] = self._det_func[v][tuple(values[p] for p in parents)]
            return values

        factual = evaluate({})
        plan = {
            "exo": exo, "assign": assign,
            "cause_factual": factual[cause], "effect_factual": factual[effect],
            # Counterfactual effect under each possible do(cause=state).
            "effect_do": {state: evaluate({cause: state})[effect]
                          for state in range(len(self._domains[cause]))},
        }
        self._plan_cache[key] = plan
        return plan

    def _interval(self, cause, effect, factual_cause, factual_effect, do_cause, cf_effect) -> list:
        """Vectorized ensemble PS/PN: build the masks, weight by each model's latent joint."""
        plan = self._plan(cause, effect)
        exo, assign = plan["exo"], plan["assign"]

        # 0/1 masks over latent states: denominator (evidence) and numerator.
        den = (plan["cause_factual"] == factual_cause) & (plan["effect_factual"] == factual_effect)
        num = den & (plan["effect_do"][do_cause] == cf_effect)
        den = den.astype(float)
        num = num.astype(float)

        # Per-model latent joint over every relevant state: weights[model, state].
        weights = np.ones((len(self._models), assign.shape[0]))
        for j, u in enumerate(exo):
            weights *= self._theta[u][:, assign[:, j]]

        den_mass = weights @ den
        num_mass = weights @ num
        # Ratio only where the evidence has positive mass under that model.
        valid = den_mass > 0
        if not valid.any():
            return [float("nan"), float("nan")]
        ratios = num_mass[valid] / den_mass[valid]

        if self._outliers_removal:
            return list(min_max_iqr(ratios))
        return [float(ratios.min()), float(ratios.max())]
