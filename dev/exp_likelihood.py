"""
Creates an adjustable Markovian model (n_levels), sample from it, and estimate 
its parameters

TODO:
- sample several time for given sample size n and see the likelihood values
- change n
- change n_levels
"""

import networkx as nx

from bcause.factors import MultinomialFactor, DeterministicFactor 
from bcause.learning.parameter.gradient import GradientLikelihood
from bcause.models.cmodel import StructuralCausalModel
from bcause.factors.deterministic import canonical_specification


def create_markovian_DAG(n_levels: int, n_endo_states: list[int] = None):
    # creates a Markovian graph with n_levels of exo and endo nodes
    if n_endo_states == None:
        n_endo_states = [2] * n_levels # binary if not specified
    edges, domains= [], {}
    for level in range(n_levels):
        exo = f'U{level}'
        endo = f'X{level}'
        edges.append((exo, endo))
        domains[endo] = [f'x{level}{i}' for i in range(n_endo_states[level])]
        if level == 0: 
            # NOTE: X0 doesn't have any endogenous parent
            domains[exo] = [f'u{level}{i}' for i in range(n_endo_states[level])]
        else:
            endo_parent = f'X{level-1}'
            edges.append((endo_parent, endo))
            domains[exo] = [f'u{level}{i}' for i in range(n_endo_states[level] ** n_endo_states[level])]
    return edges, domains

n_levels = 3
edges, domains = create_markovian_DAG(n_levels = n_levels)
dag = nx.DiGraph(edges)

import bcause.util.domainutils as dutils
import bcause.util.graphutils as gutils

p, f = {}, {}
for level in range(n_levels):
    exo = f'U{level}'
    endo = f'X{level}'
    dom_endo = dutils.subdomain(domains, *gutils.relevat_vars(dag, endo))
    dom_exo = dutils.subdomain(domains, exo)
    if level == 0:
        p[exo] = MultinomialFactor(dom_exo, values=[.3, .7])
        f[endo] = DeterministicFactor(dom_endo, left_vars=[endo], values=domains[endo])
    else:
        p[exo] = MultinomialFactor(dom_exo, values=[.2, .2, .1, .5])
        endo_parent = f'X{level-1}'
        can_values = canonical_specification(V_domain = domains[endo], Y_domains = [domains[endo_parent]])
        f[endo] = DeterministicFactor(dom_endo, left_vars=[endo], values=can_values)    


m = StructuralCausalModel(dag, list(f.values()) + list(p.values()), cast_multinomial=True)

data = m.sample(1000, as_pandas=True)[m.endogenous]

gl = GradientLikelihood(m.randomize_factors(m.exogenous, allow_zero=False))
gl.run(data)

