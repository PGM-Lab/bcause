import numpy as np
import pgmpy.models as pm
import pgmpy.factors.discrete as pfd
from networkx import DiGraph

import bcause.models as bm
import bcause.factors as bfd
from util.domainutils import assingment_space


def toMultinomialFactor(factor : pfd.TabularCPD) -> bfd.MultinomialFactor:
    domain = factor.state_names
    data = np.reshape([factor.get_value(**s) for s in assingment_space(domain)], factor.cardinality)
    right_vars = [v for v in factor.variables if v != factor.variable]
    return bfd.MultinomialFactor(domain, data, right_vars)

def toTabularCPT(f : bfd.MultinomialFactor) -> pfd.TabularCPD:
    v = list(f.left_domain.keys())[0]
    card = f.store.cardinality_dict
    args = dict(
        variable = v,
        variable_card = card[v],
        values = f.values_array.T,
        state_names = f.domain
    )

    parents = f.right_vars
    if len(parents)>0:
        args["evidence"] = parents
        args["evidence_card"] = [card[p] for p in parents]

    return pfd.TabularCPD(**args)


def toBCauseBNet(orig : pm.BayesianNetwork) -> bm.BayesianNetwork:
    dag = DiGraph(orig.in_edges)
    factors = {f.variable:toMultinomialFactor(f) for f in orig.cpds}
    return bm.BayesianNetwork(dag, factors)


def toPgmpyBNet(orig : bm.BayesianNetwork) -> pm.BayesianNetwork:
    dest = pm.BayesianNetwork(orig.network)
    dest.add_cpds(*[toTabularCPT(f) for f in orig.factors.values()])
    return dest