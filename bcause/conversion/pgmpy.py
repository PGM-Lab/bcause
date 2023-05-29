import numpy as np
import pgmpy.models as pm
import pgmpy.factors.discrete as pfd
from networkx import DiGraph

import bcause.models as bm
import bcause.factors as bfd
from bcause.util.domainutils import assingment_space


def toMultinomialFactor(factor : pfd.TabularCPD, vtype="numpy") -> bfd.MultinomialFactor:
    domain = factor.state_names
    data = np.reshape([factor.get_value(**s) for s in assingment_space(domain)], factor.cardinality)
    right_vars = [v for v in factor.variables if v != factor.variable]
    return bfd.MultinomialFactor(domain, data, right_vars=right_vars, vtype=vtype)

def toTabularCPT(f : bfd.MultinomialFactor) -> pfd.TabularCPD:
    v = list(f.left_domain.keys())[0]
    card = f.store.cardinality_dict
    values = f.to_values_array()
    if np.ndim(values)<2:
        values = np.expand_dims(values, axis=0)

    args = dict(
        variable = v,
        variable_card = card[v],
        values = values.T,
        state_names = f.domain
    )

    parents = f.right_vars
    if len(parents)>0:
        args["evidence"] = parents
        args["evidence_card"] = [card[p] for p in parents]

    return pfd.TabularCPD(**args)


def toBCauseBNet(orig : pm.BayesianNetwork, vtype="numpy") -> bm.BayesianNetwork:
    dag = DiGraph(orig.in_edges)
    factors = {f.variable:toMultinomialFactor(f, vtype) for f in orig.cpds}
    return bm.BayesianNetwork(dag, factors)


def toPgmpyBNet(orig : bm.BayesianNetwork) -> pm.BayesianNetwork:
    dest = pm.BayesianNetwork(orig.graph)
    dest.add_cpds(*[toTabularCPT(f) for f in orig.factors.values()])
    return dest