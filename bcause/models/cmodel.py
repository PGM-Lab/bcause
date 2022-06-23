from typing import Dict, Union, Hashable

import networkx as nx

import bcause.util.domainutils as dutils
import bcause.util.graphutils as gutils

from bcause.factors import DeterministicFactor
from bcause.factors.mulitnomial import random_multinomial, MultinomialFactor
from bcause.models.pgmodel import DiscreteDAGModel


import bcause.factors.factor as bf


class DiscreteCausalDAGModel(DiscreteDAGModel):

    @property
    def endogenous(self):
        return self._endogenous

    @property
    def exogenous(self):
        return [v for v in self.graph.nodes if v not in self.endogenous]

    def is_endogenous(self, x):
        return len(list(self.graph.predecessors(x)))>0

    def is_exogenous(self, x):
        return len(list(self.graph.predecessors(x)))==0

    def get_edogenous_parents(self, variable):
        return [x for x in m.get_parents(variable) if m.is_endogenous(x)]
    def get_edogenous_children(self, variable):
        return [x for x in m.get_children(variable) if m.is_endogenous(x)]
    def get_exogenous_parents(self, variable):
        return [x for x in m.get_parents(variable) if m.is_exogenous(x)]
    def get_exogenous_children(self, variable):
        return [x for x in m.get_children(variable) if m.is_exogenous(x)]


class CausalModel(DiscreteCausalDAGModel):
    def __init__(self, dag:Union[nx.DiGraph,str], factors:Union[dict,list] = None, cast_multinomial = False):

        self._initialize(dag)
        self._endogenous = [x for x in dag if len(list(dag.predecessors(x)))>0]
        self._cast_multinomial = cast_multinomial
        if factors is not None:
            self._set_factors(factors)

        def builder(*args, **kwargs): return CausalModel(*args, **kwargs)
        self.builder = builder

    def set_factor(self, var:Hashable, f:bf.DiscreteFactor):
        # SCM related check
        if self.is_endogenous(var):
            if not(isinstance(f, DeterministicFactor) or f.is_degenerate()):
                raise ValueError("Factor must be deterministic or degenerate")
            if self._cast_multinomial and isinstance(f, DeterministicFactor):
                f = f.to_multinomial()

        super().set_factor(var, f)

    def __repr__(self):
        str_card = ",".join([f"{str(v)}:{'' if d is None else str(len(d))}" for v, d in self._domains.items()])
        return f"<CausalModel ({str_card}), dag={gutils.dag2str(self.graph)}>"


if __name__ == "__main__":

    dag = nx.DiGraph([("Y" ,"X"), ("V" ,"Y"), ("U" ,"X")])
    domains = dict(X=["x1", "x2"], Y=[0, 1], U=["u1", "u2", "u3", "u4"], V=[True, False])

    import bcause.factors.factor as bf

    domy = dutils.subdomain(domains, *gutils.relevat_vars(dag, "Y"))
    fy = DeterministicFactor(domy, right_vars = ["V"], data=[1,0])

    domx = dutils.subdomain(domains, *gutils.relevat_vars(dag, "X"))

    # todo: check data shape for this
    data = ["x1", "x1", "x2", "x1","x1", "x1", "x2", "x1"]
    fx = DeterministicFactor(domx, left_vars = ["X"], data=data)

    domv = dutils.subdomain(domains, "V")
    pv = MultinomialFactor(domv, data = [.1, .9])

    domu = dutils.subdomain(domains, "U")
    pu = MultinomialFactor(domu, data = [.2, .2, .1, .5])



    m = CausalModel(dag, [fx, fy, pu, pv], cast_multinomial=False)


#


#causal_to_bnet


