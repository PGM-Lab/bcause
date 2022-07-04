from __future__ import annotations

import logging
from typing import Dict, Union, Hashable, Iterable

import networkx as nx
from networkx import relabel_nodes

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
        return x in self.endogenous

    def is_exogenous(self, x):
        return x not in self.endogenous

    def get_edogenous_parents(self, variable):
        return [x for x in m.get_parents(variable) if m.is_endogenous(x)]
    def get_edogenous_children(self, variable):
        return [x for x in m.get_children(variable) if m.is_endogenous(x)]
    def get_exogenous_parents(self, variable):
        return [x for x in m.get_parents(variable) if m.is_exogenous(x)]
    def get_exogenous_children(self, variable):
        return [x for x in m.get_children(variable) if m.is_exogenous(x)]


class StructuralCausalModel(DiscreteCausalDAGModel):
    def __init__(self, dag:Union[nx.DiGraph,str], factors:Union[dict,list] = None, endogenous:Iterable = None, cast_multinomial = True):

        self._initialize(dag)
        self._endogenous = endogenous or [x for x in dag if len(list(dag.predecessors(x)))>0]
        self._cast_multinomial = cast_multinomial
        if factors is not None:
            self._set_factors(factors)

    def builder(self, **kwargs):
        if "cast_multinomial" not in kwargs: kwargs["cast_multinomial"] = self._cast_multinomial
        return StructuralCausalModel(**kwargs)


    def set_factor(self, var:Hashable, f:bf.DiscreteFactor):
        # SCM related check
        if self.is_endogenous(var):
            if not(isinstance(f, DeterministicFactor) or f.is_degenerate()):
                raise ValueError("Factor must be deterministic or degenerate")
            if self._cast_multinomial and isinstance(f, DeterministicFactor):
                f = f.to_multinomial()

        super().set_factor(var, f)

    def to_multinomial(self) -> StructuralCausalModel:
        return StructuralCausalModel(dag=self.graph, factors=self.factors, cast_multinomial=True)

    def intervention(self, **obs):
        new_dag = gutils.remove_ingoing_edges(self.graph, obs.keys())
        new_factors = dict()
        for v, f in self.factors.items():
            new_factors[v] = f if v not in obs else f.constant(obs[v])
        return StructuralCausalModel(dag=new_dag, factors=new_factors, endogenous=self.endogenous, cast_multinomial=self._cast_multinomial)

    def rename_vars(self, names_mapping: dict) -> DiscreteDAGModel:
        logging.debug(f"Renaming variables as {names_mapping}")
        new_dag = relabel_nodes(self.graph, names_mapping)
        new_factors = [f.rename_vars(names_mapping) for f in self.factors.values()]
        new_endogenous = [names_mapping[x] for x in self.endogenous]
        return StructuralCausalModel(dag=new_dag, factors=new_factors, endogenous=new_endogenous, cast_multinomial=self._cast_multinomial)



    def __repr__(self):
        str_card_endo = ",".join([f"{str(v)}:{'' if d is None else str(len(d))}"
                                  for v, d in self._domains.items() if self.is_endogenous(v)])
        str_card_exo = ",".join([f"{str(v)}:{'' if d is None else str(len(d))}"
                                  for v, d in self._domains.items() if self.is_exogenous(v)])
        return f"<StructuralCausalModel ({str_card_endo}|{str_card_exo}), dag={gutils.dag2str(self.graph)}>"


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

    m = StructuralCausalModel(dag, [fx, fy, pu, pv], endogenous=["X"], cast_multinomial=False)


    model = m
#


#causal_to_bnet


