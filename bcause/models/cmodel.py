from __future__ import annotations

import itertools
import logging
from typing import Dict, Union, Hashable, Iterable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from networkx import relabel_nodes, DiGraph, topological_sort

import networkx as nx
import bcause.models.info as info

from bcause.factors import DeterministicFactor
from bcause.factors.mulitnomial import random_multinomial, MultinomialFactor, random_deterministic
from bcause.models import BayesianNetwork
from bcause.models.pgmodel import DiscreteDAGModel
import bcause.util.domainutils as dutils
import bcause.util.graphutils as gutils

import bcause.factors.factor as bf


class DiscreteCausalDAGModel(DiscreteDAGModel):
    '''
    Parent class for the causal models over discrete variables with a directed acyclic structure
    '''


    @property
    def endogenous(self):
        '''
        Endogenous variables
        :return: list with all the endogenous variables in the model
        '''
        return self._endogenous

    @property
    def exogenous(self):
        '''
        Exogenous variables
        :return: list with all the exogenous variables in the model
        '''
        return [v for v in self.graph.nodes if v not in self.endogenous]

    def is_endogenous(self, x):
        return x in self.endogenous

    def is_exogenous(self, x):
        return x not in self.endogenous

    def get_edogenous_parents(self, variable):
        return [x for x in self.get_parents(variable) if self.is_endogenous(x)]
    def get_edogenous_children(self, variable):
        return [x for x in self.get_children(variable) if self.is_endogenous(x)]
    def get_exogenous_parents(self, variable):
        return [x for x in self.get_parents(variable) if self.is_exogenous(x)]
    def get_exogenous_children(self, variable):
        return [x for x in self.get_children(variable) if self.is_exogenous(x)]


    @property
    def endo_graph(self):
        '''
        Graph with only the endogenous variables
        :return: object of class nx.DiGraph
        '''
        return self.graph.subgraph(self.endogenous)

    @property
    def exo_graph(self):
        '''
        Graph with only the exogenous variables
        :return: object of class nx.DiGraph
        '''
        exo_edges = [(x,y) for x,y in self.graph.edges if self.is_exogenous(x)]
        return DiGraph(self.graph.edge_subgraph(exo_edges).edges)

    @property
    def ccomponents(self):
        '''
        List of all the connected components in the causal model
        :return: List of sets of variables
        '''
        return list(nx.connected_components(self.exo_graph.to_undirected()))

    @property
    def exo_ccomponents(self):
        '''
        List of all the exogenous variables at each connected component in the causal model
        :return: List of sets of variables
        '''
        return [set(c).intersection(self.exogenous) for c in self.ccomponents]

    @property
    def endo_ccomponents(self):
        '''
        List of all the endogenous variables at each connected component in the causal model
        :return: List of sets of variables
        '''
        return [set(c).intersection(self.endogenous) for c in self.ccomponents]

    def get_ccomponent(self, v:Hashable):
        '''
        C-component to which a variable v belongs
        :param v: variable in the model
        :return: set with the variables in the component
        '''
        return info.get_ccomponent(self,v)

    def get_exo_ccomponent(self, v:Hashable):
        '''
        Exogenous variables in a C-component to which a given variable v belongs
        :param v: variable in the model
        :return: set with the variables in the component
        '''
        return info.get_exo_ccomponent(self,v)

    def get_endo_ccomponent(self, v:Hashable):
        '''
        Endogenous variables in a C-component to which a given variable v belongs
        :param v: variable in the model
        :return: set with the variables in the component
        '''
        return info.get_endo_ccomponent(self, v)

    @property
    def qgraph(self):
        '''
        DAG associated to the d-decomposition of the model
        :return:
        '''
        return info.get_qgraph(self)





class StructuralCausalModel(DiscreteCausalDAGModel):
    ''' Class defining an Structural Causal Model (SCM) over a set of discrete variables.'''

    from bcause.readwrite import scmread, scmwrite
    _reader, _writer = scmread, scmwrite



    def __init__(self, dag:Union[nx.DiGraph,str], factors:Union[dict,list] = None, endogenous:Iterable = None,
                 cast_multinomial:bool = True, check_factors:bool = True):
        self._initialize(dag)
        self._endogenous = endogenous or [x for x in dag if len(list(dag.predecessors(x)))>0]
        self._cast_multinomial = cast_multinomial
        self._check_factors = check_factors

        if factors is not None:
            self._set_factors(factors)

    def builder(self, **kwargs):
        if "cast_multinomial" not in kwargs: kwargs["cast_multinomial"] = self._cast_multinomial
        if "endogenous" not in kwargs: kwargs["endogenous"] = self.endogenous
        return StructuralCausalModel(**kwargs)

    @staticmethod
    def from_model(model:DiscreteDAGModel) -> StructuralCausalModel:
        return StructuralCausalModel(model.graph, model.factors)


    def set_factor(self, var:Hashable, f:bf.DiscreteFactor):
        '''
        Set the factor associated to a variable in the model
        :param var: label of a variable in the model
        :param f: discrete factor to associate to the variable. It is usually of class MultinomialFactor. In case of
        endogenous variables objects of class DeterministicFactor are also accepted.
        :return:
        '''
        # SCM related check
        if self.is_endogenous(var):
            if self._check_factors and not(isinstance(f, DeterministicFactor) or f.is_degenerate()):
                raise ValueError("Factor must be deterministic or degenerate")
            if self._cast_multinomial and isinstance(f, DeterministicFactor):
                f = f.to_multinomial()

        super().set_factor(var, f)

    def to_multinomial(self) -> StructuralCausalModel:
        return StructuralCausalModel(dag=self.graph, factors=self.factors, cast_multinomial=True)

    @property
    def has_deterministic(self):
        return any([isinstance(f, DeterministicFactor) for f in self.get_factors(*self.endogenous)])


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



    def randomize_factors(self, variables, in_place = False, allow_zero = True):
        m = self if in_place else self.copy()
        for x in variables:
            dom = dutils.var_parents_domain(m.domains, m.graph, x)
            f = random_multinomial(dom, [v for v in dom.keys() if x != v], allow_zero=allow_zero)
            m.set_factor(x, f)
        return m

    def fill_random_equations(self, domains):
        for x in self.endogenous:
            dom = dutils.var_parents_domain(domains, self.graph, x)
            f = random_deterministic(dom, [v for v in dom.keys() if x != v])
            self.set_factor(x, f)

    def fill_random_marginals(self, domains):
        for u in self.exogenous:
            dom = dutils.subdomain(domains, u)
            f = random_multinomial(dom)
            self.set_factor(u, f)

    def fill_random_factors(self, domains):
        self.fill_random_equations(domains)
        self.fill_random_marginals(domains)

    def get_qfactors(self, v: Hashable, data=None):
        return info.get_qfactors(self, v, data)

    def get_qfactorisation(self, data=None):
        return dict(itertools.chain(*(self.get_qfactors(c.pop(), data).items() for c in self.ccomponents)))


    def get_qbnet(self, data=None):
        return BayesianNetwork(self.qgraph, self.get_qfactorisation(data))


    def to_bnet(self):
        m = self.to_multinomial() if self.has_deterministic else self
        return BayesianNetwork(m.graph, m.factors)


    def sampleEndogenous(self, n_samples: int, as_pandas = True) -> Union[list[Dict], pd.DataFrame]:
        return self.sample(n_samples, as_pandas)[self.endogenous]

    def log_likelihood(self, data, variables=None):
        bn = self.get_qbnet()
        obs = data.to_dict("records")
        return np.sum(bn.log_prob(obs, variables))


    def max_log_likelihood(self, data, variables=None):
        bn = self.get_qbnet(data)
        obs = data.to_dict("records")
        return np.sum(bn.log_prob(obs,variables))

    def __repr__(self):
        str_card_endo = ",".join([f"{str(v)}:{'' if d is None else str(len(d))}"
                                  for v, d in self._domains.items() if self.is_endogenous(v)])
        str_card_exo = ",".join([f"{str(v)}:{'' if d is None else str(len(d))}"
                                  for v, d in self._domains.items() if self.is_exogenous(v)])
        return f"<StructuralCausalModel ({str_card_endo}|{str_card_exo}), dag={gutils.dag2str(self.graph)}>"


    def draw(self, pos=None):

        G = self.graph
        pos = pos or nx.spring_layout(G)

        nx.draw_networkx_nodes(G, pos=pos, nodelist=self.endogenous, node_color="black", node_size=500)
        nx.draw_networkx_nodes(G, pos=pos, nodelist=self.exogenous, node_color="lightgray", node_size=500)

        endo_labels = {v: v if self.is_endogenous(v) else "" for v in G.nodes}
        nx.draw_networkx_labels(G, pos=pos, labels=endo_labels, font_color="white")

        exo_labels = {v: v if self.is_exogenous(v) else "" for v in G.nodes}
        nx.draw_networkx_labels(G, pos=pos, labels=exo_labels, font_color="black")

        exo_edges = [(x, y) for x, y in G.edges if self.is_exogenous(x)]
        endo_edges = [(x, y) for x, y in G.edges if self.is_endogenous(x)]

        nx.draw_networkx_edges(G, pos=pos, edgelist=exo_edges, edge_color="gray", style="dashed", arrowsize=15)
        nx.draw_networkx_edges(G, pos=pos, edgelist=endo_edges, edge_color="black", style="solid", arrowsize=15)
        plt.box(False)

if __name__ == "__main__":

    import networkx as nx
    import bcause.factors.factor as bf
    import bcause.util.domainutils as dutils
    import bcause.util.graphutils as gutils


    dag = nx.DiGraph([("Y" ,"X"), ("V" ,"Y"), ("U" ,"X")])
    domains = dict(X=["x1", "x2"], Y=[0, 1], U=["u1", "u2", "u3", "u4"], V=[True, False])

    import bcause.factors.factor as bf

    domy = dutils.subdomain(domains, *gutils.relevat_vars(dag, "Y"))
    fy = DeterministicFactor(domy, right_vars = ["V"], values=[1, 0])

    domx = dutils.subdomain(domains, *gutils.relevat_vars(dag, "X"))

    # todo: check data shape for this
    data = ["x1", "x1", "x2", "x1","x1", "x1", "x2", "x1"]
    fx = DeterministicFactor(domx, left_vars = ["X"], values=data)

    domv = dutils.subdomain(domains, "V")
    pv = MultinomialFactor(domv, values= [.1, .9])

    domu = dutils.subdomain(domains, "U")
    pu = MultinomialFactor(domu, values= [.2, .2, .1, .5])

    m = StructuralCausalModel(dag, [fx, fy, pu, pv], cast_multinomial=False)

    print(m.endo_ccomponents)
    print(m.exo_ccomponents)


#


#causal_to_bnet


