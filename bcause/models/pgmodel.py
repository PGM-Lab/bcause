from abc import ABC, abstractmethod
from typing import Union, Dict, Hashable

import networkx as nx

from factors.factor import MultinomialFactor, DiscreteFactor, ConditionalFactor, Factor


class PGModel(ABC):

    def _initialize(self, network:nx.Graph):
        self._network = network
        self._factors = {x: None for x in network.nodes}
        self._domains = {x: None for x in network.nodes}

    @property
    def network(self) -> nx.Graph:
        return self._network

    @property
    def nodes(self) -> nx.Graph:
        return self._network.nodes

    @property
    def factors(self) -> Union[Dict,list]:
        return self._factors

    def get_factors(self, *variables) -> list:
        return [self._factors[v] for v in variables]

    @abstractmethod
    def set_factor(self, var:Hashable, f:Factor):
        pass



class DiscreteDAGModel(PGModel):
    def get_domains(self, *variables):
        pass

    def get_children(self, *variables) -> list:
        return list(set(sum([list(dag.successors(v)) for v in variables], [])))

    def get_parents(self, *variables) -> list:
        return list(set(sum([list(dag.predecessors(v)) for v in variables], [])))

    def markov_blanket(self, v)-> list:
        ch = set(self.get_children(v))
        pa = set(self.get_parents(v))
        pa_ch = set(self.get_parents(*ch)).difference({v})

        # return the union
        return list(ch | pa | pa_ch)




class BayesianNetwork(DiscreteDAGModel):
    def __init__(self, dag:nx.DiGraph, factors:dict = None):

        if not isinstance(dag, nx.DiGraph): raise ValueError("Input graph must be a DAG")
        self._initialize(dag)

        for v,f in factors.items():
            self.set_factor(v,f)


    def set_factor(self, var:Hashable, f:DiscreteFactor):
        # check type
        if not isinstance(f, MultinomialFactor): raise ValueError("Factor must  be discrete")
        if not isinstance(f, ConditionalFactor): raise ValueError("Factor must  be conditional")

        # check left right variables
        if set(self.get_parents(var)) != set(f.right_vars): raise ValueError("Wrong right variables in factor")
        if set(var) != set(f.left_vars): raise ValueError("Wrong left variables in factor")

        # check domains
        for v in f.variables:
            if v in self._domains and self._domains[v] != None:
                    if set(self._domains[v]) != set(f.domain[v]): raise ValueError(f"Inconsistent domain for {v}")

        # Update domains
        if var not in self._domains:
            self._domains[var] = f.domain[var]

        # update factor dictionary
        self._factors[var] = f

if __name__ == "__main__":

    dag = nx.DiGraph([("A","B")])

    domain = dict(A=["a1", "a2"], B=[0, 1, 3])

    # testing factors
    fb = MultinomialFactor(domain, data=[[0.2, 0.1, 0.7], [0.3, 0.6, 0.1]], right_vars=["A"])
    fa = MultinomialFactor(dict(A=["a1", "a2"]), data=[0.2, 0.8])

    bnet = BayesianNetwork(dag, dict(B=fb, A=fa))
    bnet.factors

    bnet.markov_blanket("B")


