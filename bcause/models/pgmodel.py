from abc import ABC, abstractmethod
from typing import Union, Dict, Hashable

import networkx as nx

from factors.factor import MultinomialFactor, DiscreteFactor, ConditionalFactor


class PGModel(ABC):
    @property
    def network(self) -> nx.Graph:
        return self._network

    @property
    def nodes(self) -> nx.Graph:
        return self._network.nodes

    @property
    def factors(self) -> Union[Dict,list]:
        return self._factors

    #@abstractmethod
    #def get_factors(self, *variables) -> list:
    #    pass



class DiscreteDAGModel(PGModel):
    def get_domain(self, *variables):
        pass

    def get_children(self, *variables) -> list:
        return list(set(sum([list(dag.successors(v)) for v in variables], [])))

    def get_parents(self, *variables) -> list:
        return list(set(sum([list(dag.predecessors(v)) for v in variables], [])))

    def markov_blanket(self, v)-> list:
        ch = set(self.get_children(v))
        pa = set(self.get_parents(v))
        pa_ch = set(self.get_parents(*ch).difference({v}))

        # return the union
        return list(ch | pa | pa_ch)


    # def set_factor(self, var:Hashable, f:DiscreteFactor):
    #     # check type
    #     if not isinstance(f, MultinomialFactor): raise ValueError("Factor must  be discrete")
    #     if not isinstance(f, ConditionalFactor): raise ValueError("Factor must  be conditional")
    #
    #     # check left right variables
    #     if set(self.get_parents(var)) != set(f.right_vars): raise ValueError("Wrong right variables in factor")
    #     if set(var) != set(f.left_vars): raise ValueError("Wrong left variables in factor")
    #
    #     # check domains
    #     for v in f.variables:
    #         if self._domain[v]
    #
    #
    #     # Update domains
    #
    #     # update factor dictionary
    #
    #
    #     pass


class BayesianNetwork(DiscreteDAGModel):
    def __init__(self, dag:nx.DiGraph, factors:dict = None):

        if not isinstance(dag, nx.DiGraph): raise ValueError("Input graph must be a DAG")

        self._network = dag
        self._factors = {x:None for x in dag.nodes}
        self._domains = {x:None for x in dag.nodes}



if __name__ == "__main__":

    dag = nx.DiGraph([(0,1),(1,2), (2,3), (4,2)])
    dag.successors([1,2])




