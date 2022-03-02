from abc import ABC, abstractmethod
from typing import Union, Dict, Hashable

import networkx as nx

from factors.factor import DiscreteFactor, ConditionalFactor, Factor


class PGModel(ABC):

    def _initialize(self, network:nx.Graph):
        self._network = network
        self._factors = {x: None for x in network.nodes}
        self._domains = {x: None for x in network.nodes}

    @property
    def network(self) -> nx.Graph:
        return self._network

    @property
    def variables(self) -> list:
        return list(self._network.nodes)

    @property
    def factors(self) -> Union[Dict,list]:
        return self._factors

    def get_factors(self, *variables) -> list:
        return [self._factors[v] for v in variables]

    @abstractmethod
    def set_factor(self, var:Hashable, f:Factor):
        pass



class DiscreteDAGModel(PGModel):


    def get_domains(self, variables):
        return {v:d for v,d in self._domains.items() if v in variables}

    def get_varsizes(self, variables):
        return {v:len(d) for v,d in self._domains.items() if v in variables}


    @property
    def domains(self):
        return self.get_domains(self.variables)

    @property
    def varsizes(self):
        return self.get_varsizes(self.variables)

    def get_children(self, *variables) -> list:
        return list(set(sum([list(self.network.successors(v)) for v in variables], [])))

    def get_parents(self, *variables) -> list:
        return list(set(sum([list(self.network.predecessors(v)) for v in variables], [])))

    def markov_blanket(self, v)-> list:
        ch = set(self.get_children(v))
        pa = set(self.get_parents(v))
        pa_ch = set(self.get_parents(*ch)).difference({v})

        # return the union
        return list(ch | pa | pa_ch)

    def get_dag_str(self):
        str_dag = ""
        for v in self.network.nodes:
            parents = list(self.network.predecessors(v))
            str_dag += f"[{v}"
            if len(parents) > 0:
                str_dag += "|" + ":".join(parents)
            str_dag += "]"

        return str_dag





