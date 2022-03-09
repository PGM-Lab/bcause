from typing import Hashable, Union

import networkx as nx
from networkx.algorithms import moral


import bcause.factors.factor as bf
from factors.mulitnomial import random_multinomial
from models.pgmodel import DiscreteDAGModel
from util.graphutils import str2dag, dag2str


class BayesianNetwork(DiscreteDAGModel):
    def __init__(self, dag:Union[nx.DiGraph,str], factors:dict = None):

        if isinstance(dag, str): dag = str2dag(dag)
        if not isinstance(dag, nx.DiGraph) or len(list(nx.simple_cycles(dag)))>0:
            raise ValueError("Input graph must be a DAG")
        self._initialize(dag)

        if factors is not None:
            for v,f in factors.items():
                self.set_factor(v,f)

        def builder(*args, **kwargs): return BayesianNetwork(*args, **kwargs)
        self.builder = builder


    def set_factor(self, var:Hashable, f:bf.DiscreteFactor):
        # check type
        if not isinstance(f, bf.DiscreteFactor):
            raise ValueError("Factor must  be discrete")
        if not isinstance(f, bf.ConditionalFactor): raise ValueError("Factor must  be conditional")

        # check left right variables
        if set(self.get_parents(var)) != set(f.right_vars): raise ValueError("Wrong right variables in factor")
        if set([var]) != set(f.left_vars):
            raise ValueError("Wrong left variables in factor")

        # check domains
        for v in f.variables:
            if v in self._domains and self._domains[v] != None:
                    if set(self._domains[v]) != set(f.domain[v]): raise ValueError(f"Inconsistent domain for {v}")

        # Update domains
        if var not in self._domains or self._domains[var] == None:
            self._domains[var] = f.domain[var]

        # update factor dictionary
        self._factors[var] = f

    def __repr__(self):
        str_card = ",".join([f"{str(v)}:{'' if d is None else str(len(d))}" for v, d in self._domains.items()])

        return f"<BayesianNetwork ({str_card}), dag={dag2str(self.graph)}>"

    def randomize_factors(self, domains:dict):
        for v in self.variables:
            parents = self.get_parents(v)
            dom = {x: d for x, d in domains.items() if x == v or x in parents}
            self.set_factor(v, random_multinomial(dom, right_vars=parents))




if __name__ == "__main__":

    dag = nx.DiGraph([("A","B"), ("C","B"), ("C","D")])
    domains = dict(A=["a1", "a2"], B=[0, 1, 3], C=["c1", "c2", "c3", "c4"], D=[True, False])
    factors = dict()

    for v in dag.nodes:
        parents = list(dag.predecessors(v))
        dom = {x:d for x,d in domains.items() if x == v or x in parents}
        factors[v]  = random_multinomial(dom, right_vars=parents)

    bnet = BayesianNetwork(dag, factors)

    bnet = BayesianNetwork("[A][B|A:C][C][D|C]", factors)

    bnet = BayesianNetwork("[A][B|A:C][C][D|C]")
    bnet.randomize_factors(domains)


    print(bnet.get_domains(bnet.variables))

