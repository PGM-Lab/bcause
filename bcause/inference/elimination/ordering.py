from enum import Enum, auto

import numpy as np
import networkx as nx
from networkx import moral_graph, Graph


from itertools import product, combinations

class Heuristic(Enum):
    MIN_FILL = auto()
    MIN_SIZE = auto()
    MIN_WEIGHT = auto()


def fillin_arcs(var, moral):
    return [(x,y) for x,y in combinations(list(moral[var]), r=2) if y not in moral.adj[x]]


# costs
cost_functions = {
    Heuristic.MIN_SIZE.name : lambda var, moral : len(moral.adj[var]),
    Heuristic.MIN_WEIGHT.name : lambda var, moral : np.prod([moral.nodes[v]["size"] for v in list(moral[var])+[var]]),
    Heuristic.MIN_FILL.name : lambda var, moral : len(fillin_arcs(var, moral))
}


def _get_elim_ordering(dag, varsizes=None, to_remove=None, heuristic=Heuristic.MIN_SIZE, random_state = 0):

    cost_fn = cost_functions[heuristic.name]
    choice = np.random.RandomState(random_state).choice

    # initialize moral graph
    moral = moral_graph(dag)
    if heuristic == Heuristic.MIN_WEIGHT:
        assert varsizes is not None
        for v in dag.nodes:
            moral.nodes[v]["size"] = varsizes[v]

    ordering = []

    while len(to_remove)>0:
        # compute costs
        costs = {v:cost_fn(v,moral) for v in to_remove}
        min_cost = min(costs.values())
        select_var = choice([v for v,c in costs.items() if c == min_cost])

        # remove node
        moral.edges
        for x,y in fillin_arcs(select_var, moral):
            moral.add_edge(x,y)
        moral.remove_node(select_var)
        to_remove.remove(select_var)

        # update ordering
        ordering.append(select_var)

    return ordering

def min_fill_heuristic(dag:nx.DiGraph, to_remove, random_state = 0, **kwargs) -> list:
    return _get_elim_ordering(dag, varsizes=None, to_remove=to_remove, heuristic=Heuristic.MIN_FILL, random_state=0)

def min_size_heuristic(dag:nx.DiGraph, to_remove, random_state = 0, **kwargs) -> list:
    return _get_elim_ordering(dag, varsizes=None, to_remove=to_remove, heuristic=Heuristic.MIN_SIZE, random_state=0)

def min_weight_heuristic(dag:nx.DiGraph, to_remove, random_state = 0, varsizes = None) -> list:
    if varsizes is None:
        raise ValueError("Size of Variables must be provided")
    return _get_elim_ordering(dag, varsizes, to_remove=to_remove, heuristic=Heuristic.MIN_WEIGHT, random_state=0)


heuristic_functions = {
    Heuristic.MIN_SIZE.name: min_size_heuristic,
    Heuristic.MIN_WEIGHT.name: min_weight_heuristic,
    Heuristic.MIN_FILL.name: min_fill_heuristic
}


if __name__=="__main__":

    dag = nx.DiGraph([("A", "B"), ("C", "B"), ("B", "D")])
    varsizes = dict(A=5, B=3, C=2, D=2)

    min_size_heuristic(dag)
    min_fill_heuristic(dag)
    min_weight_heuristic(dag, varsizes=varsizes)