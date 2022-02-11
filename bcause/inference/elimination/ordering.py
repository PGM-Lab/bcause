import numpy as np
import networkx as nx
from networkx import moral_graph, Graph


from itertools import product, combinations

def fillin_arcs(var, moral):
    return [(x,y) for x,y in combinations(list(moral[var]), r=2) if y not in moral.adj[x]]


# costs
cost_functions = dict(
    min_size = lambda var, moral : len(moral.adj[var]),
    min_weight = lambda var, moral : np.prod([moral.nodes[v]["size"] for v in list(moral[var])+[var]]),
    min_fill = lambda var, moral : len(fillin_arcs(var, moral))
)



def _get_elim_ordering(dag, varsizes=None, to_remove=None, heuristic="min_size", random_state = 0):

    to_remove = to_remove or list(dag.nodes)
    cost_fn = cost_functions[heuristic]
    choice = np.random.RandomState(random_state).choice

    # initialize moral graph
    moral = moral_graph(dag)
    if heuristic == "min_weight":
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


def min_fill_heuristic(dag:nx.DiGraph, to_remove=None, random_state = 0) -> list:
    return _get_elim_ordering(dag, varsizes=None, to_remove=to_remove, heuristic="min_fill", random_state=0)

def min_size_heuristic(dag:nx.DiGraph, to_remove=None, random_state = 0) -> list:
    return _get_elim_ordering(dag, varsizes=None, to_remove=to_remove, heuristic="min_size", random_state=0)

def min_weight_heuristic(dag:nx.DiGraph, varsizes, to_remove=None, random_state = 0) -> list:
    return _get_elim_ordering(dag, varsizes, to_remove=to_remove, heuristic="min_weight", random_state=0)



if __name__=="__main__":

    dag = nx.DiGraph([("A", "B"), ("C", "B"), ("B", "D")])
    varsizes = dict(A=5, B=3, C=2, D=2)

    min_size_heuristic(dag)
    min_fill_heuristic(dag)
    min_weight_heuristic(dag, varsizes)