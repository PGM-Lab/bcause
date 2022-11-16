import networkx as nx

import logging

from bcause.util.arrayutils import len_iterable, as_lists


def dag2str(dag: nx.DiGraph) -> str:
    str_dag = ""
    for v in dag.nodes:
        parents = list(dag.predecessors(v))
        str_dag += f"[{v}"
        if len(parents) > 0:
            str_dag += "|"
            str_dag += ":".join([str(p) for p in parents])
        str_dag += "]"
    return str_dag


def str2dag(str_dag: str) -> nx.DiGraph:
    arcs = []
    s = str_dag[1:-1].split("][")[1]
    for s in str_dag[1:-1].split("]["):
        if "|" in s:
            x, right_vars = s.split("|")
            for y in right_vars.split(":"):
                arcs.append((y, x))

    return nx.DiGraph(arcs)


def _barren_info(dag: nx.DiGraph, S=None):
    S = S or set()
    barren = set()
    while True:
        new_barren = set([x for x in dag.nodes if len_iterable(dag.successors(x)) == 0 and x not in S])
        if len(new_barren) > 0:
            barren = barren | new_barren
            dag = dag.subgraph([x for x in dag.nodes if x not in barren])
        else:
            break

    logging.debug(f"Barren nodes wrt {S} are: {barren}")
    return dag, barren


def barren_removal(dag: nx.DiGraph, S=None) -> nx.DiGraph:
    return _barren_info(dag, S)[0]


def barren_nodes(dag: nx.DiGraph, S=None) -> nx.DiGraph:
    return _barren_info(dag, S)[1]


def dsep_nodes(dag: nx.DiGraph, target, evidence_nodes):
    target, evidence_nodes = as_lists(target, evidence_nodes)

    dsep = set(
        [v for v in dag.nodes if
         nx.d_separated(dag, x=set(target), y={v}, z=set(evidence_nodes)) and v not in evidence_nodes])
    logging.debug(f"D-separated nodes wrt {target} given {evidence_nodes} are: {dsep}")
    return dsep


def remove_outgoing_edges(dag: nx.DiGraph, parent_vars: list) -> nx.DiGraph:
    involved_edges = [(x, y) for (x, y) in dag.edges if x in parent_vars]
    out = dag.copy()
    out.remove_edges_from(involved_edges)
    return out


def remove_ingoing_edges(dag: nx.DiGraph, children_vars: list) -> nx.DiGraph:
    involved_edges = [(x, y) for (x, y) in dag.edges if y in children_vars]
    out = dag.copy()
    out.remove_edges_from(involved_edges)
    return out


def remove_nodes(dag: nx.DiGraph, nodes: list) -> nx.DiGraph:
    out = dag.copy()
    out.remove_nodes_from(nodes)
    return out


# relevant vars in DAG
def relevat_vars(dag: nx.DiGraph, v):
    return list(dag.predecessors(v)) + [v]

def connected(G:nx.Graph, target, y):
    if y in as_lists(target): return True
    G = G.to_undirected()
    return any([nx.node_connectivity(G, t, y) > 0 for t in as_lists(target)])

def disconnected_nodes(G:nx.Graph, target):
    return set([x for x in G.nodes if not connected(G, target, x)])


