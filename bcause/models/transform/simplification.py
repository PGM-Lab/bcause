import logging
from typing import Hashable

import networkx as nx

from bcause.models.pgmodel import DiscreteDAGModel
from bcause.util.arrayutils import as_lists
from bcause.util.graphutils import dsep_nodes, barren_nodes, remove_nodes, disconnected_nodes, remove_outgoing_edges, \
    remove_ingoing_edges


def minimalize(model:DiscreteDAGModel, target:Hashable, evidence:dict = None, restrict_dsep:bool = False) -> DiscreteDAGModel:

    evidence = evidence or dict()
    target = {target} if type(target) not in [list, set] else set(target)
    graph = model.graph

    # Determine irrelevant nodes for the query
    barren = barren_nodes(graph, sum(as_lists(target, evidence.keys()),[]))

    graph = remove_outgoing_edges(graph, evidence.keys())

    disconnected = disconnected_nodes(remove_nodes(graph, barren), target)
    irrelevant =  (barren | disconnected) - target


    # Remove irrelevant nodes from DAG
    new_dag = remove_nodes(graph, irrelevant)
    new_factors = {v:f  for v, f in model.factors.items() if v not in irrelevant}
    #new_factors = dict()

    # Set observations
    for v,f in new_factors.items():
        obs_pa = {v:evidence[v] for v in set(f.right_vars).intersection(evidence.keys())}
        new_factors[v] = f.R(**obs_pa)

    logging.debug(f"Minimalized DAG: {new_dag.edges}")
    return model.builder(dag=new_dag, factors=new_factors, check_factors=False)

