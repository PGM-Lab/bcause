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


    # Determine irrelevant nodes for the query
    barren = barren_nodes(model.graph, sum(as_lists(target, evidence.keys()),[]))
    disconnected = disconnected_nodes(remove_nodes(model.graph, barren), target)
    irrelevant =  (barren | disconnected) - target

    # Remove irrelevant nodes from DAG
    new_dag = remove_nodes(model.graph, irrelevant)
    new_factors = {v:f  for v, f in model.factors.items() if v not in irrelevant}

    logging.debug(f"Minimalized DAG: {new_dag.edges}")
    return model.builder(dag=new_dag, factors=new_factors, check_factors=False)

