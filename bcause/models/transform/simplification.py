import logging
from typing import Hashable

import networkx as nx

from bcause.models.pgmodel import DiscreteDAGModel
from bcause.util.arrayutils import as_lists
from bcause.util.graphutils import dsep_nodes, barren_nodes, remove_nodes, disconnected_nodes


def minimalize(model:DiscreteDAGModel, target:Hashable, evidence:dict = None, restrict_dsep:bool = False) -> DiscreteDAGModel:

    evidence = evidence or dict()

    # Determine irrelevant nodes for the query
    dseparated = set([x for x in dsep_nodes(model.graph, target, evidence.keys()) if x not in evidence.keys()])
    barren = barren_nodes(model.graph, sum(as_lists(target, evidence.keys()),[]))
    disconnected = disconnected_nodes(remove_nodes(model.graph, dseparated | barren), target)
    irrelevant = dseparated | barren | disconnected
    #irrelevant =  barren | disconnected

    # Remove irrelevant nodes from DAG
    new_dag = remove_nodes(model.graph, irrelevant)

    # Remove factors with dseparated nodes on the left
    dsep_conf = {v: model.domains[v][0] for v in dseparated}

    new_factors = {v: f.restrict(**dsep_conf) if restrict_dsep else f for v, f in model.factors.items() if v not in irrelevant}

    logging.debug(f"Minimalized DAG: {new_dag.edges}")
    return model.builder(dag=new_dag, factors=new_factors, check_factors=False)
