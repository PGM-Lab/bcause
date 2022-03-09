from typing import Hashable

from bcause.models.pgmodel import DiscreteDAGModel
from bcause.util.graphutils import dsep_nodes, barren_nodes, remove_nodes


def minimalize(model:DiscreteDAGModel, target:Hashable, evidence:dict = None) -> DiscreteDAGModel:

    evidence = evidence or dict()

    # Determine irrelevant nodes for the query
    dseparated = dsep_nodes(model.graph, target, evidence.keys())
    barren = barren_nodes(model.graph, [target] + list(evidence.keys()))
    irrelevant = dseparated | barren

    # Remove irrelevant nodes from DAG
    new_dag = remove_nodes(model.graph, irrelevant)

    # Remove factors with dseparated nodes on the left
    dsep_conf = {v: model.domains[v][0] for v in dseparated}

    # Restrict to arbitrary values variables on the right that are dsep.
    new_factors = {v: f.restrict(**dsep_conf) for v, f in model.factors.items() if v not in irrelevant}

    return model.builder(new_dag, new_factors)
