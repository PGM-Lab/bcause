from __future__ import annotations

import networkx as nx
import itertools

from typing import Hashable
from networkx import topological_sort

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from bcause.models.cmodel import DiscreteCausalDAGModel, StructuralCausalModel




def get_ccomponent(model:DiscreteCausalDAGModel, v: Hashable):
    return nx.node_connected_component(model.exo_graph.to_undirected(), v)


def get_exo_ccomponent(model:DiscreteCausalDAGModel, v: Hashable):
    return model.get_ccomponent(v).intersection(model.exogenous)


def get_endo_ccomponent(model:DiscreteCausalDAGModel, v: Hashable):
    return model.get_ccomponent(v).intersection(model.endogenous)


def get_qgraph(model:DiscreteCausalDAGModel):
    g = model.endo_graph.copy()
    order = list(topological_sort(g))


    for c_comp in model.endo_ccomponents:
        c_comp = [v for v in order if v in c_comp]
        pa_c = set(itertools.chain(*[model.get_edogenous_parents(v) for v in c_comp]))
        pa_c = pa_c.difference(c_comp)
        for i in range(len(c_comp)):
            x = c_comp[i]
            for p in list(pa_c) + c_comp[0:i]:
                if p not in nx.descendants(model.endo_graph, x):
                    g.add_edge(p, x)

    return g

def get_qfactors(model:StructuralCausalModel, v:Hashable, data=None):
    if data is None:
        import bcause.inference.probabilistic.elimination as ve
        inf = ve.VariableElimination(model)
    else:
        import bcause.inference.probabilistic.datainference as di
        inf = di.LaplaceInference(data, model.domains)

    ccomp = model.get_endo_ccomponent(v)
    return {v:inf.query(v,conditioning=list(model.qgraph.predecessors(v))) for v in ccomp}
