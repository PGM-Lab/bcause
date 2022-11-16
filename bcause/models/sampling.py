from __future__ import annotations

import pandas as pd
import networkx as nx

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bcause.models.pgmodel import DiscreteDAGModel

def forward_sampling(model:DiscreteDAGModel, n_samples:int) -> pd.DataFrame:
    data = pd.DataFrame()
    for v in nx.topological_sort(model.graph):
        if len(model.get_parents(v)) == 0 : obs_v = model.factors[v].sample(n_samples)
        else: obs_v = model.factors[v].sample_conditional(data.to_dict("records"))
        data = pd.concat([data, pd.DataFrame(obs_v)], axis=1)
    return data


