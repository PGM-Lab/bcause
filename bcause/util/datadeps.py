


import networkx as nx
import numpy as np
import pandas as pd

from functools import cache

import bcause as bc
from bcause.models.cmodel import StructuralCausalModel
from bcause.util.graphutils import *

class DataDepAnalysis(object):

    def __init__(self, dag, data):
        self._data = data
        self._dag = dag
        self._data_unique = data.drop_duplicates().to_dict("records")

    def expectation_relevant_vars(self,v):
        return relevat_vars(self._dag,v)

    def posterior_relevant_vars(self, v, evidence_nodes):
        return list(dcon_nodes(self._dag, v, evidence_nodes))



    @cache
    def get_minimal_obs_blanket(self, target):
        def dsep_filter(d):
            evidence_vars = [v for v in d.keys() if not pd.isna(d[v])]
            connected_vars = dcon_nodes(self._dag, target, evidence_vars) | {target}
            return {k: v for k, v in d.items() if k in connected_vars and k in evidence_vars}

        return self.__obs_unique([dsep_filter(d) for d in self._data_unique])

    @staticmethod
    def __obs_unique(obs_list):
        obs_eq = lambda d1, d2: d1.keys() == d2.keys() and all([d1[k] == d2[k] for k in d1.keys()])

        idx = list(range(len(obs_list)))

        for i in range(len(obs_list) - 1, 0, -1):
            for j in range(i - 1, -1, -1):
                if obs_eq(obs_list[i], obs_list[j]):
                    del idx[i]
                    break
        return [obs_list[i] for i in idx]



if __name__ == "__main__":


    # Define a DAG and the domains
    dag = nx.DiGraph([("V1", "V2"), ("V2", "V3"),("V3", "V4"),("U1", "V1"),("U2", "V2"),("U2", "V4"),("U3", "V3")])
    model = StructuralCausalModel(dag)
    domains = dict(V1=[0,1],V2=[0,1],V3=[0,1],V4=[0,1], U1=[0,1,3],U2=[0,1,2,3],U3=[0,1,2,3])
    bc.randomUtil.seed(1)
    model.fill_random_factors(domains)
    data = model.sample(1000, as_pandas=True)[["V1", "V2", "V3", "V4"]]


    dag = nx.DiGraph([("U","X"),("U","Y"),("X","Y")])
    model = StructuralCausalModel(dag)
    domains = dict(U = [0,1,2,3], X=[0,1], Y=[0,1])
    bc.randomUtil.seed(1)
    model.fill_random_factors(domains)
    data = model.sample(20, as_pandas=True)[["X", "Y"]]

    data["S"] = data.X == 0

    data.loc[-data.S, "X"] = np.nan
    data.loc[-data.S, "Y"] = np.nan
    dag.add_edge("X","S")
    dag.add_edge("Y","S")

    dag.add_edge("H","U")


    data = data.append(dict(U=3), ignore_index=True)
    data = data.append(dict(U=2, H="h1"), ignore_index=True)
    data = data.append(dict(U=np.nan, H="h2"), ignore_index=True)


    print(data)


    deps = DataDepAnalysis(dag, data)

    print(deps.get_minimal_obs_blanket("U"))
    print(deps.get_minimal_obs_blanket("U"))

    target = "U"

    obs_blanket = deps.get_minimal_obs_blanket(target)
    [(obs, len(data.loc[data[obs.keys()].isin(obs.values()).all(axis=1), :])) for obs in obs_blanket]



    len(data)