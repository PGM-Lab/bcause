



import networkx as nx

import bcause as bc
from bcause.models.cmodel import StructuralCausalModel
from bcause.util.graphutils import *


# Define a DAG and the domains
dag = nx.DiGraph([("V1", "V2"), ("V2", "V3"),("V3", "V4"),("U1", "V1"),("U2", "V2"),("U2", "V4"),("U3", "V3")])
model = StructuralCausalModel(dag)
domains = dict(V1=[0,1],V2=[0,1],V3=[0,1],V4=[0,1], U1=[0,1,3],U2=[0,1,2,3],U3=[0,1,2,3])
bc.random.seed(1)
model.fill_random_factors(domains)
data = model.sample(1000, as_pandas=True)[["V1","V2","V3","V4"]]


dag = nx.DiGraph([("U","X"),("U","Y"),("X","Y")])
model = StructuralCausalModel(dag)
domains = dict(U = [0,1,2,3], X=[0,1], Y=[0,1])
bc.random.seed(1)
model.fill_random_factors(domains)
data = model.sample(1000, as_pandas=True)[["X","Y"]]

data["S"] = data.Y == 0

data.loc[-data.S, "X"] = np.nan
data.loc[-data.S, "Y"] = np.nan
dag.add_edge("X","S")
dag.add_edge("Y","S")


model.draw()


class c():
    pass


self = c()


dag.edges
self._dag = dag
self._data = data

import matplotlib.pyplot as plt


# expectation_vars

v = "V2"
def expectation_relevant_vars(self,v):
    return relevat_vars(self._dag,v)

def posterior_relevant_vars(self, v, evidence_nodes):
    return list(dcon_nodes(self._dag, v, evidence_nodes))





import math
import numpy as np

v = "U"

_, vals = list(unique.iterrows())[1]
vals = dict(vals)

self._dag.edges

def get_minimal_obs_blanket(v, fixed_obs=None):
    fixed_obs = fixed_obs or dict()

    hidden = [x for x,val in fixed_obs.items() if np.isnan(val)]
    blanket = list(markov_blanket(self._dag, v, hidden))
    unique = self._data[blanket].drop_duplicates()
    out = list()

    # set fixed obs
    for x,val in fixed_obs.items():
        if x in unique.columns:
            unique[x] = val
            unique = unique.drop_duplicates()


    # for each possible value in the markov blanket
    for _, vals in unique.iterrows():

        new_hidden = [x for x, val in vals.items() if np.isnan(val)]

        if set(hidden) != set(new_hidden):
            out += get_minimal_obs_blanket(v, fixed_obs=dict(vals))
        else:
            out.append(dict(vals))


    return out


get_minimal_obs_blanket("U")


fixed_obs = dict(V4=np.nan)

[x for x,val in fixed_obs.items() if np.isnan(val)]

get_minimal_obs_blanket("U2", dict(V4=np.nan))
# if it has no missing, compute the relevant and add it
#if not np.isnan(vals.values).any():



# if it has missing fix those in the mb and then consider all unique of the rest




# markov_blanket with hidden:

# observe all the rest






# get_data_expectation

# get_counts_expectation

#get_data_posterior

#get_counts_posterior


# [x for x in self._dag.nodes if x not in  and v != x]


markov_blanket(dag,v)

hidden = set(["V1"])

observed = set(dag.nodes).difference(set(hidden) | {v})
dcon_nodes(dag, v, observed)
