import pyAgrum as gum
import networkx as nx
import numpy as np
import bcause.factors as bfd

from pgmpy.models import BayesianNetwork
from itertools import starmap
from bcause.models import BayesianNetwork
from bcause.models.cmodel import StructuralCausalModel
from bcause.util import randomUtil



# Define a DAG and the domains
dag = nx.DiGraph([("V1", "V2"), ("V2", "V3"),("V3", "V4"),("U1", "V1"),("U2", "V2"),("U2", "V4"),("U3", "V3")])
model = StructuralCausalModel(dag)
domains = dict(V1=[0,1],V2=[0,1],V3=[0,1],V4=[0,1], U1=[0,1],U2=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],U3=[0,1,2,3])
randomUtil.seed(1)
model.fill_random_factors(domains)
data = model.sample(1000, as_pandas=True)

bn = BayesianNetwork(dag,model.factor_list)

def addGraph(bn: gum.BayesNet, nodes: list[str], card: dict ,  edges: list[tuple[str, str]]):
    # add Nodes
    list(map(lambda node: bn.add(node, card[node]), nodes))
    # add Edges
    list(starmap(bn.addArc, edges))

# Add CPTs
def setCPT(f: bfd.MultinomialFactor, bn: gum.BayesNet):
    v = list(f.left_domain.keys())
    rv = list(f.right_domain.keys())
    ndim = len(f.variables)
    # Check nodes
    [bn.add(i) for i in v+rv if not bn.exists(i)]
    # Check edges
    [bn.addArc(i, *v) for i in rv if not bn.existsArc(i, *v)]

    # Set the CPT
    values = f.values_array()
    shape = [len(f.domain[var]) for var in bn.cpt(*v).names[::-1]]
    values = values.reshape(*shape)
    bn.cpt(f.left_vars[0])[:] = values

def toAgrum(bn: BayesianNetwork) -> gum.BayesNet:
    gumbn = gum.BayesNet()
    card = {v: len(d) for v,d in bn.domains.items()}

    addGraph(gumbn, bn.variables, card,list(bn.graph.edges))
    [setCPT(f,gumbn) for f in bn.factors.values()]
    return gumbn

gumbn = toAgrum(bn)