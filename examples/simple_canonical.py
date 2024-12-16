import networkx as nx

from bcause.factors.mulitnomial import canonical_for_model, uniform_multinomial
from bcause.models.cmodel import StructuralCausalModel

endo_dag = nx.DiGraph([("X", "Y")])
endo_dom = dict(X=["x1", "x2"], Y=["y1","y2"])



# Define a DAG and the domains for the endogenous variables
dag = nx.DiGraph([("X", "Y"), ("U", "Y"), ("V", "X")])
endo_dom = dict(X=["x1", "x2"], Y=["y1","y2"])

# Define an empty model only with the graph
model = StructuralCausalModel(dag)

# Structural equations
fx = canonical_for_model(model, endo_dom, "X")
fy = canonical_for_model(model, endo_dom, "Y")

model.set_factor("X",fx)
model.set_factor("Y",fy)


# Set the initial exogenous distributions
domv = dict(V=fx.domain["V"])
pv = uniform_multinomial(domv)
model.set_factor("V", pv)

domu = dict(U=fy.domain["Y"])
pu = uniform_multinomial(domu)
model.set_factor("U", pu)


