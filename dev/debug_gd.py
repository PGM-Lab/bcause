import logging
import sys

import networkx as nx

import bcause.util.domainutils as dutils
from bcause.factors import DeterministicFactor
from bcause.factors.mulitnomial import MultinomialFactor
from bcause.inference.causal.multi import GDCC, EMCC
from bcause.inference.probabilistic.elimination import VariableElimination
from bcause.inference.causal.elimination import CausalVariableElimination
from bcause.models.cmodel import StructuralCausalModel

#log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'
#logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format, datefmt='%Y%m%d_%H%M%S')

# Define a DAG and the domains
dag = nx.DiGraph([("X", "Y"), ("U", "Y"), ("V", "X")])
domains = dict(X=["x1", "x2"], Y=["y1","y2"], U=["u1", "u2", "u3", "u4"], V=["v1", "v2"])


domx = dutils.var_parents_domain(domains,dag,"X")
fx = DeterministicFactor(domx, right_vars=["V"], values=["x1", "x2"])

domy = dutils.var_parents_domain(domains,dag,"Y")
values = [["y1", "y1"], ["y2", "y1"], ["y1", "y1"], ["y2", "y1"]]
values = ["y1", "y1", "y2", "y1", "y2", "y2", "y1", "y1"]
fy = DeterministicFactor(domy, left_vars=["Y"], values=values)
fy.store.data
fy.to_multinomial().restrict(X="x2", Y="y1")
domv = dutils.subdomain(domains, "V")
pv = MultinomialFactor(domv, values=[.5, .5])
domu = dutils.subdomain(domains, "U")
pu = MultinomialFactor(domu, values=[.2, .2, .6, .0])

model = StructuralCausalModel(dag, [fx, fy, pu, pv], cast_multinomial=True)



####

data = model.sampleEndogenous(1000)

inf = GDCC(model, data, num_runs=50, tol=0.00000001)
inf = EMCC(model, data, max_iter=100, num_runs=20)

lmax = model.max_log_likelihood(data)
inf.compile()

#p = inf.causal_query("Y", do=dict(X="x1"))
#print(p)


for m in inf.models:
    ll = m.log_likelihood(data)
    print(ll)



for m in inf.models:
    print(m.factors["U"])

for m in inf.models:
    print(m.get_qbnet().factors)

m.get_qbnet(data).factors

model.factors["U"]


sum(data.X=="x1")
