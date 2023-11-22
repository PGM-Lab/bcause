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
dag = nx.DiGraph([("V", "X")])
domains = dict(X=["x1", "x2"], V=["v1", "v2"])


domx = dutils.var_parents_domain(domains,dag,"X")
fx = DeterministicFactor(domx, right_vars=["V"], values=["x1", "x2"])
domv = dutils.subdomain(domains, "V")
pv = MultinomialFactor(domv, values=[.5, .5])
model = StructuralCausalModel(dag, [fx, pv], cast_multinomial=True)



####

data = model.sampleEndogenous(1000)

inf = GDCC(model, data, num_runs=50, tol=0.00000001)
#inf = EMCC(model, data, max_iter=100, num_runs=20)

lmax = model.max_log_likelihood(data)

inf.compile()



for m in inf.models:
    ll = m.log_likelihood(data)
    print(ll)



for m in inf.models:
    print(m.factors["V"])

for m in inf.models:
    print(m.get_qbnet().factors)

m.get_qbnet(data).factors

sum(data.X=="x1")
