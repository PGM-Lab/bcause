import logging
import sys

import networkx as nx

import bcause.util.domainutils as dutils
from bcause.factors import DeterministicFactor
from bcause.factors.mulitnomial import MultinomialFactor
from bcause.inference.causal.multi import EMCC
from bcause.inference.probabilistic.elimination import VariableElimination
from bcause.inference.causal.elimination import CausalVariableElimination
from bcause.models.cmodel import StructuralCausalModel
from bcause.util import randomUtil

log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format, datefmt='%Y%m%d_%H%M%S')

# Define a DAG and the domains
dag = nx.DiGraph([("X", "Y"), ("U", "Y"), ("U", "X")])
domains = dict(X=["x1", "x2"], Y=["y1","y2"], U=["u1", "u2", "u3", "u4"])


domx = dutils.var_parents_domain(domains,dag,"X")
fx = DeterministicFactor(domx, right_vars=["U"], values=["x1", "x2", "x2", "x1"])

domy = dutils.var_parents_domain(domains,dag,"Y")

# iterates first on the rightmost variable (following the variable in the domain dict)
values = ["y1", "y1", "y2", "y1", "y2", "y2", "y1", "y1"]
fy = DeterministicFactor(domy, left_vars=["Y"], values=values)

# the inner dimension is the rightmost variable
values = [['y1', 'y1', 'y2', 'y1'],['y2', 'y2', 'y1', 'y1']]
fy = DeterministicFactor(domy, left_vars=["Y"], values=values)



domu = dutils.subdomain(domains, "U")
pu = MultinomialFactor(domu, values=[.2, .2, .6, .0])

model = StructuralCausalModel(dag, [fx, fy, pu])

# Run causal inference with Variable Elimination
cve = CausalVariableElimination(model)
p = cve.causal_query("Y", do=dict(X="x1"))

# Run a counterfactual query
cve.counterfactual_query("Y",do=dict(X="x1"), evidence=dict(X="x1"))


# Run Variable elimination as if is a Bayesian network
ve = VariableElimination(model)
ve.query("X") # P(X)
ve.query("Y", conditioning="X") # P(Y|X)


randomUtil.seed(1)
data = model.sampleEndogenous(1000)
inf = EMCC(model, data, max_iter=100, num_runs=20)

inf.causal_query("Y", do=dict(X="x1"))

inf.counterfactual_query("Y", do=dict(X="x1"), evidence=dict(X="x2"))

inf.prob_necessity("X","Y")
inf.prob_sufficiency("X","Y")
inf.prob_necessity_sufficiency("X","Y")