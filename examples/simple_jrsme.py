import logging
import sys

import networkx as nx

import bcause.util.domainutils as dutils
import bcause.util.graphutils as gutils
from bcause.factors import DeterministicFactor
from bcause.factors.mulitnomial import MultinomialFactor
from bcause.inference.elimination.variableelimination import CausalVariableElimination,VariableElimination
from bcause.models.cmodel import StructuralCausalModel

log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format, datefmt='%Y%m%d_%H%M%S')

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

# Run causal inference with Variable Elimination
cve = CausalVariableElimination(model)
p = cve.causal_query("Y", do=dict(X="x1"))

# Run a counterfactual query
cve.counterfactual_query("Y",do=dict(X="x1"), evidence=dict(X="x1"))


# Run Variable elimination as if is a Bayesian network
ve = VariableElimination(model)
ve.query("X") # P(X)
ve.query("Y", conditioning="X") # P(Y|X)

#### Access to the factors and operate with them

model.factors

px = model.factors["X"]
py = model.factors["Y"]
pu = model.factors["U"]
pv = model.factors["V"]

joint = px * pv * py * pu

joint_xy = joint.marginalize("U","V")

joint_xy.marginalize("X")

marg_x = joint_xy.marginalize("X")
cond_y= joint_xy / marg_x
