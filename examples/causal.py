import logging
import sys

import networkx as nx

import bcause.util.domainutils as dutils
import bcause.util.graphutils as gutils
from bcause.factors import DeterministicFactor
from bcause.factors.mulitnomial import MultinomialFactor
from bcause.inference.elimination.variableelimination import CausalVariableElimination
from bcause.models import BayesianNetwork
from bcause.models.cmodel import StructuralCausalModel

log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format, datefmt='%Y%m%d_%H%M%S')


dag = nx.DiGraph([("Y", "X"), ("V", "Y"), ("U", "X")])
domains = dict(X=["x1", "x2"], Y=[0, 1], U=["u1", "u2", "u3", "u4"], V=[True, False])

domy = dutils.subdomain(domains, *gutils.relevat_vars(dag, "Y"))
fy = DeterministicFactor(domy, right_vars=["V"], values=[1, 0])

domx = dutils.subdomain(domains, *gutils.relevat_vars(dag, "X"))

data = ["x1", "x1", "x2", "x1", "x1", "x1", "x2", "x1"]
fx = DeterministicFactor(domx, left_vars=["X"], values=data)

domv = dutils.subdomain(domains, "V")
pv = MultinomialFactor(domv, values=[.1, .9])


domu = dutils.subdomain(domains, "U")
pu = MultinomialFactor(domu, values=[.2, .2, .1, .5])

model = StructuralCausalModel(dag, [fx, fy, pu, pv], cast_multinomial=True)

model.factors



inf = CausalVariableElimination(model)
p = inf.causal_query("X", do=dict(Y=1))



MultinomialFactor(dutils.subdomain(domains, "X"), values= [120, 80])

