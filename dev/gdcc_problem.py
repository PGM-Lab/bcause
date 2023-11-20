import networkx as nx

from bcause.factors import MultinomialFactor, DeterministicFactor 
from bcause.learning.parameter.gradient import GradientLikelihood
from bcause.models.cmodel import StructuralCausalModel

dag = nx.DiGraph([("Y", "X"), ("V", "Y"), ("U", "X")])
domains = dict(X=["x1", "x2"], Y=[0, 1], U=["u1", "u2", "u3", "u4"], V=["v1", "v2"])

import bcause.util.domainutils as dutils
import bcause.util.graphutils as gutils

domy = dutils.subdomain(domains, *gutils.relevat_vars(dag, "Y"))
fy = DeterministicFactor(domy, right_vars=["V"], values=[1, 0])
domx = dutils.subdomain(domains, *gutils.relevat_vars(dag, "X"))
values = ["x1", "x1", "x2", "x1", "x1", "x1", "x2", "x1"]

from bcause.factors.deterministic import canonical_specification
can_values = canonical_specification(V_domain = domx['X'], Y_domains = [domy['Y']]) # TODO: finalize this
#can_values = canonical_specification(V_domain = ('v1', 'v2', 'v3'), Y_domains = [('y1', 'y2', 'y3'), ('z1', 'z2', 'z3', 'z4')]) 
#can_values = canonical_specification(V_domain = ('v1', 'v2'), Y_domains = [('y1', 'y2', 'y3')]) 

fx = DeterministicFactor(domx, left_vars=["X"], values=can_values)
domv = dutils.subdomain(domains, "V")
pv = MultinomialFactor(domv, values=[.1, .9])
domu = dutils.subdomain(domains, "U")
pu = MultinomialFactor(domu, values=[.2, .2, .1, .5])

m = StructuralCausalModel(dag, [fx, fy, pu, pv], cast_multinomial=True)

data = m.sample(10000, as_pandas=True)[m.endogenous]


gl = GradientLikelihood(m.randomize_factors(m.exogenous, allow_zero=False))
gl.run(data)
