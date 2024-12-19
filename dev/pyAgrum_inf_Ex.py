import pyAgrum as gum
import networkx as nx
from bcause.conversion.pyagrum import toAgrum


from bcause.inference.probabilistic.elimination import VariableElimination
from bcause.factors import MultinomialFactor,DeterministicFactor
from bcause.models import BayesianNetwork
from bcause.models.cmodel import StructuralCausalModel
from bcause.util import randomUtil
import bcause.util.domainutils as dutils
import bcause.util.graphutils as gutils



# Set the seed
randomUtil.seed(1)

# Toy example
toy_dag = nx.DiGraph([("A", "B"), ("B", "C"),("B", "D")])
domains = dict(A=[0,1],B=[0,1],C=[0,1],D=[0,1])

domA = dutils.subdomain(domains, "A")
pA = MultinomialFactor(domA, values=[.6, .4])

domB = dutils.subdomain(domains, *gutils.relevat_vars(toy_dag, "B"))
pB_A = MultinomialFactor(domB, right_vars=["A"], left_vars=["B"], values=[.7,.3,.4,.6])

domC = dutils.subdomain(domains, *gutils.relevat_vars(toy_dag, "C"))
pC_B = MultinomialFactor(domC,right_vars=["B"], left_vars=["C"], values=[.8,.2,.1,.9])

domD = dutils.subdomain(domains, *gutils.relevat_vars(toy_dag, "D"))
pD_B = MultinomialFactor(domD, right_vars=["B"], left_vars=["D"], values=[.6,.4,.3,.7])

toy_bn = BayesianNetwork(toy_dag, [pA, pB_A, pC_B, pD_B])

# Variable elimination
ve = VariableElimination(toy_bn)
print(ve.query("C"))

# PyAgrum inference methods
toy_gumbn = toAgrum(toy_bn)

## Exact Inference

### Lazy propagation
lp_toy = gum.LazyPropagation(toy_gumbn)
print(lp_toy.posterior("C"))

### Shafer-Shenoy
ss_toy = gum.ShaferShenoyInference(toy_gumbn)
print(ss_toy.posterior("C"))

### Variable Elimination
vegum_toy = gum.VariableElimination(toy_gumbn)
print(vegum_toy.posterior("C"))

## Approx Inference

### Loopy belief propagation


# Define a DAG and the domains
dag = nx.DiGraph([("V1", "V2"), ("V2", "V3"),("V3", "V4"),("U1", "V1"),("U2", "V2"),("U2", "V4"),("U3", "V3")])
model = StructuralCausalModel(dag)
domains = dict(V1=[0,1],V2=[0,1],V3=[0,1],V4=[0,1], U1=[0,1],U2=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],U3=[0,1,2,3])
model.fill_random_factors(domains)
data = model.sample(1000, as_pandas=True)
bn = BayesianNetwork(dag,model.factor_list)

# Variable elimination
ve = VariableElimination(bn)
ve.query("V2")

#gumbn = toAgrum(bn)
#LazyPropagation(gumbn).posterior("V2")

# PyAgrum inference methods
from bcause.inference.probabilistic.infpyagrum import LazyPropagationPYAgrum
aa = LazyPropagationPYAgrum(model)
aa.query("V2")

from bcause.inference.probabilistic.infpyagrum import ShaferShenoyPYAgrum

# Other result
ab = ShaferShenoyPYAgrum(bn)
ab.query("V2")

# Import expectation maximization
from bcause.learning.parameter.expectation_maximization import ExpectationMaximization

# Example of use
em = ExpectationMaximization(model.randomize_factors(model.exogenous, allow_zero=False))
em.run(data, max_iter=10)
var = em.model_evolution[-1].factors["U2"]

# Example of use using pyAgrumInference
em = ExpectationMaximization(model.randomize_factors(model.exogenous, allow_zero=False), inference_method=ShaferShenoyPYAgrum)
em.run(data, max_iter=10)
var = em.model_evolution[-1].factors["U2"]




