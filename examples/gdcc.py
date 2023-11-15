import networkx as nx
import bcause as bc
from bcause.factors import DeterministicFactor
from bcause.factors.factor import Factor
from bcause.inference.causal.multi import EMCC, GDCC
from bcause.inference.causal.elimination import CausalVariableElimination
from bcause.models import BayesianNetwork

from bcause.models.cmodel import StructuralCausalModel

model = StructuralCausalModel.read(f"./models/modelTestA_1.uai").rename_vars(dict(V0="X", V1="Z", V2="Y"))
bc.randomUtil.seed(1)
data = model.sampleEndogenous(1000)
inf = GDCC(model, data, num_runs=20)

# <IntervalProbFactor P(Y), cardinality = (Y:2), values_low=[0.578324993626913,0.2782330759947038], values_up=[0.7217669240052962,0.4216750063730869]> EMCC
p = inf.causal_query("Y", do=dict(X=0))
print(p)

