import networkx as nx
import bcause as bc
from bcause.factors import DeterministicFactor
from bcause.factors.factor import Factor
from bcause.inference.causal.multi import EMCC
from bcause.inference.causal.elimination import CausalVariableElimination
from bcause.models import BayesianNetwork

from bcause.models.cmodel import StructuralCausalModel

model = StructuralCausalModel.read(f"./models/modelTestA_1.uai").rename_vars(dict(V0="X", V1="Z", V2="Y"))
bc.randomUtil.seed(1)
data = model.sampleEndogenous(1000)
inf = EMCC(model, data, max_iter=100, num_runs=20)


p = inf.causal_query("Y", do=dict(X=0))
print(p)

