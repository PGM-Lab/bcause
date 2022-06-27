from bcause.factors import DeterministicFactor

from bcause.models import BayesianNetwork
from bcause.models.cmodel import StructuralCausalModel

def causal_to_bnet(cmodel:StructuralCausalModel) -> BayesianNetwork:
    factors = [f.to_multinomial() if isinstance(f, DeterministicFactor) else f for f in cmodel.factors.values()]
    return BayesianNetwork(cmodel.graph, factors)

def bnet_to_causal(bnet:BayesianNetwork) ->  StructuralCausalModel:
    return StructuralCausalModel(bnet.graph, bnet.factors)
