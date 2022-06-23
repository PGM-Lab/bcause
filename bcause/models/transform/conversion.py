from bcause.factors import DeterministicFactor

from bcause.models import BayesianNetwork
from bcause.models.cmodel import CausalModel

def causal_to_bnet(cmodel:CausalModel) -> BayesianNetwork:
    factors = [f.to_multinomial() if isinstance(f, DeterministicFactor) else f for f in cmodel.factors.values()]
    return BayesianNetwork(cmodel.graph, factors)

def bnet_to_causal(bnet:BayesianNetwork) ->  CausalModel:
    return CausalModel(bnet.graph, bnet.factors)
