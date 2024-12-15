from pgmpy.inference import VariableElimination

from bcause.conversion.pgmpy import toPgmpyBNet, discrete_to_multinomial
from bcause.factors.factor import Factor
from bcause.inference.probabilistic import ProbabilisticInference
from bcause.models.pgmodel import PGModel


class VariableEliminationPGMPY(ProbabilisticInference):
    def __init__(self, model: PGModel):
        self._model = model
        self._evidence = dict()
        self._target = None
        self._inf = VariableElimination(toPgmpyBNet(self._model))
        self._compiled = False

    def run(self) -> Factor:
        p = self._inf.query(self._target, self._evidence)
        return discrete_to_multinomial(p, left_vars=self._target)





if __name__=="__main__":
    from bcause import BayesianNetwork


    # todo: silenciar mensajes pgmpy
    bnet = BayesianNetwork.read("models/asia.bif")
    inf = VariableEliminationPGMPY(bnet)

    p = inf.query("bronc")
    print(p)


    p = inf.query("bronc", evidence=dict(smoke="yes"))
    print(p)


    p = inf.query("bronc", conditioning="smoke")
    print(p)

    p = inf.query(["bronc","lung"])
    print(p)