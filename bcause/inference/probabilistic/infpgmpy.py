from typing import Union

from bcause.conversion.pgmpy import toPgmpyBNet, discrete_to_multinomial
from bcause.factors.factor import Factor
from bcause.inference.probabilistic import ProbabilisticInference
from bcause.models.pgmodel import PGModel

class VariableEliminationPGMPY(ProbabilisticInference):
    """
    A class that implements probabilistic inference using the Variable Elimination algorithm
    from the pgmpy library, adapted to work with bcause's probabilistic model framework.
    """

    def __init__(self, model: PGModel, elimination_order:Union[str,list]="minfill"):
        """
        Initializes the Variable Elimination inference engine.

        Args:
            model (PGModel): The probabilistic graphical model to be used for inference.

            elimination_order: str or list (default='minfill') Order in which to eliminate the variables in the algorithm. If list is provided,
            should contain all variables in the model except the ones in `variables`. str options
            are:  `WeightedMinFill`, `MinNeighbors`, `MinWeight`, `MinFill`. Please
            refer https://pgmpy.org/exact_infer/ve.html#module-pgmpy.inference.EliminationOrder
            for details.
        """
        self._model = model
        self._evidence = dict()
        self._target = None
        self._compiled = False
        self._elimination_order = elimination_order

        from pgmpy.inference import VariableElimination
        self._inf = VariableElimination(toPgmpyBNet(self._model))

    def run(self) -> Factor:
        """
        Executes the Variable Elimination algorithm to compute the posterior distribution
        of the target variable(s) given the evidence.

        Returns:
            Factor: The resulting probability distribution as a bcause-compatible Factor object.
        """
        p = self._inf.query(self._target, self._evidence, show_progress=False, elimination_order=self._elimination_order)
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