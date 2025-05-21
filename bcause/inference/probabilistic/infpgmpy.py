from typing import Union

import pandas as pd

import bcause.util.randomUtil
from bcause.conversion.pgmpy_conversion import toPgmpyBNet, discrete_to_multinomial
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

# Class for the Belief Propagation inference.
class BeliefPropagationPGMPY(ProbabilisticInference):
    """
    A class that implements probabilistic inference using the Belief Propagation algorithm
    from the pgmpy library, adapted to work with bcause's probabilistic model framework.
    """

    def __init__(self, model: PGModel):
        """
        Initializes the Belief Propagation inference engine.

        Args:
            model (PGModel): The probabilistic graphical model to be used for inference.
        """
        self._model = model
        self._evidence = dict()
        self._target = None
        self._compiled = False

        from pgmpy.inference import BeliefPropagation
        self._pgmpymodel = toPgmpyBNet(self._model)
        self._inf = BeliefPropagation(self._pgmpymodel)

    def run(self) -> Factor:
        """
        Executes the Belief Propagation algorithm to compute the posterior distribution
        of the target variable(s) given the evidence.

        Returns:
            Factor: The resulting probability distribution as a bcause-compatible Factor object.
        """

        p = self._inf.query(self._target, self._evidence, show_progress=False)
        return discrete_to_multinomial(p, left_vars=self._target)

# Class Approximate Inference using Sampling.
class SamplingPGMPY(ProbabilisticInference):
    """
    A class that implements probabilistic inference using the Sampling algorithm
    from the pgmpy library, adapted to work with bcause's probabilistic model framework.
    """

    def __init__(self, model: PGModel, generated_samples:int=10000, samples: pd.DataFrame=None):
        """
        Initializes the Sampling inference engine.

        Args:
            model (PGModel): The probabilistic graphical model to be used for inference.
            n_samples (int): Number of samples to generate.
            samples (pd.DataFrame): Dataframe containing the samples to use. If provided, uses these samples to compute
             the distribution instead of generating samples. Must conform with the provided evidence
        """
        self._model = model
        self._evidence = dict()
        self._target = None
        self._compiled = False
        self._n_samples = generated_samples
        self._samples = samples

        from pgmpy.inference import ApproxInference
        self._inf = ApproxInference(toPgmpyBNet(self._model))

    def run(self) -> Factor:
        """
        Executes the Sampling algorithm to compute the posterior distribution
        of the target variable(s) given the evidence.

        Returns:
            Factor: The resulting probability distribution as a bcause-compatible Factor object.
        """
        _state_names =  {key: self._model._domains[key] for key in self._target if key in self._model._domains.keys()}
        p = self._inf.query(self._target,n_samples= self._n_samples, samples=self._samples, evidence=self._evidence, state_names=_state_names ,show_progress=False)
        return discrete_to_multinomial(p, left_vars=self._target)

if __name__=="__main__":
    from bcause import BayesianNetwork
    from bcause.inference.probabilistic.elimination import VariableElimination
    from bcause.inference.ordering import Heuristic

    import warnings
    warnings.filterwarnings("ignore")

    bnet = BayesianNetwork.read("models/asia.bif")
    #inf = VariableElimination(bnet, heuristic=Heuristic.MIN_FILL)
    #inf = VariableEliminationPGMPY(bnet)
    #inf = BeliefPropagationPGMPY(bnet)
    inf = SamplingPGMPY(bnet, generated_samples=10000)

    # p = inf.query("dysp", evidence= None)
    # print(p)
    #
    # p = inf.query("dysp", evidence=dict(smoke="yes"))
    # print(p)
    #
    p = inf.query("smoke", evidence=dict(dysp="yes"))
    print(p)
    #
    # p = inf.query(["bronc","lung"])
    # print(p)

    # p = inf.query(["smoke", "dysp"])
    # print(p)

    # p = inf.query("either", evidence=None)
    # print(p)

    p = inf.query("smoke", conditioning="dysp")
    print(p)

    p = inf.query(target=["smoke"], conditioning="dysp", evidence=dict(asia="yes"))
    print(p)

    inf = VariableEliminationPGMPY(bnet)
    # p = inf.query(["smoke", "dysp"])
    # print(p)

    p = inf.query("smoke", conditioning="dysp")
    print(p)

    # p = inf.query(target="smoke", conditioning="dysp", evidence=dict(asia="yes"))
    # print(p)

    # from bcause.util.watch import Watch
    # # Check time
    # Watch.start()
    # for i in range(0, 1000):
    #
    #     p = inf.query("bronc")
    #     p = inf.query("bronc", evidence=dict(smoke="yes"))
    #     p = inf.query("bronc", conditioning="smoke")
    #     p = inf.query(["bronc", "lung"])
    #     if i % 100 == 0:
    #         print(i)
    #
    # Watch.stop_print()