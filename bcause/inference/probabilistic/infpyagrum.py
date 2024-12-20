from torch.utils.model_dump import burn_in_info

from bcause.conversion.pyagrum_conversion import toAgrum, potential_to_factor
from bcause.inference.probabilistic import ProbabilisticInference
from bcause.factors import MultinomialFactor
from bcause.models.pgmodel import PGModel


class LazyPropagationPYAgrum(ProbabilisticInference):
    """
    A class that implements probabilistic inference using the Lazy Propagation algorithm
    from the pyAgrum library, adapted to work with bcause's probabilistic model framework.
    """
    def __init__(self, model: PGModel):
        self._model = model
        self._evidence = dict()
        self._target = None
        self._compiled = False

        from pyAgrum import LazyPropagation
        self._inf = LazyPropagation(toAgrum(self.model))

    def _preprocess_model(self) -> PGModel:
        """
        Preprocess the model before running the inference algorithm.
        Set the evidence in the model and make inference if evidence is present.
        """
        self._inf.setEvidence(self._evidence)
        self._inf.makeInference()
        return self._model

    def run(self) -> MultinomialFactor:
        """
        Executes the Lazy Propagation algorithm to compute the posterior distribution
        of the target variable(s) given the evidence.

        Returns:
            Factor: The resulting probability distribution as a bcause-compatible Factor object.
        """
        if len(self._target) == 1:
            self._inf.addTarget(*self._target)
            p = self._inf.posterior(*self._target)
        else:
            self._inf.addJointTarget(set(self._target))
            p = self._inf.jointPosterior(set(self._target))
        # Create a MultinomialFactor with the result
        return potential_to_factor(p)

class ShaferShenoyPYAgrum(ProbabilisticInference):
    """
    A class that implements probabilistic inference using the Shafer-Shenoy algorithm
    from the pyAgrum library, adapted to work with bcause's probabilistic model framework.
    """
    def __init__(self, model: PGModel):
        self._model = model
        self._evidence = dict()
        self._target = None
        self._compiled = False

        from pyAgrum import ShaferShenoyInference
        self._inf = ShaferShenoyInference(toAgrum(self.model))

    def _preprocess_model(self) -> PGModel:
        """
        Preprocess the model before running the inference algorithm.
        Set the evidence in the model and make inference if evidence is present.
        """
        self._inf.setEvidence(self._evidence)
        self._inf.makeInference()
        return self._model

    def run(self) -> MultinomialFactor:
        """
        Executes the Shafer-Shenoy algorithm to compute the posterior distribution
        of the target variable(s) given the evidence.

        Returns:
            Factor: The resulting probability distribution as a bcause-compatible Factor object.
        """

        if len(self._target) == 1:
            self._inf.addTarget(*self._target)
            p = self._inf.posterior(*self._target)
        else:
            self._inf.addJointTarget(set(self._target))
            p = self._inf.jointPosterior(set(self._target))
        # Create a MultinomialFactor with the result
        return potential_to_factor(p)

class VariableEliminationPYAgrum(ProbabilisticInference):
    """
    A class that implements probabilistic inference using the Variable Elimination algorithm
    from the pyAgrum library, adapted to work with bcause's probabilistic model framework.
    """
    def __init__(self, model: PGModel):
        self._model = model
        self._evidence = dict()
        self._target = None
        self._compiled = False

        from pyAgrum import VariableElimination
        self._inf = VariableElimination(toAgrum(self.model))

    def _preprocess_model(self) -> PGModel:
        """
        Preprocess the model before running the inference algorithm.
        Set the evidence in the model and make inference if evidence is present.
        """
        self._inf.setEvidence(self._evidence)
        self._inf.makeInference()
        return self._model

    def run(self) -> MultinomialFactor:
        """
        Executes the Variable Elimination algorithm to compute the posterior distribution
        of the target variable(s) given the evidence.

        Returns:
            Factor: The resulting probability distribution as a bcause-compatible Factor object.
        """
        print(self._target)
        if len(self._target) == 1:
            self._inf.addTarget(*self._target)
            p = self._inf.posterior(*self._target)
        else:
            self._inf.addJointTarget(set(self._target))
            p = self._inf.jointPosterior(set(self._target))
        # Create a MultinomialFactor with the result
        return potential_to_factor(p)

class LoopyBeliefPropagationPYAgrum(ProbabilisticInference):
    """
    A class that implements probabilistic inference using the Loopy Belief Propagation algorithm
    from the pyAgrum library, adapted to work with bcause's probabilistic model framework.
    """
    def __init__(self, model: PGModel):
        self._model = model
        self._evidence = dict()
        self._target = None
        self._compiled = False

        from pyAgrum import LoopyBeliefPropagation
        self._inf = LoopyBeliefPropagation(toAgrum(self.model))

    def _preprocess_model(self) -> PGModel:
        """
        Preprocess the model before running the inference algorithm.
        Set the evidence in the model and make inference if evidence is present.
        """
        self._inf.setEvidence(self._evidence)
        self._inf.makeInference()
        return self._model

    def run(self) -> MultinomialFactor:
        """
        Executes the Loopy Belief Propagation algorithm to compute the posterior distribution
        of the target variable(s) given the evidence.

        Returns:
            Factor: The resulting probability distribution as a bcause-compatible Factor object.
        """

        if len(self._target) == 1:
            self._inf.addTarget(*self._target)
            p = self._inf.posterior(*self._target)
        else:
            self._inf.addJointTarget(set(self._target))
            p = self._inf.jointPosterior(set(self._target))
        # Create a MultinomialFactor with the result
        return potential_to_factor(p)

class GibbsSamplingPYAgrum(ProbabilisticInference):
    """
    A class that implements probabilistic inference using the Gibbs Sampling algorithm
    from the pyAgrum library, adapted to work with bcause's probabilistic model framework.
    """
    def __init__(self, model: PGModel, burn_in: int = 1000, max_iter: int = 10000):
        self._model = model
        self._evidence = dict()
        self._target = None
        self._compiled = False
        self._burn_in = burn_in
        self._max_iter = max_iter

        from pyAgrum import GibbsSampling
        self._inf = GibbsSampling(toAgrum(self.model))

    def _preprocess_model(self) -> PGModel:
        """
        Preprocess the model before running the inference algorithm.
        Set the evidence in the model and make inference if evidence is present.
        """
        self._inf.setBurnIn(self._burn_in)
        self._inf.setMaxIter(self._max_iter)
        self._inf.setEvidence(self._evidence)
        self._inf.makeInference()
        return self._model

    def run(self) -> MultinomialFactor:
        """
        Executes the Gibbs Sampling algorithm to compute the posterior distribution
        of the target variable(s) given the evidence.

        Returns:
            Factor: The resulting probability distribution as a bcause-compatible Factor object.
        """
        self._inf.addTarget(*self._target)
        p = self._inf.posterior(*self._target)
        # Create a MultinomialFactor with the result
        return potential_to_factor(p)

class MonteCarloSamplingPyAgrum(ProbabilisticInference):
    """
    A class that implements probabilistic inference using the Monte Carlo Sampling algorithm
    from the pyAgrum library, adapted to work with bcause's probabilistic model framework.
    """
    def __init__(self, model: PGModel,  max_iter: int = 10000):
        self._model = model
        self._evidence = dict()
        self._target = None
        self._compiled = False
        self._max_iter = max_iter

        from pyAgrum import MonteCarloSampling
        self._inf = MonteCarloSampling(toAgrum(self.model))

    def _preprocess_model(self) -> PGModel:
        """
        Preprocess the model before running the inference algorithm.
        Set the evidence in the model and make inference if evidence is present.
        """
        self._inf.setMaxIter(self._max_iter)
        self._inf.setEvidence(self._evidence)
        self._inf.makeInference()
        return self._model

    def run(self) -> MultinomialFactor:
        """
        Executes the Monte Carlo Sampling algorithm to compute the posterior distribution
        of the target variable(s) given the evidence.

        Returns:
            Factor: The resulting probability distribution as a bcause-compatible Factor object.
        """

        self._inf.addTarget(*self._target)
        p = self._inf.posterior(*self._target)
        # Create a MultinomialFactor with the result
        return potential_to_factor(p)

class WeightedSamplingPyAgrum(ProbabilisticInference):
    """
    A class that implements probabilistic inference using the Weighted Sampling algorithm
    from the pyAgrum library, adapted to work with bcause's probabilistic model framework.
    """
    def __init__(self, model: PGModel,  max_iter: int = 10000):
        self._model = model
        self._evidence = dict()
        self._target = None
        self._compiled = False
        self._max_iter = max_iter

        from pyAgrum import WeightedSampling
        self._inf = WeightedSampling(toAgrum(self.model))

    def _preprocess_model(self) -> PGModel:
        """
        Preprocess the model before running the inference algorithm.
        Set the evidence in the model and make inference if evidence is present.
        """
        self._inf.setMaxIter(self._max_iter)
        self._inf.setEvidence(self._evidence)
        self._inf.makeInference()
        return self._model

    def run(self) -> MultinomialFactor:
        """
        Executes the Weighted Sampling algorithm to compute the posterior distribution
        of the target variable(s) given the evidence.

        Returns:
            Factor: The resulting probability distribution as a bcause-compatible Factor object.
        """

        self._inf.addTarget(*self._target)
        p = self._inf.posterior(*self._target)
        # Create a MultinomialFactor with the result
        return potential_to_factor(p)

class ImportanceSamplingPyAgrum(ProbabilisticInference):
    """
    A class that implements probabilistic inference using the Importance Sampling algorithm
    from the pyAgrum library, adapted to work with bcause's probabilistic model framework.
    """
    def __init__(self, model: PGModel):
        self._model = model
        self._evidence = dict()
        self._target = None
        self._compiled = False

        from pyAgrum import ImportanceSampling
        self._inf = ImportanceSampling(toAgrum(self.model))

    def _preprocess_model(self) -> PGModel:
        """
        Preprocess the model before running the inference algorithm.
        Set the evidence in the model and make inference if evidence is present.
        """
        self._inf.setEvidence(self._evidence)
        self._inf.makeInference()
        return self._model

    def run(self) -> MultinomialFactor:
        """
        Executes the Importance Sampling algorithm to compute the posterior distribution
        of the target variable(s) given the evidence.

        Returns:
            Factor: The resulting probability distribution as a bcause-compatible Factor object.
        """
        if len(self._target) == 1:
            p = self._inf.posterior(*self._target)
        else:
            p = self._inf.jointPosterior(set(self._target))
        # Create a MultinomialFactor with the result
        return potential_to_factor(p)

if __name__=="__main__":
    import warnings
    from bcause import BayesianNetwork

    warnings.filterwarnings("ignore")

    bnet = BayesianNetwork.read("models/asia.bif")
    #inf = LazyPropagationPYAgrum(bnet)
    #inf = ShaferShenoyPYAgrum(bnet)
    inf = VariableEliminationPYAgrum(bnet)
    #inf = LoopyBeliefPropagationPYAgrum(bnet)
    #inf = GibbsSamplingPYAgrum(bnet, burn_in=1000, max_iter=10000)
    #inf = MonteCarloSamplingPyAgrum(bnet)
    #inf = WeightedSamplingPyAgrum(bnet)
    #inf = ImportanceSamplingPyAgrum(bnet)

    # p = inf.query("dysp")
    # print(p)
    #
    # p = inf.query("dysp", evidence=dict(smoke="yes"))
    # print(p)
    #
    # p = inf.query("smoke", evidence=dict(dysp="yes"))
    # print(p)
    #
    # p = inf.query(["bronc","lung"])
    # print(p)

    p = inf.query(["smoke", "dysp"])
    print(p)

    p = inf.query("smoke", conditioning="dysp")
    print(p)
    #
    # p = inf.query(target="smoke", conditioning="dysp", evidence=dict(asia="yes"))
    # print(p)


    from bcause.util.watch import Watch

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