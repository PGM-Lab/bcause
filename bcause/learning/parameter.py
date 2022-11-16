from abc import ABC, abstractmethod

import pandas as pd

from bcause.factors import DeterministicFactor, MultinomialFactor
from bcause.models.cmodel import StructuralCausalModel
from bcause.models.pgmodel import DiscreteDAGModel


class ParameterLearning(ABC):
    @property
    def prior_model(self):
        return self._prior_model

    @property
    def model(self):
        return self._model

    @property
    def trainable_vars(self):
        return self._trainable_vars

class IterativeParameterLearning(ParameterLearning):

    def step(self, data:pd.DataFrame=None):
        return self._step(self._process_step_args(data))

    @abstractmethod
    def _step(self, **kwargs):
        pass

    @abstractmethod
    def _process_step_args(self, data:pd.DataFrame=None):
        pass

    @abstractmethod
    def run(self, data:pd.DataFrame):
        pass

    @abstractmethod
    def _initialize(self):
        pass

    @abstractmethod
    def stop_inference(self) -> bool:
        pass

    def run(self, data: pd.DataFrame, max_iter: int = None):
        """
        This method performs a given number of optimization steps.
        Args:
            data: training data.
            max_iter: number of iterations. Default is None and runs util converge.

        Returns:

        """

        self._initialize()

        for _ in range(max_iter):
            self.step(data)
            if self.stop_inference(): break


class AbastractExpectationMaximization(IterativeParameterLearning):
    @abstractmethod
    def _expectation(self, **kwargs):
        pass

    @abstractmethod
    def _maximization(self, **kwargs):
        pass

    def _initialize(self):
        self.model = self.prior_model.copy()

    def _step(self, **kwargs):
        counts = self._expectation(kwargs)
        self._maximization(counts)

class ExpectationMaximization(AbastractExpectationMaximization):
    def __init__(self, prior_model : DiscreteDAGModel):
        self.prior_model = prior_model


    def _expectation(self, **kwargs):         # todo: implement
        pass

    def _maximization(self, **kwargs):         # todo: implement
        pass

    def _process_step_args(self, data: pd.DataFrame = None):
        pass

    def stop_inference(self) -> bool:
        # todo: add stopping criteria
        return False






if __name__ == "__main__":

    import networkx as nx

    dag = nx.DiGraph([("Y" ,"X"), ("V" ,"Y"), ("U" ,"X")])
    domains = dict(X=["x1", "x2"], Y=[0, 1], U=["u1", "u2", "u3", "u4"], V=[True, False])

    import bcause.util.domainutils as dutils
    import bcause.util.graphutils as gutils



    domy = dutils.subdomain(domains, *gutils.relevat_vars(dag, "Y"))
    fy = DeterministicFactor(domy, right_vars = ["V"], data=[1,0])

    domx = dutils.subdomain(domains, *gutils.relevat_vars(dag, "X"))

    # todo: check data shape for this
    data = ["x1", "x1", "x2", "x1","x1", "x1", "x2", "x1"]
    fx = DeterministicFactor(domx, left_vars = ["X"], data=data)

    domv = dutils.subdomain(domains, "V")
    pv = MultinomialFactor(domv, data = [.1, .9])

    domu = dutils.subdomain(domains, "U")
    pu = MultinomialFactor(domu, data = [.2, .2, .1, .5])

    m = StructuralCausalModel(dag, [fx, fy, pu, pv], cast_multinomial=True)


    data = m.sample(100, as_pandas=True)

    data.groupby(list(data.columns))

    data.value_counts()