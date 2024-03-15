from abc import ABC, abstractmethod

import pandas as pd

from bcause.factors.factor import Factor
from bcause.inference.probabilistic.datainference import LaplaceInference


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


class MaximumLikelihoodEstimation(ParameterLearning):
    def __init__(self, prior_model, trainable_vars = None):
        self._prior_model = prior_model
        self._trainable_vars = trainable_vars or prior_model.variables
    def run(self, data: pd.DataFrame):
        inf = LaplaceInference(data, self.prior_model.domains)
        factors = dict()
        for v in self.prior_model.variables:
            if v not in self.trainable_vars:
                factors[v] = self.prior_model.factors[v]
            else:
                factors[v] = inf.query(v, conditioning=self.prior_model.get_parents(v))

        self._model = self.prior_model.builder(dag=self.prior_model.graph, factors=factors)


class IterativeParameterLearning(ParameterLearning):

    def step(self, data:pd.DataFrame=None):
        if data is not None: self._process_data(data)
        new_probs = self._calculate_updated_factors()
        self._update_model(new_probs)

    @property
    def model(self):
        return self._model

    @abstractmethod
    def _calculate_updated_factors(self, **kwargs) -> dict[Factor]:
        pass

    @abstractmethod
    def _process_data(self, data:pd.DataFrame=None):
        pass

    @abstractmethod
    def initialize(self, data : pd.DataFrame, **kwargs):
        pass

    @abstractmethod
    def _stop_learning(self) -> bool:
        pass

    @property
    def model_evolution(self):
        if not hasattr(self, "_model_evolution"):
            self._model_evolution = [self.prior_model]
        return self._model_evolution

    def _record_model(self, m):
        if not hasattr(self, "_model_evolution"):
            self._model_evolution = [self.prior_model]
        self._model_evolution.append(m)


    def _update_model(self, new_probs):
        for v in self._model.variables:
            if v not in new_probs: new_probs[v] = self._model.factors[v]
        self._model = self._model.builder(dag=self._model.graph, factors=new_probs, check_factors=False)
        self._record_model(self.model)

    def run(self, data: pd.DataFrame, max_iter: int = float("inf")):
        """
        This method performs a given number of optimization steps.
        Args:
            data: training data.
            max_iter: number of iterations. Default is None and runs util converge.

        Returns:

        """

        self.initialize(data)
        i = 0
        while i < max_iter:
            self.step()
            #print(self.model.factors)
            if self._stop_learning(): break
            i = i+1


