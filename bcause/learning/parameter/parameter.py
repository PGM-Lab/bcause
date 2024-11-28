from abc import ABC, abstractmethod
import pandas as pd

from bcause.factors.factor import Factor
from bcause.inference.probabilistic.datainference import LaplaceInference


class ParameterLearning(ABC):
    """
    Abstract base class for parameter learning in probabilistic models.
    """

    @property
    def prior_model(self):
        """Returns the prior (initial) model."""
        return self._prior_model

    @property
    def model(self):
        """Returns the learned model."""
        return self._model

    @property
    def trainable_vars(self):
        """Returns the list of variables eligible for learning."""
        return self._trainable_vars


class MaximumLikelihoodEstimation(ParameterLearning):
    """
    Implements Maximum Likelihood Estimation (MLE) for parameter learning.
    """

    def __init__(self, prior_model, trainable_vars=None):
        self._prior_model = prior_model
        self._trainable_vars = trainable_vars or prior_model.variables

    def run(self, data: pd.DataFrame):
        """
        Runs the Maximum Likelihood Estimation on the given data.

        Args:
            data: A pandas DataFrame containing the observed data.
        """
        inf = LaplaceInference(data, self.prior_model.domains)
        factors = dict()

        for v in self.prior_model.variables:
            if v not in self.trainable_vars:
                factors[v] = self.prior_model.factors[v]
            else:
                factors[v] = inf.query(v, conditioning=self.prior_model.get_parents(v))

        self._model = self.prior_model.builder(dag=self.prior_model.graph, factors=factors)


class IterativeParameterLearning(ParameterLearning):
    """
    Abstract class for iterative parameter learning methods.

    Methods:
        - step: Performs a single iteration of parameter learning.
        - initialize: Initializes the learning process with data.
        - _calculate_updated_factors: Abstract method to compute updated factor probabilities.
        - _process_data: Abstract method for data preprocessing.
        - _stop_learning: Abstract method to check the stopping criteria for learning.
        - _update_model: Updates the model with new probabilities.
    """

    def step(self, data: pd.DataFrame = None):
        """
        Executes a single iteration of learning.

        Args:
            data: Optional. A pandas DataFrame containing the observed data.
        """
        if data is not None:
            self._process_data(data)
        new_probs = self._calculate_updated_factors()
        self._update_model(new_probs)

    @property
    def model(self):
        """Returns the current model after iterative updates."""
        return self._model

    @abstractmethod
    def _calculate_updated_factors(self, **kwargs) -> dict[Factor]:
        """
        Abstract method to calculate updated factor probabilities.

        Returns:
            A dictionary mapping variables to updated factors.
        """
        pass

    @abstractmethod
    def _process_data(self, data: pd.DataFrame = None):
        """
        Abstract method for data preprocessing.

        Args:
            data: A pandas DataFrame containing the observed data.
        """
        pass

    @abstractmethod
    def initialize(self, data: pd.DataFrame, **kwargs):
        """
        Abstract method to initialize the learning process.

        Args:
            data: A pandas DataFrame containing the observed data.
            kwargs: Additional initialization parameters.
        """
        pass

    @abstractmethod
    def _stop_learning(self) -> bool:
        """
        Abstract method to determine if learning should stop.

        Returns:
            A boolean indicating whether to stop learning.
        """
        pass

    @property
    def model_evolution(self):
        """
        Tracks the evolution of the model during iterative updates.

        Returns:
            A list of models at each step of learning.
        """
        if not hasattr(self, "_model_evolution"):
            self._model_evolution = [self.prior_model]
        return self._model_evolution

    def _record_model(self, m):
        """
        Records the current state of the model during learning.

        Args:
            m: The model to record.
        """
        if not hasattr(self, "_model_evolution"):
            self._model_evolution = [self.prior_model]
        self._model_evolution.append(m)

    def _update_model(self, new_probs):
        """
        Updates the model with new probabilities and records the change.

        Args:
            new_probs: A dictionary mapping variables to updated factors.
        """
        for v in self._model.variables:
            if v not in new_probs:
                new_probs[v] = self._model.factors[v]
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


