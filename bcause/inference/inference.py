from abc import abstractmethod, ABC

from pgmpy.inference import Inference
from bcause.factors.factor import Factor
from bcause.models.pgmodel import PGModel


class Inference(ABC):
    """
    Abstract base class for performing inference on probabilistic graphical models.

    Properties:
        - model: The probabilistic graphical model on which inference is performed.
        - inference_model: The internal inference representation (e.g., a compiled model).

    Abstract Methods:
        - _preprocess: Prepares the model or data for inference.
        - compile: Compiles or prepares the inference engine.
        - run: Executes the inference process and returns a result.
        - query: Performs a specific query on the model.
    """

    @property
    def model(self) -> PGModel:
        """
        Returns the probabilistic graphical model used for inference.

        Returns:
            PGModel: The graphical model instance.
        """
        return self._model

    @property
    def inference_model(self):
        """
        Returns the internal inference representation used by the implementation.

        Returns:
            The inference representation (e.g., a compiled version of the model).
        """
        return self._inference_model

    @abstractmethod
    def _preprocess(self, *args, **kwargs) -> PGModel:
        """
        Abstract method for preprocessing the model or data before inference.

        Args:
            *args: Positional arguments for preprocessing.
            **kwargs: Keyword arguments for preprocessing.

        Returns:
            PGModel: The preprocessed graphical model.
        """
        pass

    @abstractmethod
    def compile(self, *args, **kwargs) -> Inference:
        """
        Abstract method for compiling or preparing the inference engine.

        Args:
            *args: Positional arguments for compilation.
            **kwargs: Keyword arguments for compilation.

        Returns:
            Inference: The compiled inference object.
        """
        pass

    @abstractmethod
    def run(self) -> Factor:
        """
        Abstract method for running the inference process.

        Returns:
            Factor: The result of the inference process.
        """
        pass

    @abstractmethod
    def query(self, *args, **kwargs):
        """
        Abstract method for performing a specific query on the model.

        Args:
            *args: Positional arguments for querying.
            **kwargs: Keyword arguments for querying.

        Returns:
            The result of the query, dependent on the implementation.
        """
        pass
