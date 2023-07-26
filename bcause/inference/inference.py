from abc import abstractmethod, ABC

from pgmpy.inference import Inference

from bcause.factors.factor import Factor

from bcause.models.pgmodel import PGModel


class Inference(ABC):

    @property
    def model(self) -> PGModel:
        return self._model

    @property
    def inference_model(self):
        return self._inference_model

    @abstractmethod
    def _preprocess(self, *args, **kwargs) -> PGModel:
        pass

    @abstractmethod
    def compile(self, *args, **kwargs) -> Inference:
        pass

    @abstractmethod
    def run(self) -> Factor:
        pass

    @abstractmethod
    def query(self, *args, **kwargs):
        pass


