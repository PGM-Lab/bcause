import logging
from abc import abstractmethod, ABC

from pgmpy.inference import Inference

from bcause.factors.factor import Factor
from bcause.models.pgmodel import PGModel
from bcause.util.assertions import assert_dag_with_nodes


class Inference(ABC):
    def __init__(self, model: PGModel):
        self._model = model
        self._evidence = dict()
        self._target = None
        self._compiled = False

    @property
    def model(self):
        return self._model

    @abstractmethod
    def _preprocess(self, *args, **kwargs) -> PGModel:
        pass

    def compile(self, target, evidence=None) -> Inference:
        logging.info(f"Starting inference: target={str(target)} evidence={str(evidence)}")

        if type(target) not in [list, set]: target = [target]
        self._target = target
        self._evidence = evidence or dict()

        if not set(self._target).isdisjoint(self._evidence.keys()):
            raise ValueError(f"Target {self._target} and evidence are not disjoint {self._evidence.keys()}")
        assert_dag_with_nodes(self.model.graph, self._target | self._evidence.keys())

        self._inference_model = self._preprocess()
        self._compiled = True;
        return self

    @abstractmethod
    def run(self) -> Factor:
        pass

    def query(self, target, conditioning=None, evidence=None):

        if type(target) not in [list, set]: target = [target]
        evidence = evidence or dict()

        if conditioning is None:
            return self.compile(target, evidence).run()

        if type(conditioning) not in [list, set]: conditioning = [conditioning]
        if not set(target).isdisjoint(conditioning):
            raise ValueError(f"Target {target} and conditioning {conditioning} are not disjoint ")

        logging.info("Preparing conditional query")
        p = self.compile(set(target).union(set(conditioning)), evidence).run()

        logging.info("Normalising conditional query")
        return p.divide(p.marginalize(*target))



