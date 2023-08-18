import logging
from abc import abstractmethod

from bcause.factors.factor import Factor
from bcause.inference.inference import Inference
from bcause.models.pgmodel import PGModel
from bcause.util.arrayutils import as_lists
from bcause.util.assertions import assert_dag_with_nodes


class ProbabilisticInference(Inference):
    def __init__(self, model: PGModel):
        self._model = model
        self._evidence = dict()
        self._target = None
        self._compiled = False

    def compile(self, target, evidence=None) -> Inference:

        target = as_lists(target)
        if len(set(target)) != len(target): raise ValueError("Repeated variables in target")
        if not set(target).isdisjoint(evidence.keys()):
            raise ValueError(f"Target {target} and evidence are not disjoint {evidence.keys()}")

        self._target = target
        self._evidence = evidence or dict()
        logging.info(f"Starting inference: target={str(target)} evidence={str(evidence)}")
        assert_dag_with_nodes(self.model.graph, self._target | self._evidence.keys())

        self._inference_model = self._preprocess()
        self._compiled = True;
        return self

    @abstractmethod
    def run(self) -> Factor:
        pass

    def query(self, target, conditioning=None, evidence=None):

        evidence = evidence or dict()

        if conditioning is None:
            return self.compile(target, evidence).run()

        target, conditioning = as_lists(target, conditioning)

        if not set(target).isdisjoint(conditioning):
            raise ValueError(f"Target {target} and conditioning {conditioning} are not disjoint ")

        logging.info("Preparing conditional query")
        p = self.compile(set(target).union(set(conditioning)), evidence).run()

        logging.info("Normalising conditional query")
        return p.divide(p.marginalize(*target))
