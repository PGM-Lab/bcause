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
        logging.getLogger( __name__ ).info(f"Starting inference: target={str(target)} evidence={str(evidence)}")
        #assert_dag_with_nodes(self.model.graph, self._target | self._evidence.keys())

        self._inference_model = self._preprocess_model()
        self._compiled = True;
        return self

    @abstractmethod
    def run(self) -> Factor:
        pass

    def query(self, target, conditioning=None, evidence=None):

        evidence = evidence or dict()
        multi_evidence = {k:evidence[k] for k,v in evidence.items() if type(v) in [list, tuple] and len(v)>1}

        if conditioning is None and len(multi_evidence)==0:
            return self.compile(target, evidence).run()

        target, conditioning = as_lists(target, conditioning)


        if len(multi_evidence)>0:
            return self._query_multi_evidence(conditioning, evidence, multi_evidence, target)


        if not set(target).isdisjoint(conditioning):
            raise ValueError(f"Target {target} and conditioning {conditioning} are not disjoint ")

        logging.getLogger( __name__ ).info("Preparing conditional query")
        p = self.compile(set(target).union(set(conditioning)), evidence).run()

        logging.getLogger( __name__ ).info("Normalising conditional query")
        return p.divide(p.marginalize(*target))

    def _query_multi_evidence(self, conditioning, evidence, multi_evidence, target):
        new_target = target + list(multi_evidence.keys())
        new_evidence = {k: v for k, v in evidence.items() if k not in multi_evidence}
        p1 = self.query(new_target, evidence=new_evidence, conditioning=conditioning)
        p2 = p1.R(**multi_evidence)
        p3 = p2.marginalize(*multi_evidence.keys())
        pout = p3 / p3.marginalize(*target)
        return pout
