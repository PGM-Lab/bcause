import logging
from abc import abstractmethod, ABC
from collections.abc import Iterable
from typing import Callable

from pgmpy.inference import Inference

from bcause.factors.factor import Factor
from bcause.models.cmodel import StructuralCausalModel
from bcause.models.pgmodel import PGModel
from bcause.models.transform.combination import fusion_roots
from bcause.util.arrayutils import as_lists
from bcause.util.assertions import assert_dag_with_nodes



class Inference(ABC):

    @property
    def model(self) -> PGModel:
        return self._model

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

class CausalInference(Inference):
    def __init__(self, model: StructuralCausalModel, prob_inf_fn: Callable):
        self._model = model
        self._inf = None
        self._prob_inf_fn = prob_inf_fn
        self._compiled = False

    def compile(self, target, do, evidence=None, counterfactual=False) -> Inference:
        target = as_lists(target)
        evidence = evidence or dict()

        self._do = do
        self._counterfactual = counterfactual

        logging.info(f"Starting causal inference: target={str(target)}  intervention={str(do)} evidence={str(evidence)}")
        assert_dag_with_nodes(self.model.graph, do.keys())

        self._inf = self._prob_inf_fn(self._preprocess())
        self._inf.compile(target, evidence)
        self._compiled = True

        return self

    def _preprocess(self, *args, **kwargs) -> PGModel:

        if not self._counterfactual:
            new_model = self.model.intervention(**self._do)
            logging.debug(f"Intervened DAG: {new_model.graph.edges}")

        else:
            do = self._do if isinstance(self._do, list) else [self._do]
            models = [self.model]
            for i in range(0, len(do)):
                mapping = {v: f"{v}_{i+1}" for v in self.model.endogenous}
                models.append(self.model.intervention(**do[i]).rename_vars(mapping))

            new_endogenous = sum([m.endogenous for m in models],[])
            new_model = fusion_roots(models, on=self.model.exogenous, endogenous=new_endogenous)
            logging.debug(f"Counterfactual DAG: {new_model.graph.edges}")

        return new_model


    @property
    def counterfactual(self):
        return self._counterfactual

    def run(self) -> Factor:
        return self._inf.run()

    def query(self, target, do, evidence=None, counterfactual=False):
        if counterfactual:
            if not isinstance(do, dict): raise ValueError("Intervention must be specified in a single dictionary")
            target = [f"{v}_1" for v in as_lists(target)]
        return self.compile(target, do, evidence,counterfactual).run()

    def causal_query(self, target, do, evidence=None):
        return self.query(target, do, evidence=evidence, counterfactual=False)

    def counterfactual_query(self, target, do, evidence):
        return self.query(target, do, evidence=evidence, counterfactual=True)

    def prob_necessity_sufficiency(self, cause, effect, true_states:dict=None, false_states:dict=None):
        pass