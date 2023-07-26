import logging
from abc import abstractmethod, ABC
from typing import Callable

from pgmpy.inference import Inference

from bcause.factors.factor import Factor

from bcause.models.cmodel import StructuralCausalModel
from bcause.models.pgmodel import PGModel
from bcause.models.transform.combination import fusion_roots
from bcause.util.arrayutils import as_lists
from bcause.util.assertions import assert_dag_with_nodes

import bcause.util.domainutils as dutils


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
        do_vars = list(do.keys()) if isinstance(do,dict) else sum([list(d.keys()) for d in do],[])
        assert_dag_with_nodes(self.model.graph, do_vars)

        self._inference_model = self._preprocess()
        self._inf = self._prob_inf_fn(self._inference_model)
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

    def query(self, target, do, evidence=None, counterfactual=False, targets_subgraphs = None):
        if counterfactual:
            #if not isinstance(do, dict): raise ValueError("Intervention must be specified in a single dictionary")
            target = as_lists(target)
            targets_subgraphs = targets_subgraphs or [1]*len(target)
            target = [f"{t[0]}_{t[1]}" for t in zip(target, targets_subgraphs)]
        return self.compile(target, do, evidence,counterfactual).run()

    def causal_query(self, target, do, evidence=None):
        return self.query(target, do, evidence=evidence, counterfactual=False)

    def counterfactual_query(self, target, do, evidence=None, targets_subgraphs = None):
        return self.query(target, do, evidence=evidence, counterfactual=True, targets_subgraphs=targets_subgraphs)

    def prob_necessity_sufficiency(self, cause, effect, true_false_cause:tuple=None, true_false_effect:tuple=None):
        pass

    def _process_output(self, result, obs):
        return result.get_value(**obs)

    def prob_necessity(self, cause, effect, true_false_cause:tuple=None, true_false_effect:tuple=None):
        # PN: P(X_{Y=f} = f |X=t, Y=t)   Y->X

        # Determine the true and false states
        Tcause, Fcause = true_false_cause or dutils.identify_true_false(cause, self.model.domains[cause])
        Teffect, Feffect = true_false_effect or dutils.identify_true_false(effect, self.model.domains[effect])

        # Run the query
        return self._process_output(self.counterfactual_query(
            effect,
            do={cause: Fcause},
            evidence={cause: Tcause, effect: Teffect}
        ), {effect: Feffect})

    def prob_sufficiency(self, cause, effect, true_false_cause:tuple=None, true_false_effect:tuple=None):
        # PS: P(X_{Y=t} = t |X=f, Y=f)   Y->X

        # Determine the true and false states
        Tcause, Fcause = true_false_cause or dutils.identify_true_false(cause, self.model.domains[cause])
        Teffect, Feffect = true_false_effect or dutils.identify_true_false(effect, self.model.domains[effect])

        # Run the query
        return self._process_output(self.counterfactual_query(
            effect,
            do={cause: Tcause},
            evidence={cause: Fcause, effect: Feffect}
        ), {effect: Teffect})



class CausalMultiInference(CausalInference):
    def __init__(self, models: list[StructuralCausalModel], causal_inf_fn: Callable):
        self._models = models
        self._model = models[0]
        self._causal_inf = [causal_inf_fn(m) for m in models]


    def compile(self, *args, **kwargs) -> Inference:
        pass

    def _preprocess(self, *args, **kwargs) -> PGModel:
        pass


    @property
    def counterfactual(self):
        return self._counterfactual

    def run(self) -> Factor:
        pass

    def query(self, target, do, evidence=None, counterfactual=False, targets_subgraphs = None):
        return [inf.query(target,do,evidence,counterfactual,targets_subgraphs) for inf in self._causal_inf]

    def _process_output(self, result, obs):
        return [r.get_value(**obs) for r in result]


'''
    def causal_query(self, target, do, evidence=None):
        return [inf.query(target,do,evidence,counterfactual,targets_subgraphs) for inf in self._causal_inf]
    def counterfactual_query(self, target, do, evidence=None, targets_subgraphs = None):
        raise NotImplementedError()
    def prob_necessity_sufficiency(self, cause, effect, true_false_cause:tuple=None, true_false_effect:tuple=None):
        raise NotImplementedError()
    def prob_necessity(self, cause, effect, true_false_cause:tuple=None, true_false_effect:tuple=None):
        raise NotImplementedError()

    def prob_sufficiency(self, cause, effect, true_false_cause:tuple=None, true_false_effect:tuple=None):
        raise NotImplementedError()
'''


