from __future__ import annotations

import inspect
import logging
import time
from abc import ABC, abstractmethod
from functools import reduce
from typing import Callable, Union

from bcause.factors.mulitnomial import MultinomialFactor
from bcause.inference.elimination.ordering import min_weight_heuristic, Heuristic, heuristic_functions
from bcause.inference.inference import ProbabilisticInference, CausalInference
from bcause.models.cmodel import StructuralCausalModel
from bcause.models.pgmodel import PGModel, DiscreteDAGModel
from bcause.models.transform.simplification import minimalize



class VariableElimination(ProbabilisticInference):
    def __init__(self, model: DiscreteDAGModel, heuristic: Union[Callable, Heuristic] = None,  preprocess_flag:bool = False):

        # Default value for heuristic
        heuristic = heuristic or min_weight_heuristic

        # Heuristic is not callable
        if isinstance(heuristic, Heuristic):
            heuristic = heuristic_functions[heuristic.name]

        hargs = inspect.getfullargspec(heuristic).args
        if "dag" not in hargs or "to_remove" not in hargs:
            raise ValueError("Input heuristic function must have arguments called 'dag' and 'to_remove'")

        self._heuristic = heuristic
        self._preprocess_flag = preprocess_flag

        super(self.__class__, self).__init__(model)

    def _preprocess(self) -> DiscreteDAGModel:
        if not self._preprocess_flag:
            return self._model
        return minimalize(self.model, self._target, self._evidence)

    def run(self) -> MultinomialFactor:

        tstart = time.time()
        # Check that target is set
        if not self._compiled:
            raise ValueError("Model not compiled")

        factors = list(self._inference_model.factors.values())
        vars_in_factors = list(reduce(lambda d1,d2 : d1 | d2, [f.domain.keys() for f in factors]))
        to_remove = [v for v in vars_in_factors if v not in self._target and v not in self._evidence.keys()]
        ordering = self._heuristic(self._inference_model.graph, to_remove=to_remove,
                                   varsizes=self._inference_model.varsizes)

        logging.info(f"Starting Variable elimination loop. Ordering: {ordering}")
        logging.debug(f"Current factor list: {[f.name for f in factors]}")

        for select_var in ordering:
            # get relevant factors
            relevant = [f for f in factors if select_var in f.variables]
            logging.debug(f"Removing variable {select_var}. Relevant factors: {[f.name for f in relevant]}")

            # combine them all
            relevant_restr = [f.R(**self._evidence) for f in relevant]
            join = reduce((lambda f1, f2: f1 * f2), relevant_restr)
            # marginalize variable
            fnew = join ** select_var
            # updage factor list
            factors = [f for f in factors if f not in relevant] + [fnew]

            logging.debug(f"Updated factor list: {[f.name for f in factors]}")



        p = reduce((lambda f1, f2: f1 * f2), factors)
        # combine resulting factors and set evidence
        joint = reduce((lambda f1, f2: f1 * f2), factors).R(**self._evidence)
        result = joint / (joint ** self._target)
        self.time = (time.time()-tstart)*1000
        logging.info(f"Finished variable elimination in {self.time} ms.")
        return result


class CausalVariableElimination(CausalInference):
    def __init__(self, model: StructuralCausalModel, heuristic: Union[Callable, Heuristic] = None,  preprocess_flag:bool = False):
        prob_inf_fn = lambda m : VariableElimination(m, heuristic, preprocess_flag)
        super(self.__class__, self).__init__(model.to_multinomial(), prob_inf_fn)


class MultiCausalVE(CausalInference):
    pass