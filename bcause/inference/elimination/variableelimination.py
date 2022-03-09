from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from functools import reduce
from typing import Callable, Union

import networkx as nx

from factors.factor import Factor
from factors.mulitnomial import MultinomialFactor
from inference.elimination.ordering import min_weight_heuristic, min_size_heuristic, Heuristic, heuristic_functions, \
    min_fill_heuristic
from models.bnet import BayesianNetwork
from models.pgmodel import PGModel
from models.transform.simplification import minimalize
from util.assertions import assert_dag_with_nodes
from util.domainutils import create_domain
from util.graphutils import barren_removal, dsep_nodes, barren_nodes, remove_nodes


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


        self._target = target
        self._evidence = evidence or dict()

        assert_dag_with_nodes(self.model.graph, {self._target} | self._evidence.keys())

        self._inference_model = self._preprocess()
        self._compiled = True;
        return self

    @abstractmethod
    def run(self) -> Factor:
        pass

    def query(self, target, evidence=None):
        return self.compile(target, evidence).run()


class VariableElimination(Inference):
    def __init__(self, model: BayesianNetwork, heuristic: Union[Callable, Heuristic] = None,  preprocess_flag:bool = True):

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

    def _preprocess(self) -> BayesianNetwork:
        if not self._preprocess_flag:
            return self._model
        return minimalize(self.model, self._target, self._evidence)

    def run(self) -> MultinomialFactor:

        # Check that target is set
        if not self._compiled:
            raise ValueError("Model not compiled")

        to_remove = [v for v in self._inference_model.variables if v != self._target and v not in self._evidence.keys()]
        ordering = self._heuristic(self._inference_model.graph, to_remove=to_remove,
                                   varsizes=self._inference_model.varsizes)
        factors = list(self._inference_model.factors.values())

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

        # combine resulting factors and set evidence
        joint = reduce((lambda f1, f2: f1 * f2), factors).R(**self._evidence)
        result = joint / (joint ** self._target)

        logging.info(f"Finished Variable elimination.")
        return result
