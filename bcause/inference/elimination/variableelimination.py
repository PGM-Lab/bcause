
from __future__ import annotations


import inspect
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
from util.domainutils import create_domain

class Inference(ABC):
    def __init__(self, model:PGModel):
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

    def compile(self, target, evidence = None) -> Inference:
        self._target = target
        self._evidence = evidence or dict()
        self._inference_model = self._preprocess()
        self._compiled = True;
        return self

    @abstractmethod
    def run(self) -> Factor:
        pass

    def query(self, target, evidence = None):
        return self.compile(target, evidence).run()

class VariableElimination(Inference):
    def __init__(self, model:BayesianNetwork, heuristic:Union[Callable,Heuristic]=None):
        # Default value for heuristic
        heuristic = heuristic or min_weight_heuristic

        # Heuristic is not callable
        if isinstance(heuristic, Heuristic):
            heuristic = heuristic_functions[heuristic.name]

        hargs = inspect.getfullargspec(heuristic).args
        if "dag" not in hargs or "to_remove" not in hargs:
            raise ValueError("Input heuristic function must have arguments called 'dag' and 'to_remove'")

        self._heuristic = heuristic
        super(self.__class__, self).__init__(model)

    # todo: implement simplification
    def _preprocess(self) -> BayesianNetwork:
        return self._model

    def run(self) -> MultinomialFactor:

        # Check that target is set
        if not self._compiled:
            raise ValueError("Model not compiled")

        to_remove = [v for v in self._inference_model.variables if v != self._target and v not in self._evidence.keys()]
        ordering = self._heuristic(self._inference_model.network, to_remove=to_remove, varsizes = self._inference_model.varsizes)
        factors = list(self._inference_model.factors.values())

        for select_var in ordering:
            # get relevant factors
            relevant = [f for f in factors if select_var in f.variables]
            # combine them all
            relevant_restr = [f.R(**self._evidence) for f in relevant]
            join = reduce((lambda f1, f2: f1 * f2), relevant_restr)
            # marginalize variable
            fnew = join ** select_var
            # updage factor list
            factors = [f for f in factors if f not in relevant] + [fnew]

        # combine resulting factors and set evidence
        return reduce((lambda f1, f2: f1 * f2), factors).R(**self._evidence)


varsizes = dict(A=5, B=3, C=2, D=2)
bnet = BayesianNetwork("[A][B|A:C][C][D|B]")
bnet.randomize_factors(create_domain(varsizes))




bnet.get_domains(bnet.variables)
evidence = dict(A="a1")
target = "D"
heuristic = min_size_heuristic


inf = VariableElimination(bnet, Heuristic.MIN_SIZE)
inf.query("D", evidence)


inf = VariableElimination(bnet, Heuristic.MIN_FILL)
inf.query("D", evidence)