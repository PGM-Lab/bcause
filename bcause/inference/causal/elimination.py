from __future__ import annotations

from typing import Union, Callable

from bcause.inference.ordering import Heuristic
from bcause.inference.probabilistic.elimination import VariableElimination
from bcause.inference.causal.causal import CausalInference
from bcause.models.cmodel import StructuralCausalModel
from bcause.models.transform.combination import counterfactual_model


class CausalVariableElimination(CausalInference):
    def __init__(self, model: StructuralCausalModel, heuristic: Union[Callable, Heuristic] = None,  preprocess_flag:bool = True):
        prob_inf_fn = lambda m : VariableElimination(m, heuristic)
        super(self.__class__, self).__init__(model.to_multinomial(), prob_inf_fn)

    def get_counterfactual_model(self, do, exclude_from_common=None):
        return counterfactual_model(self.model, do, exclude_from_common)
