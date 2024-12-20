from typing_extensions import Unpack

import logging
import time
import pyAgrum as gum

from pyAgrum import LazyPropagation
from bcause.conversion.pyagrum_conversion import toAgrum, fromAgrum
from bcause.inference.probabilistic import ProbabilisticInference
from bcause.factors import MultinomialFactor
from bcause.models.pgmodel import DiscreteDAGModel

class pyAgrumInference(ProbabilisticInference):
    def __init__(self, model: DiscreteDAGModel, selected_method = LazyPropagation):
        self._model = model
        self._selected_method = selected_method
        self._gum_model = toAgrum(model)
        self._inference = selected_method(self._gum_model)

        # Create the init to the super class ProbabilisticInference
        super().__init__(model)

    def _preprocess(self) -> DiscreteDAGModel:
        return self._model

    def run(self) -> MultinomialFactor:
        tstart = time.time()
        # Check that target is set
        if not self._compiled:
            raise ValueError("Model not compiled")

        # Use self._inference, self._model, self._target, self._evidence to write the posterior
        result_gum = self._inference.posterior(*self._target)
        # Create a gum BayesNet with only the node that we want to query and result_gum as the CPT
        result_bn = gum.BayesNet()
        result_bn.add(*self._target, len(self._model.domains[self._target[0]]))
        result_bn.cpt(*self._target)[:] = result_gum.toarray().flatten()
        # End timer
        self.time = (time.time() - tstart) * 1000
        logging.info(f"Finished variable elimination in {self.time} ms.")
        # Create a MultinomialFactor with the result
        return fromAgrum(result_bn).factors[self._target[0]]













