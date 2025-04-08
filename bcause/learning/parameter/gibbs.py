"""
Compute the PMF of exogenous variables given data for endogenous variable using
Gibbs sampling
"""

from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import itertools
import random
from typing import Dict, List, Tuple, Any


import bcause as bc
from bcause.factors import MultinomialFactor, DeterministicFactor
from bcause.factors.mulitnomial import random_multinomial   # TODO: mulitnomial -> multinomial
from bcause.learning.parameter import IterativeParameterLearning
from bcause.models.cmodel import StructuralCausalModel
from bcause.util.domainutils import assingment_space, state_space # TODO: assingment_space -> assignment_space


class GibbsSampling(IterativeParameterLearning):
    '''
     This class implements a method for running a single optimization of the exogenous variables in a SCM.
     '''

    def __init__(self, prior_model: StructuralCausalModel, trainable_vars: list = None):
        self._prior_model = prior_model 
        self._trainable_vars = trainable_vars

    def initialize(self, data: pd.DataFrame, **kwargs):
        self._model = self.prior_model.copy()
        self._process_data(data)
    def _stop_learning(self) -> bool:
        pass


    def _calculate_updated_factors(self, **kwargs) -> dict[MultinomialFactor]:
        return {U: self._updated_factor(U) for U in self.trainable_vars}

    def _updated_factor(self, U) -> MultinomialFactor:

        # todo: replace this code, now it returns a random distribution while it should return the result of a step in gibbs sampling
        f = random_multinomial({U: self._model.domains[U]})
        return f

    def _process_data(self, data: pd.DataFrame):
        # add missing variables
        missing_vars = [v for v in self.prior_model.variables if v not in data.columns]
        for v in missing_vars: data[v] = float("nan")

        # Set as trainable variables those with missing
        self._trainable_vars = self.trainable_vars or list(data.columns[data.isna().any()])

        print(f"trainable: {self.trainable_vars}")

        for v in self._trainable_vars:
            # check exogenous and completely missing
            if not self.prior_model.is_exogenous(v):
                raise ValueError(f"Trainable variable {v} is not exogenous")

            if (~data[v].isna()).any():
                raise ValueError(f"Trainable variable {v} is not completely missing")

        # save the dataset
        self._data = data


if __name__ == "__main__":
    import logging, sys

    log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'

    # logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format, datefmt='%Y%m%d_%H%M%S')


    # Nota: probar también con modelos semi-markovianos
    m = StructuralCausalModel.read("./models/literature/pearl_small.bif")
    data = pd.read_csv("./models/literature/pearl_small.csv")

    gs = GibbsSampling(m)
    gs.run(data, max_iter=10)

    # print the model evolution
    for model_i in gs.model_evolution:
        print(model_i.get_factors(*model_i.exogenous))

