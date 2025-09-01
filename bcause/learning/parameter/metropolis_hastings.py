"""
Compute the PMF of exogenous variables given data for endogenous variable using
Metropolis-Hastings

improvements:
 - Accept a perturbation and a transition function as inputs for the method
 - Change dictionary approach to preloaded tables and SQL joins for conditional probability?
 - Sampling only possible values?
"""

from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import dirichlet
import itertools
import random
from typing import Dict, List, Tuple, Any

from sympy.stats.rv import probability

import bcause as bc
from bcause.factors import MultinomialFactor, DeterministicFactor
from bcause.factors.mulitnomial import random_multinomial   # TODO: mulitnomial -> multinomial
from bcause.inference.causal.multi import CausalMultiInference
from bcause.inference.probabilistic.elimination import VariableElimination
from bcause.learning.parameter import IterativeParameterLearning
from bcause.models.cmodel import StructuralCausalModel
from bcause.util.domainutils import assingment_space, state_space # TODO: assingment_space -> assignment_space
from bcause.util.equtils import seq_to_pandas
from bcause.util.graphutils import dcon_nodes

class MetropolisHastingsSampling(IterativeParameterLearning):
    '''
     This class implements a method for running a single optimization of the exogenous variables in a SCM.
     '''

    def __init__(self, prior_model: StructuralCausalModel, trainable_vars: list = None, alpha: dict = None):
        self._prior_model = prior_model.fix_numeric_domains()
        self._trainable_vars = trainable_vars
        self._alpha = alpha
        self._is_prior = True

    def initialize(self, data: pd.DataFrame, **kwargs):
        self._model = self.prior_model.copy()
        self._process_data(data.copy())
        self._get_involved_factors()
        # set the possible sampling for each exogenous
        self._sampling_set, self._exogenous_samples = self._build_sampling_and_exogenous()

        # value to index mapping for exogenous variables
        #self._value_to_index = {U: {v: i for i, v in enumerate(self._model.factors[U].domain[U])} for U in self._trainable_vars}

    def _stop_learning(self) -> bool:
        pass

    def _calculate_updated_factors(self, **kwargs) -> dict[MultinomialFactor]:
        return {U: self._updated_factor(U) for U in self.trainable_vars}

    def _build_sampling_and_exogenous(self):
        # Get the sampling set for the trainable variables
        data = self._data
        model = self._model

        def _create_sampling_set(row, U, endo_f, endo_vars):
            # Get feasible values for the endo variables using R function
            booleans = {k: np.array(endo_f[k].R(**row[v].to_dict()).values,dtype=int).astype(bool) for k, v in endo_vars.items()}
            return set.intersection(*[ set(np.array(endo_f[k].domain[U])[v]) for k,v in booleans.items()] )

        def _get_initial_exogenous(U, sampling_set, model):
            """
            This function controls the Initial U sampling
            Now: it chooses randomly from the sampling set
            """
            probs = [model.factors[U].get_value(**{U: u}) for u in list(sampling_set)]
            return np.random.choice(list(sampling_set), p=probs/np.sum(probs))

        def _loop_over_rows(row, U, endo_f, endo_vars, model):
            sampling_set = _create_sampling_set(row, U, endo_f, endo_vars)
            exogenous_value = _get_initial_exogenous(U, sampling_set, model)
            return sampling_set,exogenous_value

        sampling_set = {}
        initial_exogenous = {}
        for U in self._trainable_vars:
            endo_f = self._endo_factors[U]
            endo_vars = {v: list(filter(lambda item: item != U, f.variables)) for v, f in endo_f.items()}
            sampling_set[U], initial_exogenous[U]= zip(*data.apply(lambda row: _loop_over_rows(row, U, endo_f, endo_vars, model), axis=1))
        return sampling_set, initial_exogenous

    def perturbate(self,set):
        """
        Perturbate the row based on the uniform distribution.
        """
        return random.choice(list(set))

    def _updated_factor(self, U) -> MultinomialFactor:
        m = self._model
        data = self._data
        u_old = self._exogenous_samples[U]
        sampling_set = self._sampling_set[U]

        if self._is_prior == True:
            if self._alpha is None:
                self._set_non_informative_alpha()
            theta = dirichlet.rvs(self._alpha[U])[0]
            self._is_prior = False

        else:

            endo_f = self._endo_factors[U]
            exo_f = self._model.factors[U]

            new_sampling = [s - {v} if len(s) > 1 else s for s,v in zip(sampling_set, u_old) ]
            u_new = [ self.perturbate(s) for s in new_sampling]
            old_prob = np.array([exo_f.get_value(**{U: u}) for u in u_old])
            new_prob = np.array([exo_f.get_value(**{U: u}) for u in u_new])
            ratio = np.minimum(new_prob/old_prob, 1)
            flag = (ratio >= np.random.rand(len(data))).astype(int)
            self._exogenous_samples[U] = np.where(flag == 1, u_new, u_old)

            counts_u = [ np.count_nonzero(self._exogenous_samples[U] == u) for u in exo_f.domain[U] ]
            beta = [int(a + c) for a, c in zip(self._alpha[U], counts_u)]
            theta = dirichlet.rvs(beta)[0]

        f =  MultinomialFactor({U: m.domains[U]}, theta)
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

    def _get_involved_factors(self):
        endo_factors = {
            trainable_var: { successor:
                self._model.factors[successor]
                for successor in self._model.graph.successors(trainable_var)
            }
            for trainable_var in self._trainable_vars
        }
        self._endo_factors = endo_factors

    # def _get_initial_sampling(self, U,  data: pd.DataFrame):
    #     exo_f = self._model.factors[U]
    #     endo_f = self._endo_factors[U]
    #
    #     endo_vars = {v: list(filter(lambda item: item != U, f.variables)) for v, f in endo_f.items()}
    #
    #     # print(np.prod(np.array([ endo_f[k].R(**row[v].to_dict()).values for k, v in endo_vars.items()]),axis=0))
    #     # Get feasible values for the endo variables using R function
    #     feasible_values = np.prod(
    #         np.array([
    #             [f.R(**row).values for row in data[endo_vars[v]].to_dict(orient="records")]
    #             for v, f in endo_f.items()
    #         ]), axis=0
    #     )
    #     # Get the probabilities of the exogenous variable
    #     prob = np.array(exo_f.values) * feasible_values
    #     prob_standard = prob/prob.sum(axis = 1, keepdims=True)
    #
    #     sampled_values = np.array([
    #         np.random.choice(exo_f.domain[U], p=prob_standard[i])
    #         for i in range(len(data))
    #     ])
    #     return sampled_values


    def _set_non_informative_alpha(self):
        self._alpha = {U: np.ones(len(self._model.domains[U])) for U in self._trainable_vars}

if __name__ == "__main__":
    import logging, sys

    log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'

    # logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format, datefmt='%Y%m%d_%H%M%S')


    # Nota: probar también con modelos semi-markovianos

    m = StructuralCausalModel.read("./models/literature/pearl_small.bif")
    #m2 = m.merge_exogenous("V","U")
    #m = StructuralCausalModel.read("./models/modelTest_SM.bif")
    data = pd.read_csv("./models/literature/pearl_small.csv")
    #data = pd.read_csv("./models/modelTest_SM.csv")



    import time

    # Start the timer
    start_time = time.time()
    # Initialize the Metropolis-Hastings sampling transition function "random" or "uniform"
    mhs = MetropolisHastingsSampling(m)
    mhs.run(data, max_iter=5000)

    # End the timer
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    import matplotlib.pyplot as plt

    U_store = np.empty((0,64))
    V_store = np.empty([0,2])

    Q = []

    # print the model evolution
    for model_i in mhs.model_evolution[1000:]:
         inf = CausalMultiInference([model_i])
         q = inf.prob_sufficiency("T","S", true_false_cause=(1,0), true_false_effect=(1,0))[0]
         Q.append(q)


    plt.hist(Q, density=True)
    plt.xlim(0, 1)
    plt.show()
