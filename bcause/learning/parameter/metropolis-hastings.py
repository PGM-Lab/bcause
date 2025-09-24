"""
Compute the PMF of exogenous variables given data for endogenous variable using
Metropolis-Hastings

improvements:
 - Change dictionary approach to preloaded tables and SQL joins for conditional probability
 - Sampling only possible values?
 - Accept a perturbation and a transition function as inputs for the method
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
from bcause.factors.multinomial import random_multinomial   # TODO: mulitnomial -> multinomial
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

    def __init__(self, prior_model: StructuralCausalModel, trainable_vars: list = None, transition_select: str = "uniform", perturbation_select: str = "uniform", alpha: dict = None):
        self._prior_model = prior_model.fix_numeric_domains()
        self._trainable_vars = trainable_vars
        self._function_select(transition_select, perturbation_select)
        self._alpha = alpha
        self._is_prior = True

    def initialize(self, data: pd.DataFrame, **kwargs):
        self._model = self.prior_model.copy()
        self._process_data(data)
        self._get_involved_factors()
        # Initial sampling of U
        self._initial_sampling = {U: self._get_precise_sampling(U,data) for U in self._trainable_vars}

    def _stop_learning(self) -> bool:
        pass

    def _calculate_updated_factors(self, **kwargs) -> dict[MultinomialFactor]:
        return {U: self._updated_factor(U) for U in self.trainable_vars}

    def _updated_factor(self, U) -> MultinomialFactor:
        m = self._model
        data = self._data

        if self._is_prior == True:
            if self._alpha is None:
                self._set_non_informative_alpha()
            theta = dirichlet.rvs(self._alpha[U])[0]
            self._is_prior = False

        else:
            def filter_columns(factor, columns):
                return [col for col in columns if col in factor.variables]

            endo_f = self._endo_factors[U]
            exo_f = self._model.factors[U]

            # Probabilities with the previous U
            data['old'] = self._initial_sampling[U]
            prob_u_old = np.array(exo_f.values)[data['old'].to_numpy()]
            conditional_probs_old = {v: np.ones(len(data)) for v in endo_f.keys()} # base ("old") U should be always compatible

            # Proposal distribution to change u
            transition_matrix = self._transition_function(U)
            # How u changes
            data[U] = np.array([self._perturbation_function(u,U, transition_matrix) for u in self._initial_sampling[U]])
            prob_u_new = np.array(exo_f.values)[data[U].to_numpy()]

            filt_cols = {v: filter_columns(f, data.columns) for v, f in endo_f.items()}
            conditional_probs_new = {
                v: [f.get_value(**row) for row in data[filt_cols[v]].to_dict(orient="records")]
                for v, f in endo_f.items()}

            data = data.rename(columns={U: 'new'})

            # Generate U values
            prob_new = np.prod(np.array(list(conditional_probs_new.values())), axis = 0) * np.array(prob_u_new) * transition_matrix[data['old'].to_numpy(),data['new'].to_numpy()]
            prob_old = np.prod(np.array(list(conditional_probs_old.values())), axis = 0) * np.array(prob_u_old) * transition_matrix[data['new'].to_numpy(),data['old'].to_numpy()]

            # Compute acceptance ratios
            ratios = np.minimum(1, prob_new / prob_old)

            # Generate acceptance decisions
            acceptances = np.random.uniform(0, 1, size=len(ratios)) < ratios

            # Update accepted samples
            self._initial_sampling[U][acceptances] = data['new'].to_numpy()[acceptances]

            # Get posterior
            counts_u = [np.count_nonzero(self._initial_sampling[U] == u) for u in exo_f.domain[U]]
            beta = [int(a + c) for a, c in zip(self._alpha[U], counts_u)]

            # sample the theta and set it to the model
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

    def _get_precise_sampling(self, U,  data: pd.DataFrame):
        exo_f = self._model.factors[U]
        endo_f = self._endo_factors[U]

        endo_vars = {v: list(filter(lambda item: item != U, f.variables)) for v, f in endo_f.items()}

        feasible_values = np.prod(
            np.array([
                [f.R(**row).values for row in data[endo_vars[v]].to_dict(orient="records")]
                for v, f in endo_f.items()
            ]), axis=0
        )

        prob = np.array(exo_f.values) * feasible_values
        prob_standard = prob/prob.sum(axis = 1, keepdims=True)

        sampled_values = np.array([
            np.random.choice(exo_f.domain[U], p=prob_standard[i])
            for i in range(len(data))
        ])
        return sampled_values

    def _function_select(self, transition_select, perturbation_select):
        transition_dict = {"uniform": self._uniform_transition,
                           "random": self._random_transition}

        perturbation_dict = {"uniform": self._uniform_perturbation}

        self._transition_function = transition_dict[transition_select]
        self._perturbation_function = perturbation_dict[perturbation_select]


    # Methods to make transition or perturbation
    def _uniform_transition(self, U):
        num_states = len(self._model.factors[U].domain[U])
        return np.full((num_states, num_states), 1 / num_states)

    def _random_transition(self, U):
        num_states = len(self._model.factors[U].domain[U])
        param = np.repeat(0.1,num_states) # parameter of the dirichlet that samples transition matrix
        return np.array([dirichlet.rvs(param)[0] for _ in range(num_states)])

    def _uniform_perturbation(self, u, U, transition_matrix):
        return np.random.choice(self._model.factors[U].domain[U], p = transition_matrix[u])

    def _set_non_informative_alpha(self):
        self._alpha = {U: np.ones(len(self._model.domains[U])) for U in self._trainable_vars}

if __name__ == "__main__":
    import logging, sys

    log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'

    # logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format, datefmt='%Y%m%d_%H%M%S')


    # Nota: probar también con modelos semi-markovianos

    #m = StructuralCausalModel.read("./models/literature/pearl_small.bif")

    #m = StructuralCausalModel.read("./models/modelTest_SM.bif")
    #data = pd.read_csv("./models/literature/pearl_small.csv")
    #data = pd.read_csv("./models/modelTest_SM.csv")

    import time

    # Start the timer
    start_time = time.time()

    mhs = MetropolisHastingsSampling(m, transition_select = "random")
    mhs.run(data, max_iter=10000)

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
    for model_i in mhs.model_evolution:
         inf = CausalMultiInference([model_i])
         q = inf.prob_sufficiency("T","S", true_false_cause=(1,0), true_false_effect=(1,0))[0]
         Q.append(q)


    plt.hist(Q[10:], density=True)
    plt.xlim(0, 1)
    plt.show()

