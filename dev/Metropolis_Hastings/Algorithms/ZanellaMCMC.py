"""
Compute the PMF of exogenous variables given data for endogenous variable using
Metropolis_Hastings

improvements:
 - Accept a perturbation and a transition function as inputs for the method
 - Change dictionary approach to preloaded tables and SQL joins for conditional probability?
 - Sampling only possible values?
"""

from collections import defaultdict
import pandas as pd
import numpy as np
from numpy.array_api import astype
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

    @property
    def acceptance_rate(self) -> Dict[str, float]:
        # FIXED: Changed self.trainable_vars to self._trainable_vars
        return {U: (self.accepted_proposals[U] / self.total_proposals[U]) if self.total_proposals[U] > 0 else 0.0
                for U in self._trainable_vars}

    def __init__(self, prior_model: StructuralCausalModel, trainable_vars: list = None, alpha: dict = None):
        self._prior_model = prior_model.fix_numeric_domains()
        self._trainable_vars = trainable_vars
        self._alpha = alpha
        self._is_prior = True

        self._val2idx = {}
        self._idx2val = {}
        self._int_sampling_set = {}
        self._int_exo_samples = {}
        self._prob_tables = {}

    def initialize(self, data: pd.DataFrame, **kwargs):
        self._model = self.prior_model.copy()
        self._process_data(data.copy())

        self.acceptance_history = {U: [] for U in self._trainable_vars}
        self.total_proposals = {U: 0 for U in self._trainable_vars}
        self.accepted_proposals = {U: 0 for U in self._trainable_vars}

        for U in self._trainable_vars:
            domain = self._model.factors[U].domain[U]
            self._val2idx[U] = {val: i for i, val in enumerate(domain)}
            self._idx2val[U] = {i: val for i, val in enumerate(domain)}
            # Create the sampling sets for each unique combination of endogenous variables from data.
            self._int_sampling_set[U] = self._hash_map_sampling(U)

            # Initialize Probability Tables (as numpy arrays)
            self._prob_tables[U] = np.array(self._model.factors[U].values, dtype=float)
            self._int_exo_samples[U] = self._get_initial_exogenous(U)

    def _stop_learning(self) -> bool:
        pass

    def _calculate_updated_factors(self, **kwargs) -> dict[MultinomialFactor]:
        return {U: self._updated_factor(U) for U in self.trainable_vars}

    def _hash_map_sampling(self,U):

        def _create_sampling_set(row, U, endo_f, endo_vars):
            # Get feasible values for the endo variables using R function
            booleans = {k: np.array(endo_f[k].R(**row[v].to_dict()).values, dtype=int).astype(bool) for k, v in
                        endo_vars.items()}
            return set.intersection(*[set(np.array(endo_f[k].domain[U])[v]) for k, v in booleans.items()])

        data = self._data[U]
        endo_f = self._endo_factors[U]
        endo_vars = self._endo_vars[U]
        unique_data = data.drop_duplicates()
        endo_cols = [col for col in data.columns if col != 'key']
        val_map = self._val2idx[U]

        sampling_set = unique_data[endo_cols].apply(lambda row: _create_sampling_set(row, U, endo_f, endo_vars), axis=1)
        # int_sampling_series = sampling_set.map(lambda s: {val_map[v] for v in s})
        int_sampling_series = sampling_set.map(lambda s: tuple({val_map[v] for v in s}))
        return dict(zip(unique_data['key'], int_sampling_series))


    def _get_initial_exogenous(self,U):
        # Get the sampling set for the trainable variables
        keys = self._data[U]["key"].values
        sampling_map = self._int_sampling_set[U]

        return np.array([self.choice_with_exo_weights(U, sampling_map[k]) for k in keys])


    def choice_with_exo_weights(self, U, sampling_set):
        """
        This function controls the Initial U sampling
        Now: it chooses randomly from the sampling set
        """
        probs = [self._prob_tables[U][u] for u in list(sampling_set)]
        return np.random.choice(list(sampling_set), p=probs/np.sum(probs))


    def uniform_choice(self,set):
        """
        Perturbate the row based on the uniform distribution.
        """
        return random.choice(list(set))

    def _updated_factor(self, U) -> MultinomialFactor:
        m = self._model
        data = self._data[U]
        u_old = self._int_exo_samples[U]
        sampling_set = self._int_sampling_set[U]

        if self._is_prior == True:
            if self._alpha is None:
                self._set_non_informative_alpha()
            theta = dirichlet.rvs(self._alpha[U])[0]
            self._is_prior = False

        else:

            endo_f = self._endo_factors[U]
            exo_f = self._model.factors[U]

            # 1. Calculate the Zanella Barker Weights for all states
            theta_current = np.array(exo_f.values)
            # Add a microscopic epsilon to prevent division-by-zero if a probability is exactly 0.0
            w = np.sqrt(theta_current) + 1e-10
            keys = data["key"].values
            n_samples = len(keys)

            u_new = np.zeros(n_samples, dtype=int)
            Q_forward = np.ones(n_samples, dtype=float)
            Q_backward = np.ones(n_samples, dtype=float)

            # 2. Zanella Informed Proposal Loop
            for i, (k, v) in enumerate(zip(keys, u_old)):
                S = sampling_set[k]
                if len(S) > 1:
                    # Available moves (excluding current state v)
                    A = [x for x in S if x != v]
                    weights_A = [w[x] for x in A]
                    # Instantly pick a smart candidate proportional to its weight
                    x_prop = random.choices(A, weights=weights_A, k=1)[0]
                    u_new[i] = x_prop
                    # Calculate Q ratios for Detailed Balance
                    Z_total = sum(w[y] for y in S)
                    # Probability of proposing x_prop from v
                    Q_forward[i] = w[x_prop] / (Z_total - w[v])
                    # Probability of proposing v back from x_prop
                    Q_backward[i] = w[v] / (Z_total - w[x_prop])
                else:
                    u_new[i] = S[0]
                    # Q_forward and Q_backward remain 1.0

            # 3. Metropolis-Hastings Acceptance
            old_prob = theta_current[u_old]
            new_prob = theta_current[u_new]
            target_ratio = new_prob / old_prob
            proposal_ratio = Q_backward / Q_forward

            # Final balanced acceptance ratio
            ratio = np.minimum(target_ratio * proposal_ratio, 1.0)

            # Vectorized Accept/Reject
            flag = (ratio >= np.random.rand(n_samples)).astype(int)
            self._int_exo_samples[U] = np.where(flag == 1, u_new, u_old)

            # ---> ADDED: TRACKING ACCEPTANCE RATE <---
            n_proposal = n_samples
            n_accepted = flag.sum()

            self.accepted_proposals[U] += n_accepted
            self.total_proposals[U] += n_proposal

            current_rate = n_accepted / n_proposal if n_proposal > 0 else 0
            self.acceptance_history[U].append(current_rate)
            # -----------------------------------------

            # 4. Standard Dirichlet Update
            domain_size = len(exo_f.domain[U])
            counts_u = np.bincount(self._int_exo_samples[U], minlength=domain_size)
            beta = np.array(self._alpha[U]) + counts_u
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

        # get the involve factors
        self._get_involved_factors()

        # prepare the data
        self._data = {}
        for v in self._trainable_vars:
            context_cols = list({x for sublist in self._endo_vars[v].values() for x in sublist})
            context_data = data[context_cols].copy()
            context_data["key"] = list(zip(*[context_data[c] for c in context_cols]))
            self._data[v] = context_data

        # save the dataset
        # self._data = data

    def _get_involved_factors(self):
        endo_factors = {
            trainable_var: { successor:
                self._model.factors[successor]
                for successor in self._model.graph.successors(trainable_var)
            }
            for trainable_var in self._trainable_vars
        }
        self._endo_factors = endo_factors

        self._endo_vars = {
            U: {v: list(filter(lambda item: item != U, f.variables)) for v, f in self._endo_factors[U].items()}
            for U in self._trainable_vars
        }


    def _set_non_informative_alpha(self):
        self._alpha = {U: np.ones(len(self._model.domains[U])) for U in self._trainable_vars}

if __name__ == "__main__":
    import logging, sys

    log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'

    # logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format, datefmt='%Y%m%d_%H%M%S')


    # Nota: probar también con modelos semi-markovianos

    # m = StructuralCausalModel.read("./models/literature/pearl_small.bif")
    #m2 = m.merge_exogenous("V","U")
    #m = StructuralCausalModel.read("./models/modelTest_SM.bif")
    # data = pd.read_csv("./models/literature/pearl_small.csv")
    #data = pd.read_csv("./models/modelTest_SM.csv")

    directory_path = "/Users/antoniogonzalezalves/Documents/s23/"
    download_path = "/Users/antoniogonzalezalves/Documents/BenchMarkMH/"

    # Replace these paths with yours as needed
    m = StructuralCausalModel.read(directory_path + "simple_nparents2_nzr08_zdr10_3.uai")
    data = pd.read_csv(directory_path + "simple_nparents2_nzr08_zdr10_3.csv",index_col=0).add_prefix('V')



    import time

    # Start the timer
    start_time = time.time()
    # Initialize the Metropolis_Hastings sampling transition function "random" or "uniform"
    mhs = MetropolisHastingsSampling(m)
    # mhs.initialize(data[m.endogenous])
    mhs.run(data[m.endogenous], max_iter=10000)

    # End the timer
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    # ---> ADDED: Check the acceptance rate when finished! <---
    print(f"Final Acceptance Rates: {mhs.acceptance_rate}")

    import matplotlib.pyplot as plt

    U_store = np.empty((0,64))
    V_store = np.empty([0,2])

    Q = []

    # print the model evolution
    for model_i in mhs.model_evolution[100:]:
         inf = CausalMultiInference([model_i])
         q = inf.prob_sufficiency("V1","V2", true_false_cause=(1,0), true_false_effect=(1,0))[0]
         Q.append(q)


    plt.hist(Q, density=True)
    plt.xlim(0, 1)
    plt.show()