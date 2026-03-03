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

        # --- NEW: PARALLEL TEMPERING SETUP ---
        self.n_chains = 4
        # Temperatures: 1.0 is the Cold Chain (Target). The others are Hot.
        self.temperatures = np.array([1.0, 2.0, 5.0, 10.0])
        self.iter_count = 0
        self._prob_tables = {}
        # -------------------------------------

        for U in self._trainable_vars:
            domain = self._model.factors[U].domain[U]
            self._val2idx[U] = {val: i for i, val in enumerate(domain)}
            self._idx2val[U] = {i: val for i, val in enumerate(domain)}
            self._int_sampling_set[U] = self._hash_map_sampling(U)

            # Initialize Probability Tables for ALL chains: shape (n_chains, domain_size)
            base_theta = np.array(self._model.factors[U].values, dtype=float)
            self._prob_tables[U] = np.tile(base_theta, (self.n_chains, 1))

            # Initialize exogenous samples for ALL chains: shape (n_chains, N)
            init_samples = self._get_initial_exogenous(U)
            self._int_exo_samples[U] = np.tile(init_samples, (self.n_chains, 1))

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
        probs = [self._prob_tables[U][0][u] for u in list(sampling_set)]
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
            self.iter_count += 1
            exo_f = self._model.factors[U]
            domain_size = len(exo_f.domain[U])
            keys = data["key"].values
            u_old = self._int_exo_samples[U]  # shape: (n_chains, N_rows)
            theta = self._prob_tables[U]  # shape: (n_chains, domain_size)
            # ==========================================
            # STEP 1: INDEPENDENT MH STEP FOR ALL CHAINS
            # ==========================================
            u_prop = np.zeros_like(u_old)
            # Propose independently to maintain diversity across chains
            for c in range(self.n_chains):
                u_prop[c] = [
                    random.choice([x for x in sampling_set[k] if x != v])
                    if len(sampling_set[k]) > 1
                    else next(iter(sampling_set[k]))  # Safely grabs the only element from a set
                    for k, v in zip(keys, u_old[c])
                ]
            # Calculate probabilities for all chains simultaneously
            old_prob = np.array([theta[c, u_old[c]] for c in range(self.n_chains)])
            new_prob = np.array([theta[c, u_prop[c]] for c in range(self.n_chains)])

            # The PT Magic: Flatten the acceptance ratio using (1/T)
            T = self.temperatures[:, None]  # shape (n_chains, 1) for broadcasting
            ratio = np.minimum((new_prob / old_prob) ** (1.0 / T), 1.0)
            accept = (np.random.rand(self.n_chains, len(data)) <= ratio)
            self._int_exo_samples[U] = np.where(accept, u_prop, u_old)

            n_proposal = len(data)
            n_accepted = accept[0].sum()

            self.accepted_proposals[U] += n_accepted
            self.total_proposals[U] += n_proposal

            current_rate = n_accepted / n_proposal if n_proposal > 0 else 0
            self.acceptance_history[U].append(current_rate)

            # ==========================================
            # STEP 2: REPLICA EXCHANGE (THE SWAP)
            # ==========================================

            # Attempt a swap every 10 iterations to let the chains explore first
            if self.iter_count % 20 == 0:
                # Pick a random chain and the one directly "hotter" than it
                c = np.random.randint(0, self.n_chains - 1)
                c_next = c + 1
                # Calculate the Unnormalized Log-Likelihood (Energy) of both chains
                # L = sum(log(theta[U_i]))
                L_c = np.sum(np.log(theta[c, self._int_exo_samples[U][c]] + 1e-10))
                L_next = np.sum(np.log(theta[c_next, self._int_exo_samples[U][c_next]] + 1e-10))

                # Swap acceptance probability
                # If the Hot chain found a better state, this pushes it down to the Cold chain
                delta_beta = (1.0 / self.temperatures[c]) - (1.0 / self.temperatures[c_next])
                swap_alpha = np.exp(min((L_next - L_c) * delta_beta, 0.0))  # min() prevents overflow

                if np.random.rand() < swap_alpha:
                    # SWAP configurations AND parameters between the two chains!
                    self._int_exo_samples[U][[c, c_next]] = self._int_exo_samples[U][[c_next, c]]
                    self._prob_tables[U][[c, c_next]] = self._prob_tables[U][[c_next, c]]

            # ==========================================
            # STEP 3: DIRICHLET UPDATE
            # ==========================================

            for c in range(self.n_chains):
                counts_u = np.bincount(self._int_exo_samples[U][c], minlength=domain_size)
                # Temper the counts so the hot chains don't get stuck in sharp Dirichlet peaks
                beta = np.array(self._alpha[U]) + (counts_u / self.temperatures[c])
                self._prob_tables[U][c] = dirichlet.rvs(beta)[0]

        f = MultinomialFactor({U: self._model.domains[U]}, self._prob_tables[U][0])
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

    m = StructuralCausalModel.read(directory_path + "simple_nparents2_nzr08_zdr10_3.uai")
    data = pd.read_csv(directory_path + "simple_nparents2_nzr08_zdr10_3.csv",index_col=0).add_prefix('V')

    import time
    # Start the timer
    start_time = time.time()
    mhs = MetropolisHastingsSampling(m)
    # mhs.initialize(data[m.endogenous])
    mhs.run(data[m.endogenous], max_iter=2000)

    # End the timer
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    print("Acceptance Rates:", mhs.acceptance_rate)

    inf_mh_1 = CausalMultiInference(mhs.model_evolution, outliers_removal=False)
    res_val = inf_mh_1.prob_necessity("V2", "V0", true_false_cause=(1, 0),
                                      true_false_effect=(1, 0))
    print("Result with outliers removal:", res_val)
    inf_mh_2 = CausalMultiInference(mhs.model_evolution, outliers_removal=True)
    res_val = inf_mh_2.prob_necessity("V2", "V0", true_false_cause=(1, 0),
                                      true_false_effect=(1, 0))
