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
from bcause.util import randomUtil
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


class GibbsSamplingCollapsed(IterativeParameterLearning):
    '''
     This class implements a method for running a single optimization of the exogenous variables in a SCM.
     '''

    def __init__(self, prior_model: StructuralCausalModel, trainable_vars: list = None, alpha: dict = None):
        self._prior_model = prior_model.fix_numeric_domains()
        self._trainable_vars = trainable_vars
        self._alpha = alpha

        self._val2idx = {}
        self._idx2val = {}
        self._int_sampling_set = {}
        self._int_exo_samples = {}
        self._prob_tables = {}


    def initialize(self, data: pd.DataFrame, **kwargs):
        self._model = self.prior_model.copy()
        self._process_data(data.copy())

        if self._alpha is None:
            self._set_non_informative_alpha()


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


    def uniform_choice(self,sampling_set):
        """
        Perturb the row based on the uniform distribution.
        """
        return random.choice(list(sampling_set))

    def _updated_factor(self, U) -> MultinomialFactor:
        m = self._model
        data = self._data[U]
        u_old = self._int_exo_samples[U]
        sampling_set = self._int_sampling_set[U]

        exo_f = self._model.factors[U]
        alpha = np.array(self._alpha[U])

        domain_size = len(exo_f.domain[U])
        counts_u = np.bincount(u_old, minlength=domain_size)

        u_new = u_old.copy()
        keys = data["key"].values

        for i, (k,v) in enumerate(zip(keys, u_old)):
            S = sampling_set[k]
            if len(S) > 1:
                counts_u[v] -= 1  # Remove the count of the old value

                # We compute the weights for the compatible set instead of the whole domain to avoid searching in the unfeasable values.
                weights = [counts_u[x] + alpha[x] for x in S]
                # Random choice normalize the weights, no need to do it
                new_v = random.choices(list(S), weights=weights, k=1)[0]

                counts_u[new_v] += 1  # Add the count of the new value
                u_new[i] = new_v
            else:
                u_new[i] = S[0]

        # Update the exogenous samples with the new values
        self._int_exo_samples[U] = u_new

        beta = counts_u + np.array(self._alpha[U])
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

    m = StructuralCausalModel.read("./models/literature/pearl_small.bif")
    #m2 = m.merge_exogenous("V","U")
    #m = StructuralCausalModel.read("./models/modelTest_SM.bif")
    data = pd.read_csv("./models/literature/pearl_small.csv")
    #data = pd.read_csv("./models/modelTest_SM.csv")
    # m = StructuralCausalModel.read("./models/g2_model_18.bif")
    # data = pd.read_csv("./models/g2_data_18.csv")

    directory_path = "/Users/antoniogonzalezalves/Documents/s23/"
    download_path = "/Users/antoniogonzalezalves/Documents/BenchMarkMH/"

    # m = StructuralCausalModel.read(directory_path + "simple_nparents2_nzr08_zdr10_3.uai")
    # data = pd.read_csv(directory_path + "simple_nparents2_nzr08_zdr10_3.csv",index_col=0).add_prefix('V')

    import time
    # Start the timer
    start_time = time.time()
    gsc = GibbsSamplingCollapsed(m)
    # mhs.initialize(data[m.endogenous])
    gsc.run(data[m.endogenous], max_iter=10000)

    # End the timer
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    qtrue = [0.3,1.0]
    import matplotlib.pyplot as plt
    Q = []
    for i,model_i in enumerate(gsc.model_evolution):
        # run the query
        inf = CausalMultiInference([model_i])
        Q.append(inf.prob_sufficiency("T", "S",true_false_cause=(1, 0),true_false_effect=(1, 0))[0])
        theta = model_i.factors["U"].values

        if i % 10 == 0:
            msg = f"{i}., current query = {Q[-1]}, , theta= {theta}, true interval = {qtrue}"
            if i>100:
                qapprox = [min(Q[100:]), max(Q[100:])]
                msg += f"approx interval = {qapprox}"
            print(msg)

        if i%100==0 and i>100:
            plt.hist(Q[100:],density=True)
            plt.xlim(0, 1)
            plt.show()

    # inf_mh_1 = CausalMultiInference(mhs.model_evolution, outliers_removal=False)
    # res_val = inf_mh_1.prob_necessity("V2", "V0", true_false_cause=(1, 0),
    #                                   true_false_effect=(1, 0))
    # print("Result with outliers removal:", res_val)
    # inf_mh_2 = CausalMultiInference(mhs.model_evolution, outliers_removal=True)
    # res_val = inf_mh_2.prob_necessity("V2", "V0", true_false_cause=(1, 0),
    #                                   true_false_effect=(1, 0))
    # print("Result with outliers removal:", res_val)
    # inf_mh_2 = CausalMultiInference(mhs.model_evolution[100:])
    # res_val2 = inf_mh_2.prob_sufficiency("V2", "V0", true_false_cause=(1, 0),
    #                                     true_false_effect=(1, 0))
    # inf_mh_3 = CausalMultiInference(mhs.model_evolution[int(10000/5):])
    # res_val3 = inf_mh_3.prob_sufficiency("V2", "V0", true_false_cause=(1, 0),
    #                                     true_false_effect=(1, 0))

