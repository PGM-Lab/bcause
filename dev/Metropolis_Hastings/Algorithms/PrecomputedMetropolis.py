"""
Compute the PMF of exogenous variables given data for endogenous variable using
Metropolis_Hastings with precomputation

Incluye un parámetro opcional para alternar entre muestreo estándar y "Self-Avoiding".
"""

import pandas as pd
import numpy as np
from scipy.stats import dirichlet
from typing import Dict, List, Tuple, Any

# Dependencias de bcause
import bcause as bc
from bcause.factors import MultinomialFactor
from bcause.inference.causal.multi import CausalMultiInference
from bcause.learning.parameter import IterativeParameterLearning
from bcause.models.cmodel import StructuralCausalModel

class MetropolisHastingsSampling(IterativeParameterLearning):
    '''
    This class implements a method for running a single optimization
    of the exogenous variables in an SCM.
    '''

    @property
    def acceptance_rate(self) -> Dict[str, float]:
        return {U: (self.accepted_proposals[U] / self.total_proposals[U]) if self.total_proposals[U] > 0 else 0.0
                for U in self._trainable_vars}

    def __init__(self, prior_model: StructuralCausalModel, trainable_vars: list = None, alpha: dict = None, self_avoiding: bool = False):
        """
        Args:
            prior_model: StructuralCausalModel base.
            trainable_vars: Lista de variables a optimizar.
            alpha: Diccionario de hiperparámetros de Dirichlet.
            self_avoiding: Si es True, el algoritmo nunca propone el estado actual (a menos que sea la única opción).
        """
        self._prior_model = prior_model.fix_numeric_domains()
        self._trainable_vars = trainable_vars
        self._alpha = alpha
        self._is_prior = True
        self._self_avoiding = self_avoiding  # <-- NUEVO PARÁMETRO

        self._val2idx = {}
        self._idx2val = {}
        self._int_sampling_set = {}
        self._valid_masks = {}
        self._int_exo_samples = {}
        self._prob_tables = {}
        self._data = {}
        self._endo_factors = {}
        self._endo_vars = {}

    def initialize(self, data: pd.DataFrame, **kwargs):
        self._model = self._prior_model.copy()
        self._process_data(data.copy())

        self.acceptance_history = {U: [] for U in self._trainable_vars}
        self.total_proposals = {U: 0 for U in self._trainable_vars}
        self.accepted_proposals = {U: 0 for U in self._trainable_vars}

        for U in self._trainable_vars:
            domain = self._model.factors[U].domain[U]
            self._val2idx[U] = {val: i for i, val in enumerate(domain)}
            self._idx2val[U] = {i: val for i, val in enumerate(domain)}

            self._int_sampling_set[U] = self._hash_map_sampling(U)
            self._valid_masks[U] = self._build_valid_mask(U)

            self._prob_tables[U] = np.array(self._model.factors[U].values, dtype=float)
            self._int_exo_samples[U] = self._get_initial_exogenous(U)

    def _stop_learning(self) -> bool:
        pass

    def _calculate_updated_factors(self, **kwargs) -> Dict[str, MultinomialFactor]:
        return {U: self._updated_factor(U) for U in self._trainable_vars}

    def _process_data(self, data: pd.DataFrame):
        missing_vars = [v for v in self._prior_model.variables if v not in data.columns]
        for v in missing_vars: data[v] = float("nan")

        self._trainable_vars = self._trainable_vars or list(data.columns[data.isna().any()])

        for v in self._trainable_vars:
            if not self._prior_model.is_exogenous(v):
                raise ValueError(f"Trainable variable {v} is not exogenous")
            if (~data[v].isna()).any():
                raise ValueError(f"Trainable variable {v} is not completely missing")

        self._get_involved_factors()

        for v in self._trainable_vars:
            context_cols = list({x for sublist in self._endo_vars[v].values() for x in sublist})
            context_data = data[context_cols].copy()
            context_data["key"] = list(zip(*[context_data[c] for c in context_cols]))
            self._data[v] = context_data

    def _get_involved_factors(self):
        self._endo_factors = {
            trainable_var: {
                successor: self._model.factors[successor]
                for successor in self._model.graph.successors(trainable_var)
            }
            for trainable_var in self._trainable_vars
        }

        self._endo_vars = {
            U: {v: list(filter(lambda item: item != U, f.variables)) for v, f in self._endo_factors[U].items()}
            for U in self._trainable_vars
        }

    def _hash_map_sampling(self, U: str):
        def _create_sampling_set(row, U, endo_f, endo_vars):
            booleans = {k: np.array(endo_f[k].R(**row[v].to_dict()).values, dtype=int).astype(bool) for k, v in
                        endo_vars.items()}
            return set.intersection(*[set(np.array(endo_f[k].domain[U])[v]) for k, v in booleans.items()])

        data = self._data[U]
        endo_f = self._endo_factors[U]
        endo_vars = self._endo_vars[U]
        unique_data = data.drop_duplicates(subset=[col for col in data.columns if col != 'key'])
        endo_cols = [col for col in data.columns if col != 'key']
        val_map = self._val2idx[U]

        sampling_set = unique_data[endo_cols].apply(lambda row: _create_sampling_set(row, U, endo_f, endo_vars), axis=1)
        int_sampling_series = sampling_set.map(lambda s: tuple({val_map[v] for v in s}))
        return dict(zip(unique_data['key'], int_sampling_series))

    def _build_valid_mask(self, U: str) -> np.ndarray:
        keys = self._data[U]["key"].values
        domain_size = len(self._model.factors[U].domain[U])
        mask = np.zeros((len(keys), domain_size), dtype=bool)
        sampling_map = self._int_sampling_set[U]

        for i, k in enumerate(keys):
            mask[i, list(sampling_map[k])] = True

        return mask

    def _vectorized_choice(self, p_matrix: np.ndarray) -> np.ndarray:
        cum_p = p_matrix.cumsum(axis=1)
        # Normalizamos de manera segura tolerando filas de p_matrix vacías (0 sum)
        cum_p = np.divide(cum_p, cum_p[:, -1:], out=np.zeros_like(cum_p), where=cum_p[:, -1:] > 0)
        rnd = np.random.rand(p_matrix.shape[0], 1)
        return (cum_p > rnd).argmax(axis=1)

    def _get_initial_exogenous(self, U: str) -> np.ndarray:
        mask = self._valid_masks[U]
        probs = self._prob_tables[U] * mask
        p_matrix = probs / (probs.sum(axis=1, keepdims=True) + 1e-10)
        return self._vectorized_choice(p_matrix)

    def _updated_factor(self, U: str) -> MultinomialFactor:
        m = self._model
        data = self._data[U]
        u_old = self._int_exo_samples[U]

        valid_mask = self._valid_masks[U]
        if self._is_prior:
            if self._alpha is None:
                self._set_non_informative_alpha()
            theta = dirichlet.rvs(self._alpha[U])[0]
            self._is_prior = False
        else:
            exo_f = self._model.factors[U]
            theta_current = np.array(exo_f.values)
            n_samples = len(data)
            if self._self_avoiding:
                proposal_mask = valid_mask.copy()
                row_indices = np.arange(n_samples)
                proposal_mask[row_indices, u_old] = False

                options_count = proposal_mask.sum(axis=1)
                has_options = options_count > 0

                p_matrix = np.zeros_like(proposal_mask, dtype=float)
                p_matrix[has_options] = proposal_mask[has_options].astype(float) / options_count[has_options, None]
                u_new_proposed = self._vectorized_choice(p_matrix)
                u_new = np.where(has_options, u_new_proposed, u_old)

            else:
                options_count = valid_mask.sum(axis=1)
                p_matrix = valid_mask.astype(float) / options_count[:, None]
                u_new = self._vectorized_choice(p_matrix)

            old_prob = theta_current[u_old]
            new_prob = theta_current[u_new]

            ratio = np.minimum(new_prob / (old_prob + 1e-10), 1.0)
            flag = (ratio >= np.random.rand(n_samples)).astype(int)

            self._int_exo_samples[U] = np.where(flag == 1, u_new, u_old)

            # Tracking de Acceptance rate
            n_accepted = flag.sum()
            self.accepted_proposals[U] += n_accepted
            self.total_proposals[U] += n_samples
            self.acceptance_history[U].append(n_accepted / n_samples if n_samples > 0 else 0)

            # Actualización Dirichlet
            domain_size = len(exo_f.domain[U])
            counts_u = np.bincount(self._int_exo_samples[U], minlength=domain_size)
            beta = np.array(self._alpha[U]) + counts_u
            theta = dirichlet.rvs(beta)[0]

        return MultinomialFactor({U: m.domains[U]}, theta)

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
    # m = StructuralCausalModel.read("./models/g2_model_18.bif")
    # data = pd.read_csv("./models/g2_data_18.csv")

    directory_path = "/Users/antoniogonzalezalves/Documents/s23/"
    download_path = "/Users/antoniogonzalezalves/Documents/BenchMarkMH/"

    m = StructuralCausalModel.read(directory_path + "simple_nparents2_nzr04_zdr05_1.uai")
    data = pd.read_csv(directory_path + "simple_nparents2_nzr04_zdr05_1.csv",index_col=0).add_prefix('V')

    import time
    # Start the timer
    start_time = time.time()
    mhs = MetropolisHastingsSampling(m,self_avoiding=False)
    # mhs.initialize(data[m.endogenous])
    mhs.run(data[m.endogenous], max_iter=10000)

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
    print("Result with outliers removal:", res_val)
    # inf_mh_2 = CausalMultiInference(mhs.model_evolution[100:])
    # res_val2 = inf_mh_2.prob_sufficiency("V2", "V0", true_false_cause=(1, 0),
    #                                     true_false_effect=(1, 0))
    # inf_mh_3 = CausalMultiInference(mhs.model_evolution[int(10000/5):])
    # res_val3 = inf_mh_3.prob_sufficiency("V2", "V0", true_false_cause=(1, 0),
    #                                     true_false_effect=(1, 0))

