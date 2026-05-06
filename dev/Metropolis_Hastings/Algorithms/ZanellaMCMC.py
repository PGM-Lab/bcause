import pandas as pd
import numpy as np
from scipy.stats import dirichlet
from typing import Dict, List, Set, Any

from bcause.factors import MultinomialFactor
from bcause.learning.parameter import IterativeParameterLearning
from bcause.models.cmodel import StructuralCausalModel


class MetropolisHastingsSampling(IterativeParameterLearning):
    """
    This class implements a method for running a single optimization
    of the exogenous variables in an SCM using Zanella/Barker MCMC.
    """

    def __init__(self, prior_model: StructuralCausalModel, trainable_vars: list = None, alpha: dict = None,
                 method: str = 'sqrt'):
        self._prior_model = prior_model.fix_numeric_domains()
        self._trainable_vars = trainable_vars
        self._alpha = alpha
        self._is_prior = True
        self._method = method

        self._val2idx = {}
        self._idx2val = {}
        self._int_sampling_set = {}
        self._valid_masks = {}
        self._int_exo_samples = {}
        self._prob_tables = {}
        self._data = {}
        self._endo_factors = {}
        self._endo_vars = {}

    @property
    def acceptance_rate(self) -> Dict[str, float]:
        return {
            U: (self.accepted_proposals[U] / self.total_proposals[U]) if self.total_proposals[U] > 0 else 0.0
            for U in self._trainable_vars
        }

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

            # Crear mapeos y máscaras matriciales para vectorización
            self._int_sampling_set[U] = self._hash_map_sampling(U)
            self._valid_masks[U] = self._build_valid_mask(U)

            # Inicializar tablas de probabilidad
            self._prob_tables[U] = np.array(self._model.factors[U].values, dtype=float)
            self._int_exo_samples[U] = self._get_initial_exogenous(U)

    def _stop_learning(self) -> bool:
        return False  # Implementar lógica de parada si es necesario

    def _calculate_updated_factors(self, **kwargs) -> Dict[str, MultinomialFactor]:
        return {U: self._updated_factor(U) for U in self._trainable_vars}

    def _process_data(self, data: pd.DataFrame):
        missing_vars = [v for v in self._prior_model.variables if v not in data.columns]
        for v in missing_vars:
            data[v] = float("nan")

        self._trainable_vars = self._trainable_vars or list(data.columns[data.isna().any()])
        print(f"Trainable variables: {self._trainable_vars}")

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

    def _hash_map_sampling(self, U: str) -> Dict[tuple, tuple]:
        def _create_sampling_set(row, U, endo_f, endo_vars):
            booleans = {
                k: np.array(endo_f[k].R(**row[v].to_dict()).values, dtype=int).astype(bool)
                for k, v in endo_vars.items()
            }
            return set.intersection(*[set(np.array(endo_f[k].domain[U])[v]) for k, v in booleans.items()])

        data = self._data[U]
        unique_data = data.drop_duplicates(subset=[col for col in data.columns if col != 'key'])
        val_map = self._val2idx[U]

        endo_f = self._endo_factors[U]
        endo_vars = self._endo_vars[U]
        endo_cols = [col for col in data.columns if col != 'key']

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
        cum_p /= cum_p[:, -1:]
        rnd = np.random.rand(p_matrix.shape[0], 1)
        return (cum_p > rnd).argmax(axis=1)

    def _get_initial_exogenous(self, U: str) -> np.ndarray:
        mask = self._valid_masks[U]
        probs = self._prob_tables[U] * mask
        p_matrix = probs / (probs.sum(axis=1, keepdims=True) + 1e-10)
        return self._vectorized_choice(p_matrix)

    def _updated_factor(self, U: str) -> MultinomialFactor:
        m = self._model
        u_old = self._int_exo_samples[U]
        valid_mask = self._valid_masks[U]
        method = self._method

        if self._is_prior:
            if self._alpha is None:
                self._alpha = {u: np.ones(len(self._model.domains[u])) for u in self._trainable_vars}
            theta = dirichlet.rvs(self._alpha[U])[0]
            self._is_prior = False
        else:
            exo_f = self._model.factors[U]
            theta_current = np.array(exo_f.values)
            n_samples = len(u_old)

            if method == 'sqrt':
                w = np.sqrt(theta_current)
                weights = w * valid_mask
                p_matrix = weights / (weights.sum(axis=1, keepdims=True) + 1e-10)

                u_new = self._vectorized_choice(p_matrix)

                # MH Acceptance
                target_ratio = theta_current[u_new] / (theta_current[u_old] + 1e-10)
                proposal_ratio = w[u_old] / (w[u_new] + 1e-10)

                ratio = np.minimum(target_ratio * proposal_ratio, 1.0)
                flag = (ratio >= np.random.rand(n_samples)).astype(int)
                self._int_exo_samples[U] = np.where(flag == 1, u_new, u_old)

            elif method == 'barker':
                theta_2d = theta_current.reshape(1, -1)
                W_matrix = theta_2d / (theta_current.reshape(-1, 1) + theta_2d + 1e-10)

                # Forward proposals
                weights_fwd = W_matrix[u_old, :] * valid_mask
                Z_fwd = weights_fwd.sum(axis=1)
                p_fwd = weights_fwd / (Z_fwd[:, None] + 1e-10)

                u_new = self._vectorized_choice(p_fwd)

                # Backward proposals
                weights_bck = W_matrix[u_new, :] * valid_mask
                Z_bck = weights_bck.sum(axis=1)

                # Q probabilities
                Q_forward = W_matrix[u_old, u_new] / (Z_fwd + 1e-10)
                Q_backward = W_matrix[u_new, u_old] / (Z_bck + 1e-10)

                # MH Acceptance
                target_ratio = theta_current[u_new] / (theta_current[u_old] + 1e-10)
                proposal_ratio = Q_backward / (Q_forward + 1e-10)

                ratio = np.minimum(target_ratio * proposal_ratio, 1.0)
                flag = (ratio >= np.random.rand(n_samples)).astype(int)
                self._int_exo_samples[U] = np.where(flag == 1, u_new, u_old)

            else:
                raise ValueError(f"Unknown method: {method}")

            # Seguimiento de la tasa de aceptación
            n_accepted = flag.sum()
            self.accepted_proposals[U] += n_accepted
            self.total_proposals[U] += n_samples
            self.acceptance_history[U].append(n_accepted / n_samples if n_samples > 0 else 0)

            # Actualización Dirichlet
            counts_u = np.bincount(self._int_exo_samples[U], minlength=len(exo_f.domain[U]))
            beta = np.array(self._alpha[U]) + counts_u
            theta = dirichlet.rvs(beta)[0]

        return MultinomialFactor({U: m.domains[U]}, theta)


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
    m = StructuralCausalModel.read(directory_path + "simple_nparents2_nzr04_zdr05_1.uai")
    data = pd.read_csv(directory_path + "simple_nparents2_nzr04_zdr05_1.csv",index_col=0).add_prefix('V')



    import time

    # Start the timer
    start_time = time.time()
    # Initialize the Metropolis_Hastings sampling transition function "random" or "uniform"
    mhs = MetropolisHastingsSampling(m, method='sqrt')
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

    inf = CausalMultiInference(mhs.model_evolution[200:], outliers_removal=True)
    print(inf.prob_sufficiency("V1","V0", true_false_cause=(1,0), true_false_effect=(1,0))[0])

    # print the model evolution
    # for model_i in mhs.model_evolution[100:]:
    #      inf = CausalMultiInference([model_i])
    #      q = inf.prob_sufficiency("V0","V1", true_false_cause=(1,0), true_false_effect=(1,0))[0]
    #      Q.append(q)


    # plt.hist(Q, density=True)
    # plt.xlim(0, 1)
    # plt.show()