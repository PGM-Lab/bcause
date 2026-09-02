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

    Each latent (exogenous) variable U is inferred with a Metropolis-Hastings
    chain that runs over every data row in parallel: each row holds a current
    guess for U, a new candidate is proposed from the states compatible with the
    observed endogenous values, and the candidate is accepted with the standard
    MH probability. The accepted assignments feed a Dirichlet posterior that
    yields the updated probability table for U.
    '''

    @property
    def acceptance_rate(self) -> Dict[str, float]:
        # Fraction of proposals accepted so far, per trainable variable
        # (0.0 when no proposal has been made yet to avoid division by zero).
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
        # Base model; fix_numeric_domains() makes domain values usable as array indices.
        self._prior_model = prior_model.fix_numeric_domains()
        self._trainable_vars = trainable_vars           # latent vars to infer (None -> auto-detect)
        self._alpha = alpha                             # Dirichlet hyperparameters (None -> flat prior)
        self._is_prior = True                           # first step samples from the prior, not MH
        self._self_avoiding = self_avoiding             # if True, never re-propose the current state

        # Per-variable lookup tables and state, all keyed by variable name U:
        self._val2idx = {}          # domain value  -> integer index
        self._idx2val = {}          # integer index -> domain value
        self._int_sampling_set = {} # context "key"  -> tuple of valid state indices
        self._valid_masks = {}      # boolean matrix (rows x states): which states are allowed per row
        self._int_exo_samples = {}  # current U assignment (as index) for every data row
        self._prob_tables = {}      # current probability table for U (numpy array)
        self._data = {}             # per-variable context data with a hashable "key" column
        self._endo_factors = {}     # children factors of U (the endogenous vars it feeds)
        self._endo_vars = {}        # for each child factor, its variables other than U

    def initialize(self, data: pd.DataFrame, **kwargs):
        # Work on a copy of the prior so the original model is left untouched.
        self._model = self._prior_model.copy()
        self._process_data(data.copy())

        # Acceptance bookkeeping per variable.
        self.acceptance_history = {U: [] for U in self._trainable_vars}
        self.total_proposals = {U: 0 for U in self._trainable_vars}
        self.accepted_proposals = {U: 0 for U in self._trainable_vars}

        # Precompute everything that stays constant across MH iterations.
        for U in self._trainable_vars:
            domain = self._model.factors[U].domain[U]
            # Two-way maps between domain values and integer indices.
            self._val2idx[U] = {val: i for i, val in enumerate(domain)}
            self._idx2val[U] = {i: val for i, val in enumerate(domain)}

            # Which U states are compatible with each observed context, as a
            # per-context set and as a boolean (rows x states) matrix.
            self._int_sampling_set[U] = self._hash_map_sampling(U)
            self._valid_masks[U] = self._build_valid_mask(U)

            # Current probability table for U and an initial assignment per row.
            self._prob_tables[U] = np.array(self._model.factors[U].values, dtype=float)
            self._int_exo_samples[U] = self._get_initial_exogenous(U)

    def _stop_learning(self) -> bool:
        # No convergence criterion: the chain runs for the requested max_iter.
        pass

    def _calculate_updated_factors(self, **kwargs) -> Dict[str, MultinomialFactor]:
        # One MH step for every trainable variable, returning their new factors.
        return {U: self._updated_factor(U) for U in self._trainable_vars}

    def _process_data(self, data: pd.DataFrame):
        # Latent variables have no observations: add them as all-NaN columns.
        missing_vars = [v for v in self._prior_model.variables if v not in data.columns]
        for v in missing_vars: data[v] = float("nan")

        # If not given explicitly, the trainable vars are the columns with NaNs.
        self._trainable_vars = self._trainable_vars or list(data.columns[data.isna().any()])

        # Sanity checks: each trainable var must be exogenous and fully unobserved.
        for v in self._trainable_vars:
            if not self._prior_model.is_exogenous(v):
                raise ValueError(f"Trainable variable {v} is not exogenous")
            if (~data[v].isna()).any():
                raise ValueError(f"Trainable variable {v} is not completely missing")

        # Discover the endogenous children of each latent var and their parents.
        self._get_involved_factors()

        # For each latent var, keep only the endogenous context columns it depends
        # on and add a hashable "key" tuple per row to group identical contexts.
        for v in self._trainable_vars:
            context_cols = list({x for sublist in self._endo_vars[v].values() for x in sublist})
            context_data = data[context_cols].copy()
            context_data["key"] = list(zip(*[context_data[c] for c in context_cols]))
            self._data[v] = context_data

    def _get_involved_factors(self):
        # Children factors of each latent U: the endogenous variables U feeds
        # into (its successors in the causal graph) and their factors.
        self._endo_factors = {
            trainable_var: {
                successor: self._model.factors[successor]
                for successor in self._model.graph.successors(trainable_var)
            }
            for trainable_var in self._trainable_vars
        }

        # For each such child factor, the variables other than U itself, i.e.
        # the endogenous context that (together with U) determines the child.
        self._endo_vars = {
            U: {v: list(filter(lambda item: item != U, f.variables)) for v, f in self._endo_factors[U].items()}
            for U in self._trainable_vars
        }

    def _hash_map_sampling(self, U: str):
        # For a single context row, find which U values are consistent with the
        # observed children. For each child k, R(...) restricts the factor to the
        # observed context and yields a 0/1 vector over U's states; a U value is
        # valid only if it is compatible with *every* child (set intersection).
        def _create_sampling_set(row, U, endo_f, endo_vars):
            booleans = {k: np.array(endo_f[k].R(**row[v].to_dict()).values, dtype=int).astype(bool) for k, v in
                        endo_vars.items()}
            return set.intersection(*[set(np.array(endo_f[k].domain[U])[v]) for k, v in booleans.items()])

        data = self._data[U]
        endo_f = self._endo_factors[U]
        endo_vars = self._endo_vars[U]
        # Compute the valid set only once per distinct context, not per row.
        unique_data = data.drop_duplicates(subset=[col for col in data.columns if col != 'key'])
        endo_cols = [col for col in data.columns if col != 'key']
        val_map = self._val2idx[U]

        # Valid U values for each unique context, converted to integer indices.
        sampling_set = unique_data[endo_cols].apply(lambda row: _create_sampling_set(row, U, endo_f, endo_vars), axis=1)
        int_sampling_series = sampling_set.map(lambda s: tuple({val_map[v] for v in s}))
        # Map: context key -> tuple of allowed state indices.
        return dict(zip(unique_data['key'], int_sampling_series))

    def _build_valid_mask(self, U: str) -> np.ndarray:
        # Expand the per-context valid sets into a boolean matrix with one row
        # per data row: mask[i, s] is True iff state s is allowed for row i.
        keys = self._data[U]["key"].values
        domain_size = len(self._model.factors[U].domain[U])
        mask = np.zeros((len(keys), domain_size), dtype=bool)
        sampling_map = self._int_sampling_set[U]

        for i, k in enumerate(keys):
            mask[i, list(sampling_map[k])] = True

        return mask

    def _vectorized_choice(self, p_matrix: np.ndarray) -> np.ndarray:
        # Draw one categorical sample per row via inverse-CDF sampling.
        # cumsum gives the CDF; dividing by the last column normalizes it.
        cum_p = p_matrix.cumsum(axis=1)
        # Safe normalization tolerating empty rows (zero sum -> left as zeros).
        cum_p = np.divide(cum_p, cum_p[:, -1:], out=np.zeros_like(cum_p), where=cum_p[:, -1:] > 0)
        # First state whose cumulative probability exceeds a uniform draw.
        rnd = np.random.rand(p_matrix.shape[0], 1)
        return (cum_p > rnd).argmax(axis=1)

    def _get_initial_exogenous(self, U: str) -> np.ndarray:
        # Initial assignment: sample each row from the prior table restricted to
        # its valid states (invalid states are zeroed out before normalizing).
        mask = self._valid_masks[U]
        probs = self._prob_tables[U] * mask
        p_matrix = probs / (probs.sum(axis=1, keepdims=True) + 1e-10)
        return self._vectorized_choice(p_matrix)

    def _updated_factor(self, U: str) -> MultinomialFactor:
        # Run one MH step for U across all data rows and return its new factor.
        m = self._model
        data = self._data[U]
        u_old = self._int_exo_samples[U]   # current assignment per row (indices)

        valid_mask = self._valid_masks[U]
        if self._is_prior:
            # Very first step: there is no chain state yet, so just draw the
            # probability table from the Dirichlet prior (default: flat prior).
            if self._alpha is None:
                self._set_non_informative_alpha()
            theta = dirichlet.rvs(self._alpha[U])[0]
            self._is_prior = False
        else:
            # Current probability table for U (from last iteration's update).
            exo_f = self._model.factors[U]
            theta_current = np.array(exo_f.values)
            n_samples = len(data)

            # --- Proposal step: pick a candidate state u_new for every row ---
            if self._self_avoiding:
                # Forbid re-proposing the current state by masking it out.
                proposal_mask = valid_mask.copy()
                row_indices = np.arange(n_samples)
                proposal_mask[row_indices, u_old] = False

                # Rows may be left with no alternative (only one valid state).
                options_count = proposal_mask.sum(axis=1)
                has_options = options_count > 0

                # Uniform proposal over the remaining valid states.
                p_matrix = np.zeros_like(proposal_mask, dtype=float)
                p_matrix[has_options] = proposal_mask[has_options].astype(float) / options_count[has_options, None]
                u_new_proposed = self._vectorized_choice(p_matrix)
                # Where no alternative exists, keep the current state.
                u_new = np.where(has_options, u_new_proposed, u_old)

            else:
                # Standard proposal: uniform over each row's valid states.
                options_count = valid_mask.sum(axis=1)
                p_matrix = valid_mask.astype(float) / options_count[:, None]
                u_new = self._vectorized_choice(p_matrix)

            # --- Acceptance step (Metropolis) ---
            # The proposal is symmetric over the valid set, so the acceptance
            # ratio reduces to the ratio of target probabilities, capped at 1.
            old_prob = theta_current[u_old]
            new_prob = theta_current[u_new]
            ratio = np.minimum(new_prob / (old_prob + 1e-10), 1.0)

            # Accept a proposal where the ratio beats a uniform(0,1) draw.
            flag = (ratio >= np.random.rand(n_samples)).astype(int)
            self._int_exo_samples[U] = np.where(flag == 1, u_new, u_old)

            # Track the acceptance rate for this step.
            n_accepted = flag.sum()
            self.accepted_proposals[U] += n_accepted
            self.total_proposals[U] += n_samples
            self.acceptance_history[U].append(n_accepted / n_samples if n_samples > 0 else 0)

            # --- Posterior update ---
            # Count how many rows landed on each U state, add the Dirichlet prior
            # to get the posterior parameters, and sample the new table theta.
            domain_size = len(exo_f.domain[U])
            counts_u = np.bincount(self._int_exo_samples[U], minlength=domain_size)
            beta = np.array(self._alpha[U]) + counts_u
            theta = dirichlet.rvs(beta)[0]

        # Wrap the sampled probabilities as U's updated conditional factor.
        return MultinomialFactor({U: m.domains[U]}, theta)

    def _set_non_informative_alpha(self):
        # Flat (uniform) Dirichlet prior: all-ones concentration per variable.
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

    # Load an SCM and its observational data (prefix columns to match var names).
    m = StructuralCausalModel.read(directory_path + "simple_nparents2_nzr04_zdr05_1.uai")
    data = pd.read_csv(directory_path + "simple_nparents2_nzr04_zdr05_1.csv",index_col=0).add_prefix('V')

    import time
    # Time the full MCMC run.
    start_time = time.time()
    mhs = MetropolisHastingsSampling(m,self_avoiding=False)
    # Run 10000 MH iterations over the endogenous observations.
    mhs.run(data[m.endogenous], max_iter=10000)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    print("Acceptance Rates:", mhs.acceptance_rate)

    # Aggregate the sampled models into a causal query (probability of necessity).
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

