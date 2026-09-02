"""
Compute the PMF of exogenous variables given data for endogenous variables
using a locally-balanced Metropolis-Hastings sampler (Zanella, 2020).

Two balancing functions are supported:
 - ``sqrt``:   g(t) = sqrt(t), the globally-balanced proposal of Zanella.
 - ``barker``: g(t) = t / (1 + t), the Barker proposal.

Both run over every data row in parallel: each row proposes a candidate state
for its latent variable weighted towards the more probable states, and the
Metropolis-Hastings correction restores the exact target distribution.
"""

import pandas as pd
import numpy as np
from scipy.stats import dirichlet
from typing import Dict

from bcause.factors import MultinomialFactor
from bcause.inference.causal.multi import CausalMultiInference
from bcause.learning.parameter import IterativeParameterLearning
from bcause.models.cmodel import StructuralCausalModel


class MetropolisHastingsSampling(IterativeParameterLearning):
    """
    Run a single optimization of the exogenous variables of an SCM using a
    locally-balanced (Zanella/Barker) Metropolis-Hastings chain.

    Each latent (exogenous) variable U is inferred with an MH chain that runs
    over every data row in parallel. Unlike the plain uniform proposal, the
    candidate state is drawn with a probability proportional to a balancing
    function of the current table theta, which biases moves towards higher
    probability states; the MH acceptance step corrects for the asymmetry of
    that proposal. The accepted assignments feed a Dirichlet posterior that
    yields the updated probability table for U.
    """

    @property
    def acceptance_rate(self) -> Dict[str, float]:
        # Fraction of proposals accepted so far, per trainable variable
        # (0.0 when no proposal has been made yet to avoid division by zero).
        return {
            U: (self.accepted_proposals[U] / self.total_proposals[U]) if self.total_proposals[U] > 0 else 0.0
            for U in self._trainable_vars
        }

    def __init__(self, prior_model: StructuralCausalModel, trainable_vars: list = None, alpha: dict = None,
                 method: str = 'barker'):
        """
        Args:
            prior_model: base StructuralCausalModel.
            trainable_vars: latent variables to optimize (None -> auto-detect
                the fully unobserved exogenous columns).
            alpha: Dirichlet hyperparameters per variable (None -> flat prior).
            method: balancing function, either ``'sqrt'`` or ``'barker'``.
        """
        # Base model; fix_numeric_domains() makes domain values usable as array indices.
        self._prior_model = prior_model.fix_numeric_domains()
        self._trainable_vars = trainable_vars           # latent vars to infer (None -> auto-detect)
        self._alpha = alpha                             # Dirichlet hyperparameters (None -> flat prior)
        self._is_prior = True                           # first step samples from the prior, not MH
        self._method = method                           # balancing function ('sqrt' or 'barker')

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
        """Precompute the static structures and the initial chain state."""
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
        return False

    def _calculate_updated_factors(self, **kwargs) -> Dict[str, MultinomialFactor]:
        # One MH step for every trainable variable, returning their new factors.
        return {U: self._updated_factor(U) for U in self._trainable_vars}

    def _process_data(self, data: pd.DataFrame):
        """Attach latent columns, validate trainable vars and build per-U context."""
        # Latent variables have no observations: add them as all-NaN columns.
        missing_vars = [v for v in self._prior_model.variables if v not in data.columns]
        for v in missing_vars:
            data[v] = float("nan")

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
        """Find each latent U's endogenous children factors and their context vars."""
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

    def _hash_map_sampling(self, U: str) -> Dict[tuple, tuple]:
        """Map each distinct observed context to the tuple of valid U state indices."""
        # For a single context row, find which U values are consistent with the
        # observed children. For each child k, R(...) restricts the factor to the
        # observed context and yields a 0/1 vector over U's states; a U value is
        # valid only if it is compatible with *every* child (set intersection).
        def _create_sampling_set(row, U, endo_f, endo_vars):
            booleans = {
                k: np.array(endo_f[k].R(**row[v].to_dict()).values, dtype=int).astype(bool)
                for k, v in endo_vars.items()
            }
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
        """Expand the per-context valid sets into a boolean (rows x states) matrix."""
        keys = self._data[U]["key"].values
        domain_size = len(self._model.factors[U].domain[U])
        mask = np.zeros((len(keys), domain_size), dtype=bool)
        sampling_map = self._int_sampling_set[U]

        for i, k in enumerate(keys):
            mask[i, list(sampling_map[k])] = True

        return mask

    def _vectorized_choice(self, p_matrix: np.ndarray) -> np.ndarray:
        """Draw one categorical sample per row via inverse-CDF sampling."""
        # cumsum gives the CDF; dividing by the last column normalizes it.
        cum_p = p_matrix.cumsum(axis=1)
        # Safe normalization tolerating empty rows (zero sum -> left as zeros).
        cum_p = np.divide(cum_p, cum_p[:, -1:], out=np.zeros_like(cum_p), where=cum_p[:, -1:] > 0)
        # First state whose cumulative probability exceeds a uniform draw.
        rnd = np.random.rand(p_matrix.shape[0], 1)
        return (cum_p > rnd).argmax(axis=1)

    def _get_initial_exogenous(self, U: str) -> np.ndarray:
        """Draw an initial per-row assignment from the prior restricted to valid states."""
        mask = self._valid_masks[U]
        probs = self._prob_tables[U] * mask
        p_matrix = probs / (probs.sum(axis=1, keepdims=True) + 1e-10)
        return self._vectorized_choice(p_matrix)

    def _updated_factor(self, U: str) -> MultinomialFactor:
        """Run one locally-balanced MH step for U and return its new factor."""
        m = self._model
        u_old = self._int_exo_samples[U]   # current assignment per row (indices)
        valid_mask = self._valid_masks[U]
        method = self._method

        if self._is_prior:
            # Very first step: there is no chain state yet, so just draw the
            # probability table from the Dirichlet prior (default: flat prior).
            if self._alpha is None:
                self._set_non_informative_alpha()
            theta = dirichlet.rvs(self._alpha[U])[0]
            self._is_prior = False
        else:
            exo_f = self._model.factors[U]
            theta_current = np.array(exo_f.values)
            n_samples = len(u_old)

            if method == 'sqrt':
                # Globally-balanced proposal g(t) = sqrt(t): q(u) proportional to
                # sqrt(theta[u]) over the row's valid states.
                w = np.sqrt(theta_current)
                weights = w * valid_mask
                p_matrix = weights / (weights.sum(axis=1, keepdims=True) + 1e-10)
                u_new = self._vectorized_choice(p_matrix)

                # The row's normalizer cancels between forward and backward moves
                # (same valid set), so the proposal ratio reduces to w[old]/w[new].
                target_ratio = theta_current[u_new] / (theta_current[u_old] + 1e-10)
                proposal_ratio = w[u_old] / (w[u_new] + 1e-10)
                ratio = np.minimum(target_ratio * proposal_ratio, 1.0)

            elif method == 'barker':
                # Barker proposal g(t) = t / (1 + t): W[i, j] = theta[j]/(theta[i]+theta[j]).
                theta_2d = theta_current.reshape(1, -1)
                W_matrix = theta_2d / (theta_current.reshape(-1, 1) + theta_2d + 1e-10)

                # Forward proposal from u_old, restricted to each row's valid set.
                weights_fwd = W_matrix[u_old, :] * valid_mask
                Z_fwd = weights_fwd.sum(axis=1)
                p_fwd = weights_fwd / (Z_fwd[:, None] + 1e-10)
                u_new = self._vectorized_choice(p_fwd)

                # Backward normalizer to build the reverse proposal probability.
                Z_bck = (W_matrix[u_new, :] * valid_mask).sum(axis=1)
                q_forward = W_matrix[u_old, u_new] / (Z_fwd + 1e-10)
                q_backward = W_matrix[u_new, u_old] / (Z_bck + 1e-10)

                target_ratio = theta_current[u_new] / (theta_current[u_old] + 1e-10)
                proposal_ratio = q_backward / (q_forward + 1e-10)
                ratio = np.minimum(target_ratio * proposal_ratio, 1.0)

            else:
                raise ValueError(f"Unknown method: {method}")

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
            counts_u = np.bincount(self._int_exo_samples[U], minlength=len(exo_f.domain[U]))
            beta = np.array(self._alpha[U]) + counts_u
            theta = dirichlet.rvs(beta)[0]

        # Wrap the sampled probabilities as U's updated conditional factor.
        return MultinomialFactor({U: m.domains[U]}, theta)

    def _set_non_informative_alpha(self):
        # Flat (uniform) Dirichlet prior: all-ones concentration per variable.
        self._alpha = {U: np.ones(len(self._model.domains[U])) for U in self._trainable_vars}


if __name__ == "__main__":
    directory_path = "/Users/antoniogonzalezalves/Documents/s23/"
    download_path = "/Users/antoniogonzalezalves/Documents/BenchMarkMH/"

    # Load an SCM and its observational data (prefix columns to match var names).
    m = StructuralCausalModel.read(directory_path + "simple_nparents2_nzr04_zdr05_1.uai")
    data = pd.read_csv(directory_path + "simple_nparents2_nzr04_zdr05_1.csv", index_col=0).add_prefix('V')

    import time
    # Time the full MCMC run ('sqrt' or 'barker' balancing function).
    start_time = time.time()
    mhs = MetropolisHastingsSampling(m, method='sqrt')
    mhs.run(data[m.endogenous], max_iter=10000)
    end_time = time.time()

    print(f"Elapsed time: {end_time - start_time:.4f} seconds")
    print(f"Final Acceptance Rates: {mhs.acceptance_rate}")

    # Aggregate the sampled models into a causal query (probability of sufficiency).
    inf = CausalMultiInference(mhs.model_evolution[200:], outliers_removal=True)
    print(inf.prob_sufficiency("V1", "V0", true_false_cause=(1, 0), true_false_effect=(1, 0))[0])
