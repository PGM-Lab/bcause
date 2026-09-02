"""
Compute the PMF of exogenous variables given data for endogenous variables
using a Parallel Tempering Metropolis-Hastings sampler.

Several replicas of the chain run at increasing temperatures. The cold chain
(T = 1) targets the exact posterior while the hot chains flatten it to escape
local modes; periodic replica-exchange swaps let good states discovered by the
hot chains migrate down to the cold one. Every replica advances all data rows
in parallel with the standard self-avoiding MH proposal.
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
    Run a single optimization of the exogenous variables of an SCM using
    Parallel Tempering Metropolis-Hastings.

    Each latent (exogenous) variable U is inferred with ``n_chains`` tempered MH
    chains that run over every data row in parallel. Acceptance ratios are
    flattened by the per-chain temperature, a replica exchange is periodically
    attempted between adjacent chains, and the counts feeding each chain's
    Dirichlet posterior are tempered so the hot chains stay diffuse. The cold
    chain (temperature 1) provides the reported factor. ``change_rate`` reports,
    per chain, how often each replica's sampled state actually moves -- the
    diagnostic to watch when calibrating the temperature ladder: a chain stuck
    near 0 is too cold to explore, one near 1 is too hot to discriminate states.
    """

    @property
    def acceptance_rate(self) -> Dict[str, float]:
        # Fraction of cold-chain proposals accepted so far, per trainable variable
        # (0.0 when no proposal has been made yet to avoid division by zero).
        return {U: (self.accepted_proposals[U] / self.total_proposals[U]) if self.total_proposals[U] > 0 else 0.0
                for U in self._trainable_vars}

    @property
    def change_rate(self) -> Dict[str, np.ndarray]:
        """
        Fraction of MH steps in which each replica's sampled state actually
        changed, per trainable variable, as an array of length ``n_chains``
        (one entry per temperature, index 0 = cold chain). Unlike
        ``acceptance_rate`` this counts a real state change, not just an
        accepted proposal: rows with only one valid state can "accept" a
        self-avoiding proposal that leaves the state unchanged, which would
        otherwise inflate the rate without reflecting any actual mixing.
        Use this across chains to see which temperatures are too cold (rate
        near 0, chain stuck) or too hot (rate near 1, chain moving freely
        without discriminating states) when calibrating the ladder.
        """
        return {
            U: (self.changed_updates[U] / self.total_updates[U]) if self.total_updates[U] > 0
            else np.zeros(self._n_chains)
            for U in self._trainable_vars
        }

    def __init__(self, prior_model: StructuralCausalModel, trainable_vars: list = None, alpha: dict = None,
                 temperatures: np.ndarray = None):
        """
        Args:
            prior_model: base StructuralCausalModel.
            trainable_vars: latent variables to optimize (None -> auto-detect
                the fully unobserved exogenous columns).
            alpha: Dirichlet hyperparameters per variable (None -> flat prior).
            temperatures: increasing replica temperatures; the first must be 1.0
                (the cold, target chain). Defaults to [1, 2, 5, 10].
        """
        # Base model; fix_numeric_domains() makes domain values usable as array indices.
        self._prior_model = prior_model.fix_numeric_domains()
        self._trainable_vars = trainable_vars           # latent vars to infer (None -> auto-detect)
        self._alpha = alpha                             # Dirichlet hyperparameters (None -> flat prior)
        self._is_prior = True                           # first step samples from the prior, not MH

        # Replica temperatures: index 0 is the cold (target) chain, T = 1.0.
        if temperatures is None:
            temperatures = [1.0, 2.0, 5.0, 10.0]
        self._temperatures = np.asarray(temperatures, dtype=float)
        self._n_chains = len(self._temperatures)
        self._iter_count = 0                            # step counter driving the swap schedule

        # Per-variable lookup tables and state, all keyed by variable name U:
        self._val2idx = {}          # domain value  -> integer index
        self._idx2val = {}          # integer index -> domain value
        self._int_sampling_set = {} # context "key"  -> tuple of valid state indices
        self._valid_masks = {}      # boolean matrix (rows x states): which states are allowed per row
        self._int_exo_samples = {}  # current U assignment (as index): shape (n_chains, rows)
        self._prob_tables = {}      # current probability table for U: shape (n_chains, states)
        self._data = {}             # per-variable context data with a hashable "key" column
        self._endo_factors = {}     # children factors of U (the endogenous vars it feeds)
        self._endo_vars = {}        # for each child factor, its variables other than U

    def initialize(self, data: pd.DataFrame, **kwargs):
        """Precompute the static structures and the initial state of every replica."""
        # Work on a copy of the prior so the original model is left untouched.
        self._model = self._prior_model.copy()
        self._process_data(data.copy())

        # Acceptance bookkeeping per variable (tracked on the cold chain).
        self.acceptance_history = {U: [] for U in self._trainable_vars}
        self.total_proposals = {U: 0 for U in self._trainable_vars}
        self.accepted_proposals = {U: 0 for U in self._trainable_vars}

        # Change-rate bookkeeping per variable, tracked separately for every
        # chain (temperature) to support calibrating the temperature ladder.
        self.change_history = {U: [] for U in self._trainable_vars}
        self.changed_updates = {U: np.zeros(self._n_chains) for U in self._trainable_vars}
        self.total_updates = {U: 0 for U in self._trainable_vars}

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

            # Replicate the prior table and an initial per-row assignment across
            # all chains: tables (n_chains, states), samples (n_chains, rows).
            base_theta = np.array(self._model.factors[U].values, dtype=float)
            init_samples = self._get_initial_exogenous(U, base_theta)
            self._prob_tables[U] = np.tile(base_theta, (self._n_chains, 1))
            self._int_exo_samples[U] = np.tile(init_samples, (self._n_chains, 1))

    def _stop_learning(self) -> bool:
        # No convergence criterion: the chain runs for the requested max_iter.
        return False

    def _calculate_updated_factors(self, **kwargs) -> Dict[str, MultinomialFactor]:
        # One PT step for every trainable variable, returning their new factors.
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
            booleans = {k: np.array(endo_f[k].R(**row[v].to_dict()).values, dtype=int).astype(bool)
                        for k, v in endo_vars.items()}
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

    def _get_initial_exogenous(self, U: str, theta: np.ndarray) -> np.ndarray:
        """Draw an initial per-row assignment from the prior restricted to valid states."""
        probs = theta * self._valid_masks[U]
        p_matrix = probs / (probs.sum(axis=1, keepdims=True) + 1e-10)
        return self._vectorized_choice(p_matrix)

    def _propose_self_avoiding(self, U: str, u_old: np.ndarray) -> np.ndarray:
        """
        Propose, for every (chain, row), a uniform state among the row's valid
        states other than the current one, keeping the current state where it is
        the only valid option. Fully vectorized over chains and rows.
        """
        valid_mask = self._valid_masks[U]
        n_chains, n_rows = u_old.shape
        domain_size = valid_mask.shape[1]

        # Forbid re-proposing the current state in each (chain, row) slice.
        proposal_mask = np.broadcast_to(valid_mask, (n_chains, n_rows, domain_size)).copy()
        chain_idx, row_idx = np.indices((n_chains, n_rows))
        proposal_mask[chain_idx, row_idx, u_old] = False

        # Uniform proposal over the remaining valid states (flattened for choice).
        has_options = proposal_mask.any(axis=2)
        flat = proposal_mask.astype(float).reshape(n_chains * n_rows, domain_size)
        u_new = self._vectorized_choice(flat).reshape(n_chains, n_rows)

        # Rows with a single valid state keep their current assignment.
        return np.where(has_options, u_new, u_old)

    def _updated_factor(self, U: str) -> MultinomialFactor:
        """Run one Parallel Tempering step for U and return the cold-chain factor."""
        m = self._model

        if self._is_prior:
            # Very first step: there is no chain state yet, so seed every replica's
            # table from the Dirichlet prior (default: flat prior).
            if self._alpha is None:
                self._set_non_informative_alpha()
            self._prob_tables[U] = dirichlet.rvs(self._alpha[U], size=self._n_chains)
            self._is_prior = False
            return MultinomialFactor({U: m.domains[U]}, self._prob_tables[U][0])

        self._iter_count += 1
        exo_f = self._model.factors[U]
        domain_size = len(exo_f.domain[U])
        u_old = self._int_exo_samples[U]        # (n_chains, rows)
        theta = self._prob_tables[U]            # (n_chains, states)
        n_chains, n_rows = u_old.shape
        chain_ar = np.arange(n_chains)[:, None]

        # --- Independent tempered MH step for all chains at once ---
        u_new = self._propose_self_avoiding(U, u_old)
        old_prob = theta[chain_ar, u_old]
        new_prob = theta[chain_ar, u_new]

        # Parallel Tempering flattens each chain's acceptance ratio by 1/T.
        inv_temp = (1.0 / self._temperatures)[:, None]
        ratio = np.minimum((new_prob / (old_prob + 1e-10)) ** inv_temp, 1.0)
        accept = np.random.rand(n_chains, n_rows) <= ratio
        self._int_exo_samples[U] = np.where(accept, u_new, u_old)

        # Track the cold-chain acceptance rate for this step.
        n_accepted = accept[0].sum()
        self.accepted_proposals[U] += n_accepted
        self.total_proposals[U] += n_rows
        self.acceptance_history[U].append(n_accepted / n_rows if n_rows > 0 else 0)

        # Track the per-chain change rate: an accepted proposal only counts as
        # a change if the sampled value actually moved (rows with a single
        # valid state can "accept" a no-op proposal, see change_rate docstring).
        n_changed = (self._int_exo_samples[U] != u_old).sum(axis=1)
        self.changed_updates[U] += n_changed
        self.total_updates[U] += n_rows
        self.change_history[U].append(n_changed / n_rows if n_rows > 0 else np.zeros(n_chains))

        # --- Replica exchange (swap) ---
        # Periodically try to swap a random adjacent pair of chains, letting good
        # states found by the hotter chain migrate towards the cold target.
        if self._iter_count % 20 == 0:
            samples = self._int_exo_samples[U]
            c = np.random.randint(0, n_chains - 1)
            pair = [c, c + 1]
            # Unnormalized log-likelihood (energy) of each chain's configuration.
            pair_prob = np.take_along_axis(theta[pair], samples[pair], axis=1)
            log_energy = np.sum(np.log(pair_prob + 1e-10), axis=1)
            delta_beta = (1.0 / self._temperatures[c]) - (1.0 / self._temperatures[c + 1])
            swap_alpha = np.exp(min((log_energy[1] - log_energy[0]) * delta_beta, 0.0))
            if np.random.rand() < swap_alpha:
                # Swap both configurations and parameter tables between the pair.
                samples[pair] = samples[pair[::-1]]
                theta[pair] = theta[pair[::-1]]

        # --- Tempered Dirichlet update, per chain ---
        # Hot chains scale their counts by 1/T so their posteriors stay diffuse.
        for c in range(n_chains):
            counts_u = np.bincount(self._int_exo_samples[U][c], minlength=domain_size)
            beta = np.array(self._alpha[U]) + counts_u / self._temperatures[c]
            self._prob_tables[U][c] = dirichlet.rvs(beta)[0]

        # The cold chain (index 0) is the target distribution we report.
        return MultinomialFactor({U: m.domains[U]}, self._prob_tables[U][0])

    def _set_non_informative_alpha(self):
        # Flat (uniform) Dirichlet prior: all-ones concentration per variable.
        self._alpha = {U: np.ones(len(self._model.domains[U])) for U in self._trainable_vars}


if __name__ == "__main__":
    directory_path = "/Users/antoniogonzalezalves/Documents/s23/"
    download_path = "/Users/antoniogonzalezalves/Documents/BenchMarkMH/"

    # Load an SCM and its observational data (prefix columns to match var names).
    m = StructuralCausalModel.read(directory_path + "simple_nparents2_nzr08_zdr10_3.uai")
    data = pd.read_csv(directory_path + "simple_nparents2_nzr08_zdr10_3.csv", index_col=0).add_prefix('V')

    import time
    # Time the full Parallel Tempering run.
    start_time = time.time()
    mhs = MetropolisHastingsSampling(m)
    mhs.run(data[m.endogenous], max_iter=2000)
    end_time = time.time()

    print(f"Elapsed time: {end_time - start_time:.4f} seconds")
    print("Acceptance Rates:", mhs.acceptance_rate)
    print("Change Rates per chain (temperatures =", list(mhs._temperatures), "):", mhs.change_rate)

    # Aggregate the sampled models into a causal query (probability of necessity).
    inf_mh_1 = CausalMultiInference(mhs.model_evolution, outliers_removal=False)
    print("Result without outliers removal:",
          inf_mh_1.prob_necessity("V2", "V0", true_false_cause=(1, 0), true_false_effect=(1, 0)))
    inf_mh_2 = CausalMultiInference(mhs.model_evolution, outliers_removal=True)
    print("Result with outliers removal:",
          inf_mh_2.prob_necessity("V2", "V0", true_false_cause=(1, 0), true_false_effect=(1, 0)))
