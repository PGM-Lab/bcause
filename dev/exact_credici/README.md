# exact_credici

Exact, in-process replacement for credici's counterfactual solver
(`runCVE.java` → `CredalCausalVE`), computing **PN / PS / PNS** interval bounds
for a Structural Causal Model constrained by observational data.

credici solves the same problem with vertex credal variable elimination, which
is a valid but sometimes **loose outer** approximation. This computes the
**exact tight** bound as a small linear program.

## Files

| file | what it is |
|---|---|
| `exact_credal_solver.py` | **My method.** Builds the credal set + query objective and solves it exactly with `scipy.optimize.linprog` (Charnes–Cooper for the fractional PN/PS). Handles binary effects and the 3-state `ysize3` effects (pins k-1 conditional states per config). Module docstring explains the full derivation. |
| `brute_force.py` | **Independent check.** Optimises the *same* problem by random-vertex sampling (no Charnes–Cooper), prints LP vs brute-force vs credici side by side, and shows the witness `theta*` so results are verifiable by hand. |
| `generate_markovian.py` | **Batch pipeline (Markovian).** Runs the exact solver over every model in an input directory and writes a new dataset (`.uai` + `.csv` + exact `_query.csv`) — replacing credici's bounds with the exact tight ones, keeping credici's column layout. |
| `exact_credal_scm.py` | **General solver** for (semi-)Markovian SCMs. Generalises `exact_credal_solver` from "one confounder under the effect" to "the effect's whole c-component", so it also handles the confounders created by merging exogenous variables. Vectorised (numpy tensor propagation) and exposes `is_feasible()`. |
| `generate_semimarkovian.py` | **Batch pipeline (semi-Markovian).** Merges exogenous variables (every combination of sizes 2..E), solves each merged model with `exact_credal_scm`, resampling the data on infeasibility and skipping merges that never become feasible. Writes `s23_semimarkovian/`. |
| `mismatches.txt` | Every *binary-effect* query (across the 1193 s23 models) where my bound differs from credici's golden bound: model name, query, both intervals, difference, and status. |

## Run

```bash
# exact bounds for one model (defaults to the largest-gap model + a few queries)
python exact_credal_solver.py [<model_base_without_extension>] ["PS(V1,V0)" ...]

# verify those bounds by brute force and compare to credici's golden file
python brute_force.py [<model_base>] ["PS(V1,V0)" ...] --samples 40000
```

`<model_base>` points at a `<base>.uai` + `<base>.csv` pair (credici's golden
`<base>_query.csv` is read automatically if present).

### Regenerate a whole dataset with exact bounds

```bash
python generate_markovian.py --input /path/to/s23 --output /path/to/s23_markovian
```

Reads every `<base>.uai` + `<base>.csv` + `<base>_query.csv` triple in `--input`,
recomputes each query's `[low, upp]` with the exact solver, and writes the model,
data and the new exact `_query.csv` to `--output` (parallelised across cores).
Run on the 1193 `s23` models this produces `s23_markovian/` in ~16 s (3811
queries; 1 row skipped where the cause is not a direct parent). Binary-effect
results are the verified exact refinement of credici; the `ysize3` (3-state
effect) results are also exact but diverge from credici, whose multi-state
bounds can be *wrong* (they exclude valid SCMs — demonstrable with a witness
`theta`), not merely loose.

### Build the semi-Markovian dataset

```bash
python generate_semimarkovian.py --input /path/to/s23 --output /path/to/s23_semimarkovian --all
```

For each Markovian model it merges the exogenous variables in every combination
of sizes `2..E` (E = number of exogenous), producing shared confounders. Each
merged model is solved with `exact_credal_scm`; on infeasibility
(`NoFeasibleSolution`, credici's `NoFeasibleSolutionException`) the dataset is
resampled from a fresh `Dirichlet(0.5)` over the endogenous joint and retried up
to `MAX_RESAMPLE_ATTEMPTS` times, then skipped. Solved merges write
`semi_<...>_<positions>.uai/.csv/_query.csv`, where `<positions>` are the 1-based
indices of the merged exogenous variables (e.g. `_23`, `_123`).

**Exactness of the semi-Markovian bounds.** When the queried cause and the
effect share the merged confounder (the intended confounded case) the LP bound is
exact — independently confirmed by brute-force vertex sampling of the same
polytope. When a merge leaves them under two independent free confounders, taking
their joint as free relaxes that independence, giving a valid *outer* bound
(credici, enforcing the independence, can be tighter there). Merges requiring
exact marginal independence between two roots (e.g. confounding the effect with
only one of two causes) are typically infeasible for both this solver and
credici, and get skipped.

## Result in one line

Across all s23 models: my intervals are **always within credici's** (0 bugs),
**exact-equal on ~88%** of queries, and **strictly tighter on the rest** — the
212 genuine refinements are listed in `mismatches.txt`. The LP and brute force
agree to ~1e-5, and each witness `theta*` satisfies every constraint exactly
(`max|A·theta − b| = 0`) while reproducing the reported bound.
