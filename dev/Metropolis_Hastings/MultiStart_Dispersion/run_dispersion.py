"""
Multi-start dispersion experiment: do the four samplers escape their starting
point, or does the answer they return depend on where they began?

Each (model, query) is attacked from ``--starts`` deliberately over-dispersed
starting points, and each starting point is run ``--replicates`` times with
independent sampling streams. Crossing the two makes the variance decomposition
in ``analyze_dispersion.py`` able to separate genuine start-dependence
(mode-trapping) from ordinary Monte Carlo noise -- a plain seed sweep cannot,
because changing the seed changes both at once.

Two departures from ``Evaluation_MH_Variants.py``, both required for the
experiment to measure anything at all:

1. Starting points come from a low-concentration Dirichlet, not from
   ``randomize_factors``. The latter draws iid Uniform(0,1) and normalizes,
   which concentrates near the simplex centroid: for the latent cardinalities
   here (up to 2048 states) every draw is essentially the uniform distribution,
   so different seeds would give near-identical starts.
2. The samplers' first-step prior draw is disabled. ``_is_prior`` is a single
   scalar flag flipped inside the per-variable ``_updated_factor`` while the
   caller loops over every trainable variable, so only the *first* latent takes
   the prior branch -- and that branch overwrites its theta with a fresh
   Dirichlet draw, destroying the dispersed start for that one variable while
   leaving the others intact. Disabling it keeps the start intact and makes the
   four samplers behave consistently. An explicit alpha is always passed, since
   the disabled branch is also where the all-ones default was lazily filled in.

   Consequence for Parallel Tempering, accepted deliberately: that prior branch
   was also the only place PT's replicas diverged (each rung drew its own
   Dirichlet sample). With it disabled every replica starts from the identical
   tiled theta and the ladder must separate them through tempering alone. This
   is the intended configuration here -- all four samplers begin from the same
   dispersed point, which is the common baseline the comparison rests on -- but
   it does mean PT is not initialized the way Evaluation_MH_Variants.py runs it,
   and a smoke test showed PT scoring as the most start-dependent sampler. Read
   PT's result with that in mind: some of its start-dependence may be the
   missing replica diversity rather than a failure to escape modes.

Results are written wide (one row per model/start/replicate/checkpoint/query)
and appended as each run finishes, so an interrupted experiment resumes cleanly.
"""

import argparse
import hashlib
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# Make bcause and the sampler variants importable in the parent process and in
# spawned workers (macOS re-imports this module per worker).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_ALGO_DIR = os.path.join(_PARENT, "Algorithms")
for _p in (_REPO_ROOT, _ALGO_DIR, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

warnings.filterwarnings("ignore")

from bcause.factors import MultinomialFactor
from bcause.learning.parameter.gibbs import GibbsSampling
from bcause.models.cmodel import StructuralCausalModel
from bcause.util import randomUtil

from inference_queries import InferenceQueries
from PrecomputedMetropolis import MetropolisHastingsSampling as StandardMH
from ZanellaMCMC import MetropolisHastingsSampling as ZanellaMH
from ParallelTemperingMCMC import MetropolisHastingsSampling as ParallelTemperingMH

from Evaluation_MH_Variants import (
    SEMIMARKOVIAN_TEMPERATURES,
    _acceptance_scalar,
    _checkpoint_iterations,
    _norm_val,
    discover_sets,
)

# --- Experiment configuration -------------------------------------------------

DEFAULT_DATA_DIRS = ["/Users/antoniogonzalezalves/Documents/s23_semimarkovian/"]
DEFAULT_OUTPUT_DIR = os.path.join(_HERE, "output")
OUTPUT_FILENAME = "dispersion_runs.csv"
MANIFEST_FILENAME = "model_manifest.csv"

DEFAULT_N_MODELS = 40
DEFAULT_STARTS = 6
DEFAULT_REPLICATES = 3
DEFAULT_BURN_IN = 1000
DEFAULT_ITERATIONS = 10000
DEFAULT_CHECKPOINT_EVERY = 500
DEFAULT_WORKERS = 6
DEFAULT_OUTLIERS_REMOVAL = True

# Dirichlet concentration for the dispersed starts. Well below 1 so draws land
# near the simplex vertices and the starts are genuinely far apart -- the
# Gelman-Rubin requirement that starts be more dispersed than the posterior.
DEFAULT_START_CONCENTRATION = 0.05
# Floor applied after the Dirichlet draw so no latent state gets exactly zero
# mass (mirrors the intent of randomize_factors' allow_zero=False).
PROB_FLOOR = 1e-6

# Dirichlet prior concentration for the posterior updates. Always materialized
# explicitly -- see the module docstring on why the lazy default cannot be used.
DEFAULT_ALPHA = 1.0

# A query is only usable if its exact bounds leave something to explore.
# Degenerate (width ~ 0) and vacuous ([0,1]) queries have no between-start
# variance to measure; together they are ~47% of the semi-Markovian instances.
MIN_EXACT_WIDTH = 0.01
MAX_EXACT_WIDTH = 0.999

PT_TEMPERATURES = SEMIMARKOVIAN_TEMPERATURES

# (label, time column, acceptance column or None, factory)
SAMPLERS = [
    ("Gibbs", "Time_gibbs", None,
     lambda m, method, alpha: GibbsSampling(m, alpha=alpha)),
    ("MH", "Time_mh", "Acceptance_mh",
     lambda m, method, alpha: StandardMH(m, alpha=alpha)),
    ("Zanella", "Time_zanella", "Acceptance_zanella",
     lambda m, method, alpha: ZanellaMH(m, alpha=alpha, method=method)),
    ("PT", "Time_pt", "Acceptance_pt",
     lambda m, method, alpha: ParallelTemperingMH(m, alpha=alpha,
                                                  temperatures=PT_TEMPERATURES)),
]

META_COLUMNS = ["Model_Index", "nparents", "nzr", "zdr", "merged", "cardinality"]
RUN_COLUMNS = ["start_id", "replicate_id", "start_seed", "run_seed"]
# One run = one (model, start, replicate); resume must key on all three, or a
# restart would see the model present and skip its remaining starts.
IDENTITY_COLUMNS = META_COLUMNS + ["start_id", "replicate_id"]

OUTPUT_COLUMNS = (
    META_COLUMNS + RUN_COLUMNS
    + ["Iteration", "cause", "effect", "query"]
    + ["Gibbs_low", "Gibbs_upp", "Time_gibbs"]
    + ["MH_low", "MH_upp", "Time_mh", "Acceptance_mh"]
    + ["Zanella_low", "Zanella_upp", "Time_zanella", "Acceptance_zanella"]
    + ["PT_low", "PT_upp", "Time_pt", "Acceptance_pt"]
    + ["exact_low", "exact_upp"]
)


# --- Seeding ------------------------------------------------------------------

def stable_seed(*parts) -> int:
    """
    Deterministic 32-bit seed from arbitrary identifying parts.

    Unlike ``Evaluation_MH_Variants``' discovery-position seeding, this depends
    only on the identity handed in, so a seed does not silently change when the
    data directory or the filters change.

    Args:
        *parts: Values identifying the thing being seeded; stringified and
            joined, so ints, strings and ``None`` all work.

    Returns:
        An integer in ``[0, 2**32)``, usable as a numpy seed.
    """
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:8], 16)


# --- Dispersed starting points ------------------------------------------------

def dispersed_prior(model: StructuralCausalModel, concentration: float,
                    rng: np.random.Generator) -> StructuralCausalModel:
    """
    Build a copy of ``model`` whose exogenous factors are drawn from a
    low-concentration Dirichlet, giving a starting point near a random vertex of
    each latent simplex.

    Args:
        model: The SCM whose exogenous marginals are replaced.
        concentration: Dirichlet concentration; values well below 1 push draws
            toward the simplex vertices, spreading the starts far apart.
        rng: Generator supplying the draws. Passed explicitly (rather than using
            the global numpy state) so a start depends only on its own seed and
            is reproducible across replicates.

    Returns:
        A new model, identical except for its exogenous factors.

    Raises:
        ValueError: If an exogenous variable has parents, which would make its
            factor conditional and the single-vector draw below wrong.
    """
    m = model.copy()
    for u in model.exogenous:
        if any(True for _ in model.graph.predecessors(u)):
            raise ValueError(f"exogenous {u} has parents; expected a root variable")
        theta = rng.dirichlet(np.full(len(model.domains[u]), concentration))
        theta = np.clip(theta, PROB_FLOOR, None)
        theta /= theta.sum()
        m.set_factor(u, MultinomialFactor({u: model.domains[u]}, theta))
    return m


def _build_sampler(factory, prior, method, alpha):
    """
    Construct a sampler and disable its first-step prior draw.

    The prior branch would overwrite the first trainable variable's theta with a
    fresh Dirichlet sample, erasing the dispersed start for that variable while
    leaving every other variable's start intact. Disabling it keeps the start
    intact and removes that asymmetry. Safe only because ``alpha`` is always
    passed explicitly: the disabled branch is also where the all-ones default
    was lazily materialized.

    Applied uniformly to all four samplers, Parallel Tempering included, so every
    replica of the ladder starts from the same dispersed point -- see the module
    docstring for why that is deliberate and how it colours PT's result.
    """
    sampler = factory(prior, method, alpha)
    sampler._is_prior = False
    return sampler


# --- Model selection ----------------------------------------------------------

def _merged_cardinality(model_path: str) -> int:
    """
    Cardinality of the model's merged exogenous confounder -- the latent
    dimension along which mode-trapping is expected to scale. Returns the
    largest exogenous domain among variables feeding two or more endogenous
    children, or 0 if the model has no confounder.
    """
    m = StructuralCausalModel.read(model_path)
    sizes = [len(m.domains[u]) for u in m.exogenous
             if sum(1 for c in m.graph.successors(u) if m.is_endogenous(c)) >= 2]
    return max(sizes) if sizes else 0


def _informative_share(query_path: str) -> tuple:
    """(usable query count, total query count) for one model's query file."""
    q = pd.read_csv(query_path)
    q = q[q["query"].isin(("PS", "PN"))]
    if q.empty:
        return 0, 0
    width = q["upp"] - q["low"]
    usable = ((width > MIN_EXACT_WIDTH) & (width < MAX_EXACT_WIDTH)
              & np.isfinite(width))
    return int(usable.sum()), int(len(q))


def build_manifest(sets: list, manifest_path: str) -> pd.DataFrame:
    """
    Scan every candidate model once, recording its merged-confounder cardinality
    and how many of its queries are usable, then cache the result.

    Scanning reads all 1,187 models, so the result is cached: selection is
    auditable and a restart does not repeat the scan.
    """
    rows = []
    for i, s in enumerate(sets, start=1):
        meta = s["meta"]
        try:
            usable, total = _informative_share(s["query"])
            if usable == 0:
                continue
            card = _merged_cardinality(s["model"])
        except Exception as exc:
            print(f"  skipping {meta['index']}_{meta['merged']}: {exc}")
            continue
        rows.append({**{c: _norm_val(v) for c, v in zip(
            META_COLUMNS,
            (meta["index"], meta["nparents"], meta["nzr"], meta["zdr"],
             meta["merged"], meta["cardinality_children"]))},
            "merged_cardinality": card,
            "usable_queries": usable,
            "total_queries": total,
            "usable_share": usable / total if total else 0.0,
            "model": s["model"], "data": s["data"], "query": s["query"]})
        if i % 200 == 0:
            print(f"  scanned {i}/{len(sets)} models")
    manifest = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def select_models(manifest: pd.DataFrame, n_models: int, min_share: float) -> pd.DataFrame:
    """
    Choose ``n_models`` stratified by merged-confounder cardinality.

    Stratifying across the latent-size range turns "start-dependence grows with
    latent dimension" into a testable claim rather than one aggregate number.
    Within each stratum the models with the highest usable-query share come
    first, so each selected model yields as many measurable queries as possible.
    """
    pool = manifest[manifest["usable_share"] >= min_share].copy()
    if pool.empty:
        raise SystemExit(f"no models with usable_share >= {min_share}")

    # Quartile strata over the confounder cardinality, deduplicated because the
    # distribution is lumpy (many models share the same D).
    edges = np.unique(np.quantile(pool["merged_cardinality"], [0, .25, .5, .75, 1.0]))
    pool["stratum"] = (np.digitize(pool["merged_cardinality"], edges[1:-1])
                       if len(edges) > 2 else 0)

    picks, strata = [], sorted(pool["stratum"].unique())
    per = max(1, n_models // len(strata))
    for s in strata:
        block = pool[pool["stratum"] == s].sort_values(
            ["usable_share", "usable_queries", "Model_Index"],
            ascending=[False, False, True])
        picks.append(block.head(per))
    chosen = pd.concat(picks)

    # Top up from whatever is left if integer division left a shortfall.
    if len(chosen) < n_models:
        rest = pool.drop(chosen.index).sort_values(
            ["usable_share", "usable_queries"], ascending=False)
        chosen = pd.concat([chosen, rest.head(n_models - len(chosen))])
    return chosen.head(n_models).sort_values("Model_Index").reset_index(drop=True)


# --- Running ------------------------------------------------------------------

def _run_with_checkpoints(sampler, endo_data: pd.DataFrame, burn_in: int,
                          checkpoints: list, outliers_removal: bool) -> dict:
    """
    Run one sampler to the last checkpoint, snapshotting an InferenceQueries over
    the post-burn-in models plus the cumulative learning time and acceptance rate.

    Mirrors ``Evaluation_MH_Variants._run_with_checkpoints``; kept separate only
    because the start-diagnostic below needs the initial theta captured between
    ``initialize()`` and the first ``step()``.

    Returns:
        ``{iteration: (inference, elapsed_seconds, acceptance_rate)}``.
    """
    results = {}
    start = time.time()
    sampler.initialize(endo_data)
    elapsed = time.time() - start
    pointer = 0
    for step in range(1, checkpoints[-1] + 1):
        step_start = time.time()
        sampler.step()
        elapsed += time.time() - step_start
        if step == checkpoints[pointer]:
            results[step] = (InferenceQueries(sampler.model_evolution[burn_in:],
                                              outliers_removal=outliers_removal),
                             elapsed, _acceptance_scalar(sampler))
            pointer += 1
    return results


def _query_value(inf: InferenceQueries, instance) -> list:
    """Estimate PS/PN for one query instance, as ``[low, upp]``."""
    if instance.query == "PS":
        return inf.prob_sufficiency(instance.cause, instance.effect,
                                    true_false_cause=(1, 0), true_false_effect=(1, 0))
    return inf.prob_necessity(instance.cause, instance.effect,
                              true_false_cause=(1, 0), true_false_effect=(1, 0))


def process_run(job: dict) -> list:
    """
    Run all four samplers for one (model, start, replicate) and return one wide
    row per (checkpoint, usable query). Designed to run in a worker process.
    """
    meta, start_id, replicate_id = job["meta"], job["start_id"], job["replicate_id"]
    burn_in, iterations = job["burn_in"], job["iterations"]
    checkpoints = _checkpoint_iterations(iterations, job["checkpoint_every"], burn_in)
    if not checkpoints:
        return []

    model = StructuralCausalModel.read(job["model"])
    data = pd.read_csv(job["data"], index_col=0).add_prefix("V")
    query = pd.read_csv(job["query"])
    endo_data = data[model.endogenous]

    alpha = {u: np.full(len(model.domains[u]), job["alpha"]) for u in model.exogenous}

    # The start depends only on (model, start_id) via its own Generator, so it is
    # byte-identical across replicates; the run seed then drives the sampling.
    # Keeping the two separate is what lets the analysis attribute between-start
    # variance to the start rather than to the random stream.
    start_seed = job["start_seed"]
    run_seed = job["run_seed"]
    prior = dispersed_prior(model, job["start_concentration"],
                            np.random.default_rng(start_seed))

    runs = {}
    for label, time_col, accept_col, factory in SAMPLERS:
        randomUtil.seed(run_seed)
        runs[label] = (time_col, accept_col, _run_with_checkpoints(
            _build_sampler(factory, prior, job["method"], alpha),
            endo_data, burn_in, checkpoints, job["outliers_removal"]))

    base = {c: _norm_val(v) for c, v in zip(
        META_COLUMNS,
        (meta["index"], meta["nparents"], meta["nzr"], meta["zdr"],
         meta["merged"], meta["cardinality_children"]))}
    base.update({"start_id": start_id, "replicate_id": replicate_id,
                 "start_seed": start_seed, "run_seed": run_seed})

    rows = []
    for iteration in checkpoints:
        for instance in query.itertuples(index=False):
            if instance.query not in ("PS", "PN"):
                continue
            width = instance.upp - instance.low
            if not (MIN_EXACT_WIDTH < width < MAX_EXACT_WIDTH):
                continue
            row = dict(base)
            row.update({"Iteration": iteration, "cause": instance.cause,
                        "effect": instance.effect, "query": instance.query,
                        "exact_low": instance.low, "exact_upp": instance.upp})
            for label, (time_col, accept_col, per_iter) in runs.items():
                inf, elapsed, acceptance = per_iter[iteration]
                low, upp = _query_value(inf, instance)
                row[f"{label}_low"], row[f"{label}_upp"] = low, upp
                row[time_col] = elapsed
                if accept_col is not None:
                    row[accept_col] = acceptance
            rows.append(row)
    return rows


def load_completed_keys(output_path: str) -> set:
    """Identities of the (model, start, replicate) runs already in the output."""
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return set()
    try:
        existing = pd.read_csv(output_path, usecols=IDENTITY_COLUMNS)
    except (ValueError, pd.errors.EmptyDataError):
        return set()
    return {tuple(_norm_val(v) for v in row)
            for row in existing[IDENTITY_COLUMNS].itertuples(index=False, name=None)}


def build_jobs(chosen: pd.DataFrame, args, completed: set) -> list:
    """Expand the selected models into one job per (model, start, replicate)."""
    jobs = []
    for rec in chosen.to_dict("records"):
        meta = {"index": int(rec["Model_Index"]), "nparents": int(rec["nparents"]),
                "nzr": int(rec["nzr"]), "zdr": int(rec["zdr"]),
                "merged": rec["merged"], "cardinality_children": int(rec["cardinality"])}
        model_key = tuple(_norm_val(rec[c]) for c in META_COLUMNS)
        for start_id in range(args.starts):
            start_seed = stable_seed(*model_key, "start", start_id)
            for replicate_id in range(args.replicates):
                key = model_key + (_norm_val(start_id), _norm_val(replicate_id))
                if key in completed:
                    continue
                jobs.append({
                    "meta": meta, "model": rec["model"], "data": rec["data"],
                    "query": rec["query"], "start_id": start_id,
                    "replicate_id": replicate_id, "start_seed": start_seed,
                    "run_seed": stable_seed(*model_key, "run", start_id, replicate_id),
                    "burn_in": args.burn_in, "iterations": args.iterations,
                    "checkpoint_every": args.checkpoint_every,
                    "method": args.zanella_method, "alpha": args.alpha,
                    "start_concentration": args.start_concentration,
                    "outliers_removal": args.outliers_removal,
                })
    return jobs


def parse_args() -> argparse.Namespace:
    """Command-line configuration for the dispersion experiment."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", nargs="+", default=DEFAULT_DATA_DIRS)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--n-models", type=int, default=DEFAULT_N_MODELS)
    p.add_argument("--starts", type=int, default=DEFAULT_STARTS,
                   help="Dispersed starting points per model (K).")
    p.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES,
                   help="Independent streams per starting point (R); needs >= 2 "
                        "so within-start variance is estimable.")
    p.add_argument("--burn-in", type=int, default=DEFAULT_BURN_IN)
    p.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    p.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                   help="Dirichlet prior concentration for the posterior updates.")
    p.add_argument("--start-concentration", type=float, default=DEFAULT_START_CONCENTRATION,
                   help="Dirichlet concentration for the dispersed starts; "
                        "well below 1 spreads them toward the simplex vertices.")
    p.add_argument("--min-usable-share", type=float, default=0.4,
                   help="Only consider models where at least this share of "
                        "queries have non-degenerate, non-vacuous exact bounds.")
    p.add_argument("--zanella-method", default="sqrt", choices=("sqrt", "barker"))
    p.add_argument("--outliers-removal", action=argparse.BooleanOptionalAction,
                   default=DEFAULT_OUTLIERS_REMOVAL)
    p.add_argument("--rebuild-manifest", action="store_true",
                   help="Rescan every candidate model even if a manifest exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Select models and print the job plan without running.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.burn_in >= args.iterations:
        raise ValueError("burn-in must be smaller than the total number of iterations")
    if args.replicates < 2:
        raise ValueError("--replicates must be >= 2 to estimate within-start variance")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, OUTPUT_FILENAME)
    manifest_path = os.path.join(args.output_dir, MANIFEST_FILENAME)

    if args.rebuild_manifest or not os.path.exists(manifest_path):
        sets = discover_sets(args.data_dir, [], [], None)
        if not sets:
            raise SystemExit(f"No complete (model, data, query) triples in {args.data_dir}")
        print(f"Scanning {len(sets)} candidate models (cached to {manifest_path})...")
        manifest = build_manifest(sets, manifest_path)
    else:
        manifest = pd.read_csv(manifest_path)
    print(f"Manifest: {len(manifest)} candidate models")

    chosen = select_models(manifest, args.n_models, args.min_usable_share)
    print(f"Selected {len(chosen)} models, stratified by confounder cardinality "
          f"({chosen.merged_cardinality.min()}-{chosen.merged_cardinality.max()}); "
          f"{int(chosen.usable_queries.sum())} usable queries in total")

    completed = load_completed_keys(output_path)
    jobs = build_jobs(chosen, args, completed)
    total = len(chosen) * args.starts * args.replicates
    workers = args.workers if args.workers is not None else os.cpu_count()
    print(f"Runs: {total} total | {total - len(jobs)} already done | {len(jobs)} to run "
          f"| starts={args.starts} replicates={args.replicates} "
          f"iterations={args.iterations} workers={workers}")

    if args.dry_run:
        print("\n--dry-run: nothing executed. Selected models:")
        print(chosen[META_COLUMNS + ["merged_cardinality", "usable_queries",
                                     "usable_share"]].to_string(index=False))
        return
    if not jobs:
        print("Nothing to do; every run already recorded.")
        return

    write_header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_run, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            tag = (f"model {job['meta']['index']}_{job['meta']['merged']} "
                   f"start {job['start_id']} rep {job['replicate_id']}")
            try:
                rows = future.result()
                if rows:
                    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(
                        output_path, mode="a", header=write_header, index=False)
                    write_header = False
                print(f"[{done}/{len(jobs)}] {tag} done ({len(rows)} rows)")
            except Exception as exc:  # one failed run must not stop the sweep
                print(f"[{done}/{len(jobs)}] {tag} FAILED: {exc}")

    print(f"Results in {output_path}")


if __name__ == "__main__":
    main()
