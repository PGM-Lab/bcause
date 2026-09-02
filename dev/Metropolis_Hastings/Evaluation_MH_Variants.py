"""
Parallel evaluation of four causal-effect samplers on semi-Markovian SCMs:
Gibbs Sampling, standard Metropolis-Hastings, Zanella (locally-balanced) MH and
Parallel Tempering MH.

Two filename layouts are supported, and several data directories may be scanned
at once:
    semi_nparents{P}_nzr{NZR}_zdr{ZDR}_{ID}_{MERGED}{.uai | .csv | _query.csv}
    simple_nparents{P}_nzr{NZR}_zdr{ZDR}[_ysize{C}]_{ID}{...}
where ``ID`` identifies the model and the optional trailing ``MERGED`` labels the
merged exogenous variables. Models without a merged number (the ``s23`` set)
report ``NA`` in the ``merged`` column. Every complete triple is a job; jobs run
in parallel across processes. For each query the four samplers are run for
``iterations`` steps, the first ``burn_in`` recorded models are discarded, and
the probability of sufficiency/necessity is estimated and compared against the
exact bounds stored in the query file. The three MH variants also report their
acceptance rate. Results are written wide (one row per query) to
``Results_MH.csv``.

The Dirichlet prior concentration is configurable with ``--alpha`` (a scalar
broadcast to every latent state); the default is the non-informative all-ones
prior.

Each model's rows are appended to the CSV as soon as it finishes, so an
interrupted run keeps every model completed up to that point. On restart the
models already present in the CSV are detected and skipped.
"""

import argparse
import math
import os
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# Make bcause and the algorithm variants importable in both the parent process
# and the spawned workers (macOS default start method re-imports this module).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_ALGO_DIR = os.path.join(_HERE, "Algorithms")
for _p in (_REPO_ROOT, _ALGO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

warnings.filterwarnings("ignore")

from bcause.learning.parameter.gibbs import GibbsSampling
from bcause.models.cmodel import StructuralCausalModel
from bcause.util import randomUtil

from inference_queries import InferenceQueries
from PrecomputedMetropolis import MetropolisHastingsSampling as StandardMH
from ZanellaMCMC import MetropolisHastingsSampling as ZanellaMH
from ParallelTemperingMCMC import MetropolisHastingsSampling as ParallelTemperingMH

# Defaults; override any of them from the command line (see parse_args()).
DEFAULT_DATA_DIRS = [
    #"/Users/antoniogonzalezalves/Documents/s23_semi/",
    #"/Users/antoniogonzalezalves/Documents/s23/",
    "/Users/antoniogonzalezalves/Documents/s23_markovian/"
    #"/Users/antoniogonzalezalves/Documents/s23_semimarkovian/"
]
DEFAULT_OUTPUT_DIR = "/Users/antoniogonzalezalves/Documents/resultados_markovian/"
DEFAULT_BURN_IN = 1000
DEFAULT_ITERATIONS = 10000
DEFAULT_CHECKPOINT_EVERY = 500
DEFAULT_OUTLIERS_REMOVAL = True  # drop outlier models inside the inference
DEFAULT_WORKERS = 6  # parallel worker processes; None uses all available CPU cores
DEFAULT_ALPHA = None  # Dirichlet prior concentration; None -> all-ones (uniform)
DEFAULT_SEED_OFFSET = 1000  # bump (e.g. 1000) for an independent replication of the run
DEFAULT_MAX_NPARENTS = 5  # drop models whose nparents exceeds this; None keeps all
OUTPUT_FILENAME = "Results_MH.csv"

# Wide output layout: the "merged" column sits right after "zdr" and every
# sampler contributes its estimate plus the wall-clock time of its run; the three
# MH variants also contribute their acceptance rate.
OUTPUT_COLUMNS = [
    "Model_Index", "nparents", "nzr", "zdr", "merged", "cardinality",
    "Iteration", "cause", "effect", "query",
    # RNG seed this model was run with: its position in the discovery order plus
    # --seed-offset. Identifies the realization, and makes any single row
    # reproducible on its own.
    "seed",
    "Gibbs_Sampling", "Time_gibbs",
    "Metropolis_Hastings", "Time_mh", "Acceptance_mh",
    "Zanella", "Time_zanella", "Acceptance_zanella",
    "Parallel_Tempering", "Time_pt", "Acceptance_pt",
    "Exact_Probability",
]

# Columns that jointly identify one model triple, used to skip already-done work.
IDENTITY_COLUMNS = ["Model_Index", "nparents", "nzr", "zdr", "merged", "cardinality"]

# Parallel Tempering ladders, calibrated separately per model family so every
# adjacent pair's replica-swap acceptance rate lands in the standard 20-40%
# target band (local/PT_calibration/calibrate_temperatures.py). The two
# families need substantially different ladders -- semi-Markovian models have
# merged exogenous confounders shared across endogenous children, giving a
# more rugged latent posterior that only tolerates small temperature jumps
# between adjacent chains, whereas the simpler Markovian models tolerate much
# larger jumps (and hit a hard swap-rate floor by the 4th rung, so they get
# one fewer rung entirely).
#
# Markovian (s23_markovian): swap rates 0.26 / 0.35 / 0.28. A 5th rung is
# infeasible -- past T[3]=455.7 the swap rate floors around 0.67-0.72 even as
# T->inf (delta_beta = 1/T[3] - 1/T[4] saturates at 1/T[3]).
MARKOVIAN_TEMPERATURES = [1.0, 8.409, 41.524, 455.653]

# Semi-Markovian (s23_semi / s23_semimarkovian): swap rates 0.27 / 0.30 / 0.34
# / 0.23, all 5 rungs reachable -- max temperature ~6.6x lower than the
# Markovian ladder's, reflecting the rougher posterior.
SEMIMARKOVIAN_TEMPERATURES = [1.0, 2.9, 7.361, 16.632, 69.256]

# Active ladder: point this at whichever one matches the currently active
# DEFAULT_DATA_DIRS above, and swap it by hand when switching datasets.
PT_TEMPERATURES = MARKOVIAN_TEMPERATURES
#PT_TEMPERATURES = SEMIMARKOVIAN_TEMPERATURES

# (estimate column, time column, acceptance column or None, sampler factory) per
# algorithm. Gibbs has no acceptance rate, so its acceptance column is None.
SAMPLERS = [
    ("Gibbs_Sampling", "Time_gibbs", None,
     lambda m, method, alpha: GibbsSampling(m, alpha=alpha)),
    ("Metropolis_Hastings", "Time_mh", "Acceptance_mh",
     lambda m, method, alpha: StandardMH(m, alpha=alpha)),
    ("Zanella", "Time_zanella", "Acceptance_zanella",
     lambda m, method, alpha: ZanellaMH(m, alpha=alpha, method=method)),
    ("Parallel_Tempering", "Time_pt", "Acceptance_pt",
     lambda m, method, alpha: ParallelTemperingMH(m, alpha=alpha, temperatures=PT_TEMPERATURES)),
]


def extract_file_info(filename: str) -> dict:
    """
    Parse a data/model/query filename into its parameters.

    Returns a dict with the ``unique_key`` (shared by the three files of one
    configuration) and the parsed ``meta`` fields, or ``None`` if no trailing ID
    can be found. ``merged`` is ``None`` when the filename carries no merged
    number (the ``s23`` layout).

    Examples:
        ``semi_nparents2_nzr04_zdr10_14_35.uai``          -> id 14, merged 35.
        ``simple_nparents1_nzr02_zdr05_ysize3_129.uai``   -> id 129, merged None.
    """
    # Prefer the two trailing numbers (ID, MERGED); fall back to a single ID.
    merged_match = re.search(r'_(\d+)_(\d+)(?:_query)?\.(?:uai|csv)$', filename)
    if merged_match:
        model_id, merged = int(merged_match.group(1)), int(merged_match.group(2))
    else:
        id_match = re.search(r'_(\d+)(?:_query)?\.(?:uai|csv)$', filename)
        if not id_match:
            return None
        model_id, merged = int(id_match.group(1)), None

    nparents_match = re.search(r'_nparents(\d+)', filename)
    nzr_match = re.search(r'_nzr(\d+)', filename)
    zdr_match = re.search(r'_zdr(\d+)', filename)
    nparents = int(nparents_match.group(1)) if nparents_match else None
    nzr = int(nzr_match.group(1)) if nzr_match else None
    zdr = int(zdr_match.group(1)) if zdr_match else None
    # Children cardinality is 3 only for the explicit 'ysize3' models.
    cardinality_children = 3 if 'ysize3' in filename else 2

    return {
        "unique_key": (nparents, nzr, zdr, cardinality_children, model_id, merged),
        "meta": {
            "index": model_id,
            "nparents": nparents,
            "nzr": nzr,
            "zdr": zdr,
            "merged": merged,
            "cardinality_children": cardinality_children,
        },
    }


def discover_sets(data_dirs: list, exclude_nparents: list, exclude_ids: list,
                  max_nparents=None) -> list:
    """
    Group the files across ``data_dirs`` into complete (model, data, query)
    triples. Missing directories are skipped with a warning.

    ``exclude_nparents`` and ``exclude_ids`` drop matching configurations; empty
    lists keep everything. ``max_nparents`` additionally drops any configuration
    whose nparents exceeds it (``None`` imposes no cap). The returned list is
    sorted for a deterministic order.
    """
    file_map = {}
    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            print(f"Warning: data directory not found, skipping: {data_dir}")
            continue
        for f in os.listdir(data_dir):
            if f.endswith('.uai'):
                file_type = 'model'
            elif f.endswith('_query.csv'):
                file_type = 'query'
            elif f.endswith('.csv'):
                file_type = 'data'
            else:
                continue

            info = extract_file_info(f)
            if not info:
                continue
            entry = file_map.setdefault(info['unique_key'], {'files': {}, 'meta': info['meta']})
            entry['files'][file_type] = os.path.join(data_dir, f)

    exclude_nparents = set(exclude_nparents)
    exclude_ids = set(exclude_ids)
    sets = []
    for val in file_map.values():
        files, meta = val['files'], val['meta']
        if not all(k in files for k in ('model', 'data', 'query')):
            continue
        if meta['nparents'] in exclude_nparents or meta['index'] in exclude_ids:
            continue
        if (max_nparents is not None and meta['nparents'] is not None
                and meta['nparents'] > max_nparents):
            continue
        sets.append({'meta': meta, **files})

    # None (missing merged) sorts consistently after any numeric merged value.
    sets.sort(key=lambda s: (s['meta']['index'],
                             (s['meta']['merged'] is None, s['meta']['merged'] or 0),
                             s['meta']['nparents']))
    return sets


def _query_value(inf: InferenceQueries, instance) -> list:
    """Estimate the requested causal quantity (PS/PN) for one query instance."""
    if instance.query == "PS":
        return inf.prob_sufficiency(instance.cause, instance.effect,
                                    true_false_cause=(1, 0), true_false_effect=(1, 0))
    if instance.query == "PN":
        return inf.prob_necessity(instance.cause, instance.effect,
                                  true_false_cause=(1, 0), true_false_effect=(1, 0))
    return None


def _checkpoint_iterations(iterations: int, every: int, burn_in: int) -> list:
    """
    Iteration counts at which to record results: every ``every`` steps plus the
    final step, keeping only those past the burn-in so each has usable samples.
    """
    points = set(range(every, iterations + 1, every))
    points.add(iterations)
    return sorted(p for p in points if p > burn_in)


def _acceptance_scalar(sampler):
    """
    Mean acceptance rate across the sampler's trainable variables, or ``None``
    for samplers without an acceptance concept (e.g. Gibbs).
    """
    rates = getattr(sampler, "acceptance_rate", None)
    if not rates:
        return None
    return float(np.mean(list(rates.values())))


def _run_with_checkpoints(sampler, endo_data: pd.DataFrame, burn_in: int, checkpoints: list,
                          outliers_removal: bool) -> dict:
    """
    Run one sampler up to the last checkpoint, snapshotting at each checkpoint an
    InferenceQueries over the post-burn-in models, the cumulative learning time
    (initialization plus steps, excluding inference) and the current acceptance
    rate. ``outliers_removal`` is forwarded to InferenceQueries. Returns
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
            # Slice makes a snapshot list, insulating it from later steps.
            results[step] = (InferenceQueries(sampler.model_evolution[burn_in:],
                                              outliers_removal=outliers_removal),
                             elapsed, _acceptance_scalar(sampler))
            pointer += 1
    return results


def process_model(job: dict) -> list:
    """
    Evaluate every sampler on a single model and return one wide result row per
    (checkpoint iteration, query). Designed to run inside a worker process.
    """
    meta = job['meta']
    burn_in, iterations, method, seed = job['burn_in'], job['iterations'], job['method'], job['seed']
    outliers_removal, alpha_value = job['outliers_removal'], job['alpha']
    checkpoints = _checkpoint_iterations(iterations, job['checkpoint_every'], burn_in)
    if not checkpoints:
        return []

    model = StructuralCausalModel.read(job['model'])
    data = pd.read_csv(job['data'], index_col=0).add_prefix('V')
    query = pd.read_csv(job['query'])
    endo_data = data[model.endogenous]

    # A scalar --alpha broadcasts to a flat Dirichlet prior over every latent
    # state; None lets each sampler fall back to its all-ones default.
    alpha = (None if alpha_value is None
             else {u: np.full(len(model.domains[u]), alpha_value) for u in model.exogenous})

    # Run each sampler from the same seeded random initialization for a fair
    # comparison; keep its per-checkpoint inference objects, times and rates.
    runs = {}
    for value_col, time_col, accept_col, factory in SAMPLERS:
        randomUtil.seed(seed)
        np.random.seed(seed)
        prior = model.randomize_factors(model.exogenous, allow_zero=False)
        runs[value_col] = (time_col, accept_col, _run_with_checkpoints(
            factory(prior, method, alpha), endo_data, burn_in, checkpoints, outliers_removal))

    base_row = {
        "Model_Index": meta['index'], "nparents": meta['nparents'], "nzr": meta['nzr'],
        "zdr": meta['zdr'], "merged": meta['merged'] if meta['merged'] is not None else "NA",
        "cardinality": meta['cardinality_children'], "seed": seed,
    }
    rows = []
    for iteration in checkpoints:
        for instance in query.itertuples(index=False):
            if instance.query not in ("PS", "PN"):
                continue
            row = dict(base_row)
            row.update({
                "Iteration": iteration, "cause": instance.cause, "effect": instance.effect,
                "query": instance.query, "Exact_Probability": [instance.low, instance.upp],
            })
            for value_col, (time_col, accept_col, per_iter) in runs.items():
                inf, elapsed, acceptance = per_iter[iteration]
                row[value_col] = _query_value(inf, instance)
                row[time_col] = elapsed
                if accept_col is not None:
                    row[accept_col] = acceptance
            rows.append(row)
    return rows


def _norm_val(value) -> str:
    """
    Canonical string form of an identity value so the same model matches whether
    it is read from meta (ints, ``None``) or from the CSV (floats, ``NaN``,
    ``"NA"``). Missing merged values all normalize to ``"NA"``.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    if isinstance(value, str):
        stripped = value.strip()
        return "NA" if stripped in ("", "NA", "nan", "NaN") else stripped
    if isinstance(value, (int, np.integer)) or (isinstance(value, float) and value.is_integer()):
        return str(int(value))
    return str(value)


def _model_key(meta: dict) -> tuple:
    """Normalized identity tuple of a model, matching the order of IDENTITY_COLUMNS."""
    values = (meta['index'], meta['nparents'], meta['nzr'], meta['zdr'],
              meta['merged'], meta['cardinality_children'])
    return tuple(_norm_val(v) for v in values)


def load_completed_keys(output_path: str) -> set:
    """
    Read the identity of every model already present in ``output_path`` so those
    models can be skipped on a restart. Returns an empty set if the file is
    missing, empty, or written with an incompatible set of columns.
    """
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return set()
    try:
        existing = pd.read_csv(output_path, usecols=IDENTITY_COLUMNS)
    except (ValueError, pd.errors.EmptyDataError):
        return set()
    return {tuple(_norm_val(v) for v in row)
            for row in existing[IDENTITY_COLUMNS].itertuples(index=False, name=None)}


def parse_args() -> argparse.Namespace:
    """Command-line configuration: iterations, burn-in, filters and parallelism."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", nargs="+", default=DEFAULT_DATA_DIRS,
                        help="One or more directories with model/data/query files.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory where Results_MH.csv is written.")
    parser.add_argument("--burn-in", type=int, default=DEFAULT_BURN_IN,
                        help="Number of initial recorded models discarded per run.")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                        help="Total MCMC steps per sampler.")
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY,
                        help="Record results every N steps (set equal to --iterations "
                             "for a single final measurement).")
    parser.add_argument("--exclude-nparents", type=int, nargs="*", default=[],
                        help="nparents values to skip (empty keeps all).")
    parser.add_argument("--max-nparents", type=int, default=DEFAULT_MAX_NPARENTS,
                        help="Drop models whose nparents exceeds this (default: 2). "
                             "Pass a larger value to include them.")
    parser.add_argument("--exclude-ids", type=int, nargs="*", default=[],
                        help="Model ids to skip (empty keeps all).")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Parallel worker processes (default: all CPU cores).")
    parser.add_argument("--zanella-method", default="sqrt", choices=("sqrt", "barker"),
                        help="Balancing function for the Zanella sampler.")
    parser.add_argument("--outliers-removal", action=argparse.BooleanOptionalAction,
                        default=DEFAULT_OUTLIERS_REMOVAL,
                        help="Remove outlier models in the inference "
                             "(use --no-outliers-removal to keep them).")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Dirichlet prior concentration broadcast to every latent "
                             "state (e.g. 0.5 -> all-0.5 prior). Default: all-ones.")
    parser.add_argument("--seed-offset", type=int, default=DEFAULT_SEED_OFFSET,
                        help="Added to every model's seed, giving an independent "
                             "replication of the whole experiment (e.g. 1000). Default "
                             "0 reproduces the original run bit-for-bit.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.burn_in >= args.iterations:
        raise ValueError("burn-in must be smaller than the total number of iterations")

    sets = discover_sets(args.data_dir, args.exclude_nparents, args.exclude_ids,
                         args.max_nparents)
    if not sets:
        raise SystemExit(f"No complete (model, data, query) triples found in {args.data_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, OUTPUT_FILENAME)

    # Seed each model by its fixed position (shifted by --seed-offset) so a
    # resumed run reproduces results, then skip the models already recorded in
    # the output file. A non-zero offset yields an independent replication:
    # same models and settings, different random realization.
    completed = load_completed_keys(output_path)
    jobs = [{**s, "burn_in": args.burn_in, "iterations": args.iterations,
             "checkpoint_every": args.checkpoint_every, "method": args.zanella_method,
             "outliers_removal": args.outliers_removal, "alpha": args.alpha,
             "seed": n + args.seed_offset}
            for n, s in enumerate(sets) if _model_key(s['meta']) not in completed]

    # None means "use every available core"; resolved explicitly here so the
    # actual worker count is known up front rather than left to the executor.
    workers = args.workers if args.workers is not None else os.cpu_count()

    print(f"Found {len(sets)} models | {len(sets) - len(jobs)} already done, {len(jobs)} to process "
          f"| burn-in={args.burn_in} iterations={args.iterations} workers={workers} "
          f"seed-offset={args.seed_offset}")
    if not jobs:
        print("Nothing to do; all models already processed.")
        return

    # Append each model's rows as it finishes so partial runs survive a stop.
    write_header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_model, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            meta = futures[future]['meta']
            tag = f"{meta['index']}_{meta['merged']}"
            try:
                rows = future.result()
                if rows:
                    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(
                        output_path, mode='a', header=write_header, index=False)
                    write_header = False
                print(f"[{done}/{len(jobs)}] model {tag} done ({len(rows)} rows)")
            except Exception as exc:  # keep going even if one model fails
                print(f"[{done}/{len(jobs)}] model {tag} FAILED: {exc}")

    print(f"Results in {output_path}")


if __name__ == "__main__":
    main()
