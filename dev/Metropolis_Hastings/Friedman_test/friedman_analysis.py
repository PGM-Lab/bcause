"""
Friedman test of the four causal-effect samplers, per model family and per
number of parents.

Stage 1 of a two-stage pipeline (this script prepares the data, then invokes
``run_friedman_test.R`` unless --no-r is passed). For the chosen dataset it
builds one long-format table per analysis group:

    all         every model in the dataset, combined
    nparentsN   only the models with nparents == N (one group per value present)

Each results CSV is wide: one row per (model, cause, query) problem instance x
checkpoint iteration, with Gibbs_Sampling / Metropolis_Hastings / Zanella /
Parallel_Tempering each holding a string-encoded [low, upp] interval estimate.
exreport::expCreate() instead needs one row per (method, problem) with a single
numeric output column, and no duplicate or unbalanced (method, problem) cells.

For every group this script drops exact duplicate rows, keeps only the final
checkpoint (Iteration == FINAL_ITERATION -- one clean, non-autocorrelated
snapshot per problem rather than 18 autocorrelated ones), parses the interval
strings, computes each sampler's RMSE against Exact_Probability (same formula
as create_graphs.py), and melts the four RMSE columns into long format.

``merged`` is always part of the problem identity. It is all-NaN in the
Markovian data (so it contributes a constant and changes nothing), but in the
semi-Markovian data the same (Model_Index, nparents, nzr, zdr, cardinality,
cause, effect, query) tuple can carry several distinct merged-exogenous
variants, which would otherwise collide into a single "problem" label.

Usage:
    python friedman_analysis.py --dataset markovian
    python friedman_analysis.py --dataset semimarkovian
    python friedman_analysis.py --dataset all
"""

import argparse
import ast
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(HERE, "output")
R_SCRIPT = os.path.join(HERE, "run_friedman_test.R")

# Each dataset is a wide results CSV produced by Evaluation_MH_Variants.py.
DATASETS = {
    "markovian": "/Users/antoniogonzalezalves/Documents/resultados_markovian/Results_MH.csv",
    "semimarkovian": "/Users/antoniogonzalezalves/Documents/resultados_markovian/Results_MH_semi.csv",
}

FINAL_ITERATION = 10000
IDENTITY_COLUMNS = ["Model_Index", "nparents", "nzr", "zdr", "merged", "cardinality",
                    "cause", "effect", "query"]
ALGORITHMS = ["Gibbs_Sampling", "Metropolis_Hastings", "Zanella", "Parallel_Tempering"]
EXACT_COLUMN = "Exact_Probability"


def parse_interval(value) -> tuple:
    """Parse a stored '[low, upp]' interval, tolerating a literal 'nan' bound."""
    try:
        return tuple(ast.literal_eval(value.replace("nan", "float('nan')")))
    except (ValueError, SyntaxError, AttributeError):
        return (np.nan, np.nan)


def rmse(low: pd.Series, upp: pd.Series, exact_low: pd.Series, exact_upp: pd.Series) -> pd.Series:
    """Interval RMSE of (low, upp) against the exact bounds."""
    return np.sqrt(((low - exact_low) ** 2 + (upp - exact_upp) ** 2) / 2)


def build_group_input(df: pd.DataFrame, out_dir: str, label: str) -> int:
    """
    Reduce one already-nparents-filtered frame to the long-format table and
    write it to ``out_dir/friedman_input.csv``. Returns the problem count.
    """
    df = df[df["Iteration"] == FINAL_ITERATION].copy()

    exact = df[EXACT_COLUMN].apply(parse_interval)
    df["exact_low"] = exact.apply(lambda x: x[0])
    df["exact_upp"] = exact.apply(lambda x: x[1])

    df["problem"] = df[IDENTITY_COLUMNS].astype(str).agg("_".join, axis=1)
    assert not df["problem"].duplicated().any(), \
        f"[{label}] duplicate problem keys -- identity columns are insufficient"

    for algo in ALGORITHMS:
        bounds = df[algo].apply(parse_interval)
        low = bounds.apply(lambda x: x[0])
        upp = bounds.apply(lambda x: x[1])
        df[algo + "_rmse"] = rmse(low, upp, df["exact_low"], df["exact_upp"])

    # Drop problems whose exact bounds are undefined (NaN) -- all four of that
    # problem's method-rows disappear together, preserving balance.
    rmse_cols = [a + "_rmse" for a in ALGORITHMS]
    valid = df.dropna(subset=rmse_cols, how="any")
    n_dropped = len(df) - len(valid)

    long_df = valid.melt(id_vars=["problem"], value_vars=rmse_cols,
                         var_name="method", value_name="rmse")
    long_df["method"] = long_df["method"].str.replace("_rmse", "", regex=False)

    # Every problem must contribute exactly one row per algorithm. Counting rows
    # (not distinct methods) also catches a silent identity collision in which
    # two source rows share a method.
    assert not long_df.duplicated(["problem", "method"]).any(), \
        f"[{label}] duplicate (problem, method) pairs -- identity columns are insufficient"
    counts = long_df.groupby("problem")["method"].count()
    incomplete = counts[counts != len(ALGORITHMS)].index
    if len(incomplete) > 0:
        print(f"  [{label}] dropping {len(incomplete)} incomplete problem(s)")
        long_df = long_df[~long_df["problem"].isin(incomplete)]

    os.makedirs(out_dir, exist_ok=True)
    long_df.to_csv(os.path.join(out_dir, "friedman_input.csv"), index=False)

    n_problems = long_df["problem"].nunique()
    print(f"  [{label}] {n_problems} problems x {len(ALGORITHMS)} methods = {len(long_df)} rows"
          + (f" ({n_dropped} dropped: undefined exact bounds)" if n_dropped else ""))
    return n_problems


def prepare_dataset(dataset: str, input_csv: str) -> list:
    """Build the 'all' group plus one group per nparents value present."""
    print(f"\n=== {dataset}: {input_csv} ===")
    df = pd.read_csv(input_csv)
    n_read = len(df)
    df = df.drop_duplicates()
    print(f"Read {n_read} rows -> {len(df)} after dropping full-row duplicates")

    dataset_dir = os.path.join(OUTPUT_ROOT, dataset)
    # Rebuild from scratch so a rerun never mixes in stale groups.
    if os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)

    groups = [("all", df)]
    for n in sorted(df["nparents"].dropna().unique()):
        groups.append((f"nparents{int(n)}", df[df["nparents"] == n]))

    built = []
    for label, group_df in groups:
        build_group_input(group_df, os.path.join(dataset_dir, label), label)
        built.append(label)
    return built


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="all", choices=list(DATASETS) + ["all"],
                        help="Which results file to analyse (default: all of them).")
    parser.add_argument("--no-r", action="store_true",
                        help="Only prepare the inputs; skip the R test stage.")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]

    for dataset in datasets:
        prepare_dataset(dataset, DATASETS[dataset])
        if args.no_r:
            continue
        print(f"--- running Friedman tests for {dataset} ---")
        result = subprocess.run(["Rscript", R_SCRIPT, dataset], cwd=HERE)
        if result.returncode != 0:
            sys.exit(f"R stage failed for {dataset} (exit {result.returncode})")


if __name__ == "__main__":
    main()
