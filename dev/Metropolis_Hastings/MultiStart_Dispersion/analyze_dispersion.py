"""
Analyse the multi-start dispersion experiment: how much of each sampler's answer
is decided by where it started?

Reads ``output/dispersion_runs.csv`` (written by ``run_dispersion.py``) and, for
every (problem, sampler, checkpoint), decomposes the variance of the estimate
across the K starting points and R replicates into a between-start and a
within-start component. The ratio is an intraclass correlation:

    ICC = var_between / (var_between + var_within)

ICC near 0 means the chain forgets where it started -- good mixing. ICC near 1
means the answer is decided by the start -- mode-trapping. Because replicates
share a starting point but not a random stream, the two components separate
cleanly; a plain seed sweep would confound them.

Outputs (all under ``output/``):
    icc_by_checkpoint.csv   ICC per problem, sampler, checkpoint and endpoint
    icc_summary.csv         per-sampler summary at the final checkpoint
    forgetting_curve.png    mean ICC against iteration, one line per sampler
    icc_by_stratum.csv      final ICC broken down by confounder cardinality
    friedman_input.csv      long format for Friedman_test/friedman_analysis.py

Usage:  python analyze_dispersion.py [--output-dir DIR]
"""

import argparse
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))

SAMPLERS = ["Gibbs", "MH", "Zanella", "PT"]
TIME_COLUMNS = {"Gibbs": "Time_gibbs", "MH": "Time_mh",
                "Zanella": "Time_zanella", "PT": "Time_pt"}
PROBLEM_COLUMNS = ["Model_Index", "nparents", "nzr", "zdr", "merged",
                   "cardinality", "cause", "effect", "query"]


def icc_one_way(values: np.ndarray) -> tuple:
    """
    One-way random-effects variance decomposition of a starts x replicates grid.

    Uses the ANOVA estimator rather than the variance of the per-start means:
    that naive version is biased upward by ``var_within / n_replicates``, which
    at R=3 would inflate ICC noticeably and could manufacture apparent
    start-dependence out of pure Monte Carlo noise.

        MSB = R * var(start means, ddof=1)
        MSW = mean(within-start variances, ddof=1)
        var_between = max((MSB - MSW) / R, 0)      # clipped: the unbiased
        var_within  = MSW                          # estimator can go negative

    Args:
        values: Array of shape (n_starts, n_replicates) of estimates. Must have
            at least 2 starts and 2 replicates.

    Returns:
        ``(icc, var_between, var_within)``. ICC is ``nan`` when the total
        variance is zero (every run agreed exactly), since the ratio is then
        undefined rather than zero.

    Example:
        >>> grid = np.array([[0.1, 0.1], [0.9, 0.9]])   # starts disagree,
        >>> round(icc_one_way(grid)[0], 3)              # replicates do not
        1.0
    """
    n_starts, n_reps = values.shape
    if n_starts < 2 or n_reps < 2:
        return np.nan, np.nan, np.nan

    start_means = values.mean(axis=1)
    msb = n_reps * np.var(start_means, ddof=1)
    msw = float(np.mean(np.var(values, axis=1, ddof=1)))

    var_between = max((msb - msw) / n_reps, 0.0)
    var_within = msw
    total = var_between + var_within
    return (np.nan if total <= 0 else var_between / total), var_between, var_within


def icc_null(values: np.ndarray, rng: np.random.Generator, n_perm: int = 20) -> float:
    """
    Expected ICC for this grid when the starting point has no effect.

    ICC is a ratio of non-negative variance estimates, so with few starts and
    replicates it is biased upward: on pure noise at K=6, R=3 it averages about
    0.11 rather than 0. Reporting an observed ICC without this reference invites
    reading that bias as genuine start-dependence. Permuting the grid's values
    destroys any start structure while preserving the marginal spread, giving a
    per-cell null on exactly the same scale.

    Args:
        values: The observed (n_starts, n_replicates) grid.
        rng: Generator driving the permutations.
        n_perm: Permutations averaged over.

    Returns:
        Mean ICC across permutations, or ``nan`` if undefined throughout.
    """
    flat = values.ravel()
    draws = [icc_one_way(rng.permutation(flat).reshape(values.shape))[0]
             for _ in range(n_perm)]
    return float(np.nanmean(draws)) if np.any(np.isfinite(draws)) else np.nan


def rhat_from_components(var_between: float, var_within: float) -> float:
    """
    Gelman-Rubin-style potential scale reduction from the variance components.

    Derived from the decomposition rather than from within-chain traces: the
    recorded estimates are cumulative statistics over the whole post-burn-in
    history, not raw draws, so a conventional trace-based R-hat would be
    misleading. On this design R-hat reduces to ``sqrt(1 + between/within)``,
    a monotone transform of the ICC on the familiar scale where 1.0 is perfect
    agreement and > 1.1 is conventionally taken as non-convergence.
    """
    if not np.isfinite(var_within) or var_within <= 0:
        return np.nan
    return float(np.sqrt(1.0 + var_between / var_within))


def compute_icc_table(df: pd.DataFrame, starts: int, replicates: int) -> pd.DataFrame:
    """
    ICC for every (problem, sampler, checkpoint, endpoint) with a complete
    starts x replicates grid. Incomplete cells are skipped and counted.
    """
    records, skipped = [], 0
    group_cols = PROBLEM_COLUMNS + ["Iteration"]
    rng = np.random.default_rng(0)

    for keys, block in df.groupby(group_cols, sort=False):
        for sampler in SAMPLERS:
            for endpoint in ("low", "upp"):
                col = f"{sampler}_{endpoint}"
                grid = block.pivot_table(index="start_id", columns="replicate_id",
                                         values=col, aggfunc="first")
                if grid.shape != (starts, replicates) or grid.isna().any().any():
                    skipped += 1
                    continue
                arr = grid.to_numpy()
                icc, vb, vw = icc_one_way(arr)
                records.append({**dict(zip(group_cols, keys)),
                                "sampler": sampler, "endpoint": endpoint,
                                "icc": icc, "icc_null": icc_null(arr, rng),
                                "var_between": vb, "var_within": vw,
                                "rhat": rhat_from_components(vb, vw)})
    if skipped:
        print(f"  note: {skipped} cells skipped for an incomplete "
              f"{starts}x{replicates} grid")
    return pd.DataFrame(records)


def collapse_endpoints(icc: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce the two interval endpoints to one number per problem/sampler/checkpoint
    by taking the worse (larger) ICC. A sampler that is stable at one end and
    start-dependent at the other has still failed to explore the space.
    """
    keys = PROBLEM_COLUMNS + ["Iteration", "sampler"]
    return (icc.groupby(keys, sort=False)
               .agg(icc=("icc", "max"), icc_null=("icc_null", "max"),
                    rhat=("rhat", "max"))
               .reset_index())


def matched_time_icc(df: pd.DataFrame, starts: int, replicates: int) -> pd.DataFrame:
    """
    ICC recomputed at a common wall-clock budget rather than a common iteration
    count.

    The samplers differ ~19x in cost per iteration, so an iteration-matched
    comparison hands the slowest one far more compute. The budget is the
    smallest per-sampler median total runtime, and each run's estimate is
    interpolated to it before the decomposition.
    """
    budgets = []
    for sampler in SAMPLERS:
        totals = df.groupby(["Model_Index", "start_id", "replicate_id"])[
            TIME_COLUMNS[sampler]].max()
        budgets.append(totals.median())
    budget = float(np.nanmin(budgets))

    records = []
    for keys, block in df.groupby(PROBLEM_COLUMNS, sort=False):
        for sampler in SAMPLERS:
            grid = np.full((starts, replicates), np.nan)
            for (s, r), run in block.groupby(["start_id", "replicate_id"]):
                run = run.sort_values("Iteration")
                t = run[TIME_COLUMNS[sampler]].to_numpy(dtype=float)
                v = run[f"{sampler}_low"].to_numpy(dtype=float)
                ok = np.isfinite(t) & np.isfinite(v)
                if ok.sum() >= 2 and s < starts and r < replicates:
                    grid[s, r] = np.interp(budget, t[ok], v[ok])
            if np.isnan(grid).any():
                continue
            icc, vb, vw = icc_one_way(grid)
            records.append({**dict(zip(PROBLEM_COLUMNS, keys)),
                            "sampler": sampler, "icc_matched_time": icc})
    out = pd.DataFrame(records)
    out.attrs["budget_seconds"] = budget
    return out


def plot_forgetting_curves(collapsed: pd.DataFrame, path: str) -> None:
    """Mean ICC against iteration, one line per sampler."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = (collapsed.groupby(["Iteration", "sampler"])["icc"]
                      .mean().unstack("sampler"))
    fig, ax = plt.subplots(figsize=(9, 6))
    for sampler in SAMPLERS:
        if sampler in curve:
            ax.plot(curve.index, curve[sampler], marker="o", label=sampler)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean ICC (start-dependence)")
    ax.set_title("Forgetting curves: how fast does each sampler shed its start?")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=os.path.join(_HERE, "output"))
    parser.add_argument("--runs-file", default=None,
                        help="Defaults to <output-dir>/dispersion_runs.csv")
    args = parser.parse_args()

    runs_path = args.runs_file or os.path.join(args.output_dir, "dispersion_runs.csv")
    if not os.path.exists(runs_path):
        raise SystemExit(f"no results at {runs_path}; run run_dispersion.py first")

    df = pd.read_csv(runs_path).drop_duplicates()
    starts = int(df.start_id.nunique())
    replicates = int(df.replicate_id.nunique())
    print(f"Loaded {len(df):,} rows | {df.Model_Index.nunique()} models "
          f"| {starts} starts x {replicates} replicates "
          f"| {df.Iteration.nunique()} checkpoints")

    icc = compute_icc_table(df, starts, replicates)
    if icc.empty:
        raise SystemExit("no complete starts x replicates grids found")
    icc.to_csv(os.path.join(args.output_dir, "icc_by_checkpoint.csv"), index=False)

    collapsed = collapse_endpoints(icc)
    final_iter = int(collapsed.Iteration.max())
    final = collapsed[collapsed.Iteration == final_iter]

    summary = (final.groupby("sampler")
                    .agg(problems=("icc", "size"), mean_icc=("icc", "mean"),
                         median_icc=("icc", "median"),
                         mean_null_icc=("icc_null", "mean"),
                         # Median, not mean: R-hat diverges when within-start
                         # variance approaches zero, and a single such cell
                         # would dominate an average.
                         median_rhat=("rhat", "median"),
                         frac_rhat_above_1_1=("rhat", lambda s: float((s > 1.1).mean())))
                    .reindex(SAMPLERS).round(4))
    summary["excess_over_null"] = (summary.mean_icc - summary.mean_null_icc).round(4)
    summary.to_csv(os.path.join(args.output_dir, "icc_summary.csv"))
    print(f"\n=== Start-dependence at iteration {final_iter} (lower = better mixing) ===")
    print(summary.to_string())
    print("\nICC is biased upward at small K,R, so compare mean_icc against "
          "mean_null_icc:\nexcess_over_null near 0 means the start is forgotten.")

    curve = (collapsed.groupby(["Iteration", "sampler"])["icc"].mean()
                      .unstack("sampler").reindex(columns=SAMPLERS).round(4))
    print("\n=== Forgetting curves: mean ICC by checkpoint ===")
    print(curve.to_string())
    plot_forgetting_curves(collapsed, os.path.join(args.output_dir, "forgetting_curve.png"))

    # Friedman across problems, mirroring the existing RMSE pipeline.
    wide = final.pivot_table(index=PROBLEM_COLUMNS, columns="sampler", values="icc")
    wide = wide.dropna()
    if len(wide) >= 2 and wide.shape[1] == len(SAMPLERS):
        from scipy.stats import friedmanchisquare
        test = friedmanchisquare(*[wide[s].values for s in SAMPLERS])
        print(f"\n=== Friedman on ICC (n={len(wide)} problems) ===")
        print("mean rank (1 = least start-dependent):",
              wide.rank(axis=1).mean().round(3).to_dict())
        print(f"chi2={test.statistic:.4f}  p={test.pvalue:.4g}")

    long = (final.assign(problem=final[PROBLEM_COLUMNS].astype(str).agg("|".join, axis=1))
                 .rename(columns={"sampler": "method"})[["problem", "method", "icc"]])
    long.to_csv(os.path.join(args.output_dir, "friedman_input.csv"), index=False)

    # Breakdown by confounder cardinality, if the manifest is alongside.
    manifest_path = os.path.join(args.output_dir, "model_manifest.csv")
    if os.path.exists(manifest_path):
        manifest = pd.read_csv(manifest_path)[["Model_Index", "merged_cardinality"]]
        merged = final.merge(manifest.drop_duplicates("Model_Index"),
                             on="Model_Index", how="left")
        if merged.merged_cardinality.notna().any():
            # qcut needs several distinct cardinalities; with only a few, group
            # on the raw value so the breakdown is not silently empty.
            distinct = merged.merged_cardinality.nunique()
            merged["stratum"] = (pd.qcut(merged.merged_cardinality, 4,
                                         duplicates="drop")
                                 if distinct >= 4 else merged.merged_cardinality)
            by_stratum = (merged.groupby(["stratum", "sampler"], observed=True)["icc"]
                                .mean().unstack("sampler")
                                .reindex(columns=SAMPLERS).round(4))
            by_stratum.to_csv(os.path.join(args.output_dir, "icc_by_stratum.csv"))
            print("\n=== Mean ICC by confounder cardinality ===")
            print(by_stratum.to_string())

    matched = matched_time_icc(df, starts, replicates)
    if not matched.empty:
        matched.to_csv(os.path.join(args.output_dir, "icc_matched_time.csv"),
                       index=False)
        print(f"\n=== Mean ICC at a matched {matched.attrs['budget_seconds']:.3g}s "
              f"budget ===")
        print(matched.groupby("sampler")["icc_matched_time"].mean()
                     .reindex(SAMPLERS).round(4).to_string())

    print(f"\nWrote results to {args.output_dir}/")


if __name__ == "__main__":
    main()
