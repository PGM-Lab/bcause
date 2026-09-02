"""
Draw convergence graphs from the sampler evaluation results (Results_MH.csv).

Three figures are produced, each comparing the four samplers:
 - RMSE  vs Iteration : accuracy against the exact bounds as the chain grows.
 - Time  vs Iteration : cumulative learning time as the chain grows.
 - RMSE  vs Time       : the accuracy/cost trade-off trajectory.

The results can be filtered (and manually excluded) by model id, number of
parents and the width of the exact interval (upp - low) before plotting. The
filtered rows underlying the plots are also written to graph_data.csv.
"""

import argparse
import ast
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Change the path with the results file you want to use.
DEFAULT_INPUT = "/Users/antoniogonzalezalves/Documents/prueba_mh/results_MH_1000itet.csv"

# ---------------------------------------------------------------------------
# Editable configuration. Change these to set the defaults without using the
# command line; the matching command-line flag still overrides any value here.
# ---------------------------------------------------------------------------
CONFIG_IDS = []          # model ids to keep, e.g. [7, 43]; empty list keeps all
CONFIG_EXCLUDE_IDS = []  # model ids to drop, e.g. [12, 19]; empty list drops none
CONFIG_NPARENTS = []     # nparents values to keep, e.g. [2]; empty list keeps all
CONFIG_MIN_WIDTH = 0.1   # keep exact-interval widths >= this, e.g. 0.2; None = no limit
CONFIG_MAX_WIDTH = None  # keep exact-interval widths <= this, e.g. 0.5; None = no limit

# Each sampler bound to a fixed (time column, legend label, colour, marker). The
# colour follows the algorithm, never its rank, so filtering models never
# repaints a series. Palette is the Wong/colourblind-safe set (validated: worst
# adjacent CVD dE 37); the distinct markers give a second, colour-free cue.
ALGORITHMS = {
    "Gibbs_Sampling": ("Time_gibbs", "Gibbs", "#0173B2", "o"),
    "Metropolis_Hastings": ("Time_mh", "MH-Uniform", "#DE8F05", "s"),
    "Zanella": ("Time_zanella", "MH-Zanella", "#029E73", "^"),
    "Parallel_Tempering": ("Time_pt", "MH-Parallel Tempering", "#D55E00", "D"),
}
EXACT_COLUMN = "Exact_Probability"
WIDTH_TOL = 0  # tolerance so float rounding never drops a boundary-width query
MAX_XTICKS = 20  # cap on x-axis ticks so densely checkpointed runs stay readable


def parse_interval(value) -> list:
    """Parse a stored ``[low, upp]`` interval (string or sequence) into a list."""
    try:
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, str):
            return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass
    return [np.nan, np.nan]


def load_results(input_path: str, ids: list, exclude_ids: list, nparents: list,
                 min_width: float, max_width: float) -> tuple:
    """
    Read the results CSV, apply the id/exclude-id/nparents/interval-width
    filters (``None`` or empty keeps all / drops none) and compute the interval
    RMSE of every sampler present against the exact bounds. Returns the
    enriched, filtered frame and the list of algorithm columns found.

    ``min_width``/``max_width`` keep only queries whose exact interval width
    (``upp - low``) is >= min_width and <= max_width respectively.
    """
    df = pd.read_csv(input_path)

    if ids:
        df = df[df["Model_Index"].isin(ids)]
    if exclude_ids:
        df = df[~df["Model_Index"].isin(exclude_ids)]
    if nparents:
        df = df[df["nparents"].isin(nparents)]
    if df.empty:
        raise SystemExit("No rows left after filtering; relax --ids / --exclude-ids / --nparents.")

    algorithms = [col for col in ALGORITHMS if col in df.columns]
    if not algorithms:
        raise SystemExit("None of the expected sampler columns are present in the CSV.")

    exact = df[EXACT_COLUMN].apply(parse_interval)
    exact_low = exact.apply(lambda x: x[0])
    exact_upp = exact.apply(lambda x: x[1])

    # Filter by the width of the exact interval, keeping the parsed bounds aligned.
    width = exact_upp - exact_low
    if min_width is not None:
        mask = width >= min_width - WIDTH_TOL
        df, exact_low, exact_upp, width = df[mask], exact_low[mask], exact_upp[mask], width[mask]
    if max_width is not None:
        mask = width <= max_width + WIDTH_TOL
        df, exact_low, exact_upp = df[mask], exact_low[mask], exact_upp[mask]
    if df.empty:
        raise SystemExit("No rows left after filtering; relax --min-width / --max-width.")

    # Interval RMSE: root mean square error of the lower and upper bounds.
    for algo in algorithms:
        bounds = df[algo].apply(parse_interval)
        low = bounds.apply(lambda x: x[0])
        upp = bounds.apply(lambda x: x[1])
        df[algo + "_RMSE"] = np.sqrt(((low - exact_low) ** 2 + (upp - exact_upp) ** 2) / 2)

    return df, algorithms


def aggregate(df: pd.DataFrame, algo: str) -> pd.DataFrame:
    """Mean RMSE and cumulative time per iteration for one algorithm, iteration-ordered."""
    time_col = ALGORITHMS[algo][0]
    return (df.groupby("Iteration")
              .agg(RMSE=(algo + "_RMSE", "mean"), Time=(time_col, "mean"))
              .reset_index()
              .sort_values("Iteration"))


def line_plot(curves: dict, x: str, y: str, xlabel: str, ylabel: str, title: str,
              save_path: str, xticks=None):
    """
    Draw one comparison line plot from ``{algo: aggregated_frame}`` using each
    algorithm's fixed colour and marker, then save it.
    """
    fig, ax = plt.subplots(figsize=(11, 8))
    for algo, frame in curves.items():
        _, label, colour, marker = ALGORITHMS[algo]
        ax.plot(frame[x], frame[y], color=colour, marker=marker, label=label,
                linewidth=2.5, markersize=9)

    if xticks is not None:
        # Thin the candidate ticks to at most MAX_XTICKS evenly-spaced values and
        # slant the labels so densely checkpointed runs do not overlap.
        ticks = list(xticks)
        if len(ticks) > MAX_XTICKS:
            ticks = ticks[::int(np.ceil(len(ticks) / MAX_XTICKS))]
        ax.set_xticks(ticks)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(title, fontsize=20, fontweight="bold", pad=16)
    ax.set_xlabel(xlabel, fontsize=15, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=15, labelpad=10)
    ax.legend(title="Algorithm", title_fontsize=13, fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)


def parse_args() -> argparse.Namespace:
    """Command-line configuration: input file, id/nparents filters and output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to Results_MH.csv.")
    parser.add_argument("--ids", type=int, nargs="*", default=CONFIG_IDS,
                        help="Keep only these model ids (empty keeps all).")
    parser.add_argument("--exclude-ids", type=int, nargs="*", default=CONFIG_EXCLUDE_IDS,
                        help="Drop these model ids (empty drops none).")
    parser.add_argument("--nparents", type=int, nargs="*", default=CONFIG_NPARENTS,
                        help="Keep only these nparents values (empty keeps all).")
    parser.add_argument("--min-width", type=float, default=CONFIG_MIN_WIDTH,
                        help="Keep only queries with exact interval width (upp - low) "
                             ">= this value.")
    parser.add_argument("--max-width", type=float, default=CONFIG_MAX_WIDTH,
                        help="Keep only queries with exact interval width (upp - low) "
                             "<= this value.")
    parser.add_argument("--save-dir", default=None,
                        help="Where to write the PNGs (default: the input's directory).")
    parser.add_argument("--no-show", action="store_true",
                        help="Save the figures without opening a window.")
    return parser.parse_args()


def main():
    args = parse_args()

    df, algorithms = load_results(args.input, args.ids, args.exclude_ids, args.nparents,
                                  args.min_width, args.max_width)
    print(df[["Model_Index", "nparents", "nzr", "zdr", "merged", "cardinality"]].drop_duplicates()
          .to_string(index=False))

    curves = {algo: aggregate(df, algo) for algo in algorithms}
    xticks = sorted(df["Iteration"].unique())

    filters = []
    if args.ids:
        filters.append(f"ids={','.join(map(str, args.ids))}")
    if args.exclude_ids:
        filters.append(f"exclude_ids={','.join(map(str, args.exclude_ids))}")
    if args.nparents:
        filters.append(f"nparents={','.join(map(str, args.nparents))}")
    if args.min_width is not None:
        filters.append(f"width>={args.min_width:g}")
    if args.max_width is not None:
        filters.append(f"width<={args.max_width:g}")
    suffix = f" ({'; '.join(filters)})" if filters else ""

    save_dir = args.save_dir or os.path.dirname(args.input)
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, "graph_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved filtered results ({len(df)} rows) to {csv_path}")

    line_plot(curves, "Iteration", "RMSE", "Iteration", "Mean RMSE",
              "RMSE vs Iteration" + suffix,
              os.path.join(save_dir, "rmse_vs_iteration.png"), xticks=xticks)
    line_plot(curves, "Iteration", "Time", "Iteration", "Mean cumulative time (s)",
              "Time vs Iteration" + suffix,
              os.path.join(save_dir, "time_vs_iteration.png"), xticks=xticks)
    line_plot(curves, "Time", "RMSE", "Mean cumulative time (s)", "Mean RMSE",
              "RMSE vs Time" + suffix,
              os.path.join(save_dir, "rmse_vs_time.png"))

    print(f"Saved 3 figures to {save_dir}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
