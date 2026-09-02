"""Build a semi-Markovian dataset by merging exogenous variables and solving it
with the exact LP solver (``exact_credal_scm``) -- the pure-Python analogue of
``local/semi_markovian_export/generate_semi_markovian.py`` (which shells out to
credici's JVM).

For each Markovian model ``<base>`` in the input directory it enumerates
exogenous-variable merge groups, merges each group into a single shared
confounder (producing a semi-Markovian model), and solves the model's queries
with :class:`ExactCredalSCM`. Feasible results are written to the output folder.

Merge groups (per the requested rule): for a model with ``E`` exogenous
variables, take every combination of sizes ``2, 3, ..., E``. So a "2-parent"
model (3 exogenous) gets every pair and the single triple; a "3-parent" model
(4 exogenous) gets every pair, every triple and the single all-four merge.

Naming: the merged group is tagged by the 1-based POSITIONS of its members among
the model's index-sorted exogenous variables -- e.g. merging the 2nd and 3rd
exogenous appends ``_23``; merging the 1st, 2nd and 3rd appends ``_123``. The
``simple_`` prefix becomes ``semi_``, mirroring the original tool.

Feasibility / resampling: a merged model often cannot reproduce the original
empirical distribution (its confounder can't realise the observed joint) -- the
solver raises :class:`NoFeasibleSolution`, exactly as credici throws
``NoFeasibleSolutionException``. On that, the dataset is resampled from a fresh
random distribution over the endogenous joint state space and the solve retried,
up to ``MAX_RESAMPLE_ATTEMPTS`` times; if still infeasible the merge is skipped.

Usage:
    python generate_semimarkovian.py [--input DIR] [--output DIR]
                                     [--workers N] [-n N] [--all]
"""

from __future__ import annotations

import argparse
import glob
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exact_credal_scm import ExactCredalSCM, NoFeasibleSolution  # noqa: E402
from bcause.models.cmodel import StructuralCausalModel  # noqa: E402

DEFAULT_INPUT = "/Users/antoniogonzalezalves/Documents/s23"
DEFAULT_OUTPUT = "/Users/antoniogonzalezalves/Documents/s23_semimarkovian"
QUERY_METHOD = {"PS": "prob_sufficiency", "PN": "prob_necessity",
                "PNS": "prob_necessity_sufficiency"}
OUT_COLUMNS = ["cause", "effect", "low", "query", "tinfer", "tlearn", "upp"]
EXCLUSIONS = ("_ysize3",)                 # 3-state effects: skip (as the original did)
MAX_RESAMPLE_ATTEMPTS = 25                 # resample budget before giving up a merge
                                           # (structurally-infeasible merges never
                                           # resolve, so a large budget is wasted)
DIRICHLET_ALPHA = 0.5                      # concentration of the resampling distribution


# ------------------------------------------------------------------ merging

def merge_model(m: StructuralCausalModel, group: list) -> StructuralCausalModel:
    """Cascade-merge ``group`` exogenous variables into one shared confounder."""
    result = m.merge_exogenous(group[0], group[1])
    current = group[1] + group[0]                     # merge_exogenous names it str(U)+str(V)
    for nxt in group[2:]:
        result = result.merge_exogenous(current, nxt)
        current = nxt + current
    return result


def normalize_exogenous_names(m: StructuralCausalModel) -> StructuralCausalModel:
    """Rename exogenous nodes to canonical ``V{n_endo+i}`` for a stable UAI round-trip."""
    n = len(m.endogenous)
    mapping = {e: f"V{n + i}" for i, e in enumerate(m.exogenous) if e != f"V{n + i}"}
    return cast(StructuralCausalModel, m.rename_vars(mapping)) if mapping else m


def merge_configs(m: StructuralCausalModel):
    """(group, suffix) pairs: every combination of sizes 2..E of the exogenous vars.

    ``suffix`` are the 1-based positions of the merged variables among the
    index-sorted exogenous (e.g. 2nd+3rd -> "23").
    """
    exo = sorted(m.exogenous, key=lambda v: int(v[1:]))
    pos = {v: i + 1 for i, v in enumerate(exo)}
    configs = []
    for size in range(2, len(exo) + 1):
        for group in itertools.combinations(exo, size):
            configs.append((list(group), "".join(str(pos[v]) for v in group)))
    return configs


def semi_name(base: str, suffix: str) -> str:
    stem = base[len("simple_"):] if base.startswith("simple_") else base
    return f"semi_{stem}_{suffix}"


# ------------------------------------------------------------------ resampling

def random_dataset(template: pd.DataFrame, model: StructuralCausalModel,
                   rng: np.random.Generator) -> pd.DataFrame:
    """A dataset drawn from a fresh random distribution over the endogenous joint.

    Same shape/columns as ``template``; values are sampled from a
    ``Dirichlet(DIRICHLET_ALPHA)`` distribution over the full endogenous joint
    state space, so successive seeds explore widely separated distributions (a
    broad search for a feasible one), exactly like the credici-based tool.
    """
    endogenous = list(template.columns)
    states = [model.domains[v] for v in endogenous]
    joint = np.array(list(itertools.product(*states)))
    dist = rng.dirichlet(np.full(len(joint), DIRICHLET_ALPHA))
    rows = rng.choice(len(joint), size=len(template), p=dist)
    return pd.DataFrame(joint[rows], columns=endogenous, index=template.index)


# ------------------------------------------------------------------ per job

def solve_queries(inf: ExactCredalSCM, data: pd.DataFrame, queries: list):
    """All query rows for one (feasible) dataset, or raise NoFeasibleSolution.

    Reuses ``inf`` (its model / deterministic-response cache) and only swaps the
    dataset, so resampling attempts don't re-pay the model setup cost.
    """
    inf._data = data
    t0 = time.perf_counter()
    if not inf.is_feasible():
        raise NoFeasibleSolution("model cannot reproduce the empirical law")
    tlearn = int((time.perf_counter() - t0) * 1000)
    rows = []
    for cause, effect, qtype in queries:
        t1 = time.perf_counter()
        low, upp = getattr(inf, QUERY_METHOD[qtype])(cause, effect)
        rows.append({"cause": cause, "effect": effect, "low": low, "query": qtype,
                     "tinfer": int((time.perf_counter() - t1) * 1000),
                     "tlearn": tlearn, "upp": upp})
    return rows


def process_job(job: dict) -> dict:
    """Merge, solve (with resampling), and write one (model, merge-group) result."""
    name = semi_name(job["base_name"], job["suffix"])
    try:
        m = StructuralCausalModel.read(job["uai"])
        merged = normalize_exogenous_names(merge_model(m, job["group"]))
        var_order = sorted(merged.endogenous + merged.exogenous, key=lambda v: int(v[1:]))
        # Round-trip through UAI so the merged confounder gets clean contiguous
        # integer states (its in-memory labels are strings like '3_1'); this makes
        # the per-solve domain fix cheap and matches what credici would read.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = os.path.join(td, "m.uai")
            merged.save(tmp, var_order=var_order)
            model = StructuralCausalModel.read(tmp)
    except Exception as e:
        return {"name": name, "status": "error", "message": f"merge: {type(e).__name__}: {e}"}

    template = pd.read_csv(job["csv"], index_col=0).add_prefix("V")[model.endogenous]
    inf = ExactCredalSCM(model, template)          # domain fix + response cache: once
    data = template
    for attempt in range(MAX_RESAMPLE_ATTEMPTS + 1):
        try:
            rows = solve_queries(inf, data, job["queries"])
        except NoFeasibleSolution:
            data = random_dataset(template, model, np.random.default_rng(attempt))
            continue
        except Exception as e:
            return {"name": name, "status": "error", "message": f"solve: {type(e).__name__}: {e}"}
        out = Path(job["output_dir"])          # success: write model, data, results
        model.save(str(out / f"{name}.uai"), var_order=var_order)
        stripped = data.copy()
        stripped.columns = [c[1:] for c in stripped.columns]      # credici csv layout
        stripped.to_csv(out / f"{name}.csv", index=True)
        pd.DataFrame(rows, columns=OUT_COLUMNS).to_csv(
            out / f"{name}_query.csv", index=False)
        return {"name": name, "status": "solved", "attempts": attempt + 1}
    return {"name": name, "status": "unsolved", "message": "infeasible after resampling"}


# ------------------------------------------------------------------ driver

def build_jobs(input_dir: str, output_dir: str, n_models):
    jobs = []
    uais = sorted(glob.glob(os.path.join(input_dir, "*.uai")))
    picked = 0
    for uai in uais:
        base = uai[:-4]
        name = os.path.basename(base)
        if any(x in name for x in EXCLUSIONS):
            continue
        if not (os.path.exists(base + ".csv") and os.path.exists(base + "_query.csv")):
            continue
        q = pd.read_csv(base + "_query.csv")
        queries = [(r["cause"], r["effect"], r["query"]) for _, r in q.iterrows()]
        try:
            m = StructuralCausalModel.read(uai)
        except Exception:
            continue
        for group, suffix in merge_configs(m):
            sname = semi_name(name, suffix)
            if os.path.exists(os.path.join(output_dir, f"{sname}_query.csv")):
                continue                    # resume: already done
            jobs.append({"uai": uai, "csv": base + ".csv", "base_name": name,
                         "group": group, "suffix": suffix, "queries": queries,
                         "output_dir": output_dir})
        picked += 1
        if n_models is not None and picked >= n_models:
            break
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--all", action="store_true", help="process every model")
    grp.add_argument("-n", "--n-models", type=int, default=None,
                     help="process only the first N models (default: all)")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    n_models = None if args.all else args.n_models
    jobs = build_jobs(args.input, args.output, n_models)
    print(f"Input : {args.input}\nOutput: {args.output}\n{len(jobs)} merge jobs "
          f"| workers={args.workers}\n")
    if not jobs:
        print("Nothing to do.")
        return

    t0 = time.perf_counter()
    solved = unsolved = errored = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_job, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            solved += res["status"] == "solved"
            unsolved += res["status"] == "unsolved"
            errored += res["status"] == "error"
            if res["status"] != "solved":
                print(f"  [{i}/{len(jobs)}] {res['name']}: {res['status']} "
                      f"{res.get('message', '')}")
            if i % 500 == 0:
                print(f"  ... {i}/{len(jobs)} done (solved={solved})")

    print(f"\nDone in {time.perf_counter()-t0:.0f}s. "
          f"solved={solved} unsolved={unsolved} errored={errored}")
    print(f"Output dataset: {args.output}")


if __name__ == "__main__":
    main()
