"""Recompute every s23 query with the exact LP solver into a new dataset folder.

For each model ``<base>`` in the input directory that has the triple
``<base>.uai`` + ``<base>.csv`` + ``<base>_query.csv``, this:

  1. copies the model (.uai) and data (.csv) to the output folder unchanged, and
  2. writes a fresh ``<base>_query.csv`` whose ``low``/``upp`` are the **exact**
     PN/PS/PNS bounds from :class:`ExactCredalCausalInference` (this replaces
     credici's outer-approximation with the tight bound).

The output ``_query.csv`` keeps credici's column layout
``cause,effect,low,query,tinfer,tlearn,upp`` so the files are drop-in compatible;
``tinfer`` is this solver's wall-clock per query in milliseconds and ``tlearn``
is the one-off model/data load time.

Which queries are computed is taken from each input ``_query.csv`` (its
cause/effect/query rows), so the new dataset answers exactly the same questions.
Rows the solver cannot evaluate (e.g. a cause that is not a direct parent) are
skipped and reported, never guessed.

Usage:
    python generate_markovian.py [--input DIR] [--output DIR] [--workers N]
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exact_credal_solver import ExactCredalCausalInference, _load  # noqa: E402

DEFAULT_INPUT = "/Users/antoniogonzalezalves/Documents/s23"
DEFAULT_OUTPUT = "/Users/antoniogonzalezalves/Documents/s23_markovian"
QUERY_METHOD = {"PS": "prob_sufficiency", "PN": "prob_necessity",
                "PNS": "prob_necessity_sufficiency"}
OUT_COLUMNS = ["cause", "effect", "low", "query", "tinfer", "tlearn", "upp"]


def process_model(base: str, out_dir: str) -> dict:
    """Solve one model's queries exactly and write the output triple.

    Returns a status dict: ``{name, status, n_queries, n_skipped, message}``.
    """
    name = Path(base).name
    try:
        t0 = time.perf_counter()
        model, data = _load(base)
        inf = ExactCredalCausalInference(model, data)
        tlearn_ms = int((time.perf_counter() - t0) * 1000)
        q = pd.read_csv(base + "_query.csv")
    except Exception as e:                       # unreadable model/data/query
        return {"name": name, "status": "error", "n_queries": 0,
                "n_skipped": 0, "message": f"load: {type(e).__name__}: {e}"}

    rows, skipped = [], 0
    for _, r in q.iterrows():
        method = QUERY_METHOD.get(str(r["query"]).upper())
        if method is None:
            skipped += 1
            continue
        try:
            t1 = time.perf_counter()
            low, upp = getattr(inf, method)(r["cause"], r["effect"])
            tinfer_ms = int((time.perf_counter() - t1) * 1000)
        except Exception:                        # cause not a parent, etc.
            skipped += 1
            continue
        rows.append({"cause": r["cause"], "effect": r["effect"], "low": low,
                     "query": r["query"], "tinfer": tinfer_ms, "tlearn": tlearn_ms,
                     "upp": upp})

    if not rows:
        return {"name": name, "status": "no_queries", "n_queries": 0,
                "n_skipped": skipped, "message": "no evaluable queries"}

    # Copy model + data and write the exact query results.
    shutil.copy2(base + ".uai", os.path.join(out_dir, name + ".uai"))
    shutil.copy2(base + ".csv", os.path.join(out_dir, name + ".csv"))
    pd.DataFrame(rows, columns=OUT_COLUMNS).to_csv(
        os.path.join(out_dir, name + "_query.csv"), index=False)
    return {"name": name, "status": "ok", "n_queries": len(rows),
            "n_skipped": skipped, "message": ""}


def collect_bases(input_dir: str) -> list[str]:
    """Model bases in ``input_dir`` that have the full uai/csv/query triple."""
    bases = []
    for uai in sorted(glob.glob(os.path.join(input_dir, "*.uai"))):
        base = uai[:-4]
        if os.path.exists(base + ".csv") and os.path.exists(base + "_query.csv"):
            bases.append(base)
    return bases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="Parallel worker processes (default: all cores).")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    bases = collect_bases(args.input)
    print(f"Input : {args.input}\nOutput: {args.output}\n{len(bases)} models to process "
          f"| workers={args.workers}\n")

    t0 = time.perf_counter()
    ok = err = empty = q_total = q_skipped = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_model, b, args.output): b for b in bases}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            ok += res["status"] == "ok"
            err += res["status"] == "error"
            empty += res["status"] == "no_queries"
            q_total += res["n_queries"]
            q_skipped += res["n_skipped"]
            if res["status"] != "ok":
                print(f"  [{i}/{len(bases)}] {res['name']}: {res['status']} "
                      f"{res['message']}")
            if i % 200 == 0:
                print(f"  ... {i}/{len(bases)} done")

    print(f"\nDone in {time.perf_counter()-t0:.0f}s. "
          f"models written={ok}, empty={empty}, errored={err} "
          f"| queries written={q_total}, skipped rows={q_skipped}")
    print(f"Output dataset: {args.output}")


if __name__ == "__main__":
    main()
