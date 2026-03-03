import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing

from bcause.models.cmodel import StructuralCausalModel
from bcause.util import randomUtil
from bcause.inference.causal.multi import CausalMultiInference

# Import your samplers
# from bcause.learning.parameter.gibbs import GibbsSampling
from bcause.learning.parameter.metropolis_hastings import MetropolisHastingsSampling

import warnings
import os
import re
import time

warnings.filterwarnings("ignore")

# Define models and data paths
DIRECTORY_PATH = "/Users/antoniogonzalezalves/Documents/s23/"
DOWNLOAD_PATH = "/Users/antoniogonzalezalves/Documents/prueba_mh/"
N_ITERATIONS = 10000
NUM_WORKERS = 6


# =====================================================================
# 1. FILE PARSING
# =====================================================================

def extract_file_info(filename):
    """Parses the filename to extract model parameters and ID."""
    id_match = re.search(r'_(\d+)(?:_query)?\.(?:uai|csv)$', filename)
    if not id_match: return None
    model_id = int(id_match.group(1))

    nparents_match = re.search(r'_nparents(\d+)', filename)
    nparents = int(nparents_match.group(1)) if nparents_match else None

    nzr_match = re.search(r'_nzr(\d+)', filename)
    nzr = int(nzr_match.group(1)) if nzr_match else None

    zdr_match = re.search(r'_zdr(\d+)', filename)
    zdr = int(zdr_match.group(1)) if zdr_match else None

    cardinality_children = 3 if 'ysize3' in filename else 2

    return {
        # CHANGED: model_id is now the FIRST element in the tuple
        "unique_key": (model_id, nparents, nzr, zdr, cardinality_children),
        "meta": {"nparents": nparents, "nzr": nzr, "zdr": zdr, "cardinality_children": cardinality_children,
                 "index": model_id}
    }


# =====================================================================
# 2. THE WORKER FUNCTION (Must be top-level for Multiprocessing)
# =====================================================================

def process_single_model_task(task_args):
    """
    Executes a specific sampler algorithm for a single model.
    task_args is a tuple: (model_info, sampler_class, method_name, n_steps, track_acceptance, outliers_removal)
    """
    model_info, sampler_class, method_name, n_steps, track_acceptance, outliers_removal = task_args
    results_list = []

    try:
        model = StructuralCausalModel.read(model_info["model"])
        data_int = pd.read_csv(model_info["data"], index_col=0).add_prefix('V')
        query = pd.read_csv(model_info["query"])

        # 'unique_id' is stored here, so it automatically gets passed to every row!
        model_meta = {
            "unique_id": model_info["unique_id"],
            "Model_Index": model_info["index"],
            "nparents": model_info["nparents"],
            "nzr": model_info["nzr"],
            "zdr": model_info["zdr"],
            "cardinality": model_info["cardinality_children"]
        }

        randomUtil.seed(model_info["index"])
        sampler_instance = sampler_class(model.randomize_factors(model.exogenous, allow_zero=False))

        start_time_init = time.time()
        sampler_instance.initialize(data_int[model.endogenous])
        total_time = time.time() - start_time_init

        for iter in range(n_steps + 1):
            start_time_step = time.time()
            sampler_instance.step()
            total_time += (time.time() - start_time_step)

            if (iter % 1000 == 0) and (iter > 0):
                # NOTE: Be careful here with memory. Passing an ever-growing list
                # (model_evolution) might cause memory issues at 10,000 iterations.
                inf_engine = CausalMultiInference(
                    sampler_instance.model_evolution[200:],
                    outliers_removal=outliers_removal
                )

                for instance in query.itertuples(index=False):
                    res_val = None
                    print(f"    -> Model ID {model_info['unique_id']}, Iter {iter}, Time {total_time:.2f}s, Processing Query: {instance.query}({instance.cause} -> {instance.effect})")
                    if instance.query == "PS":
                        res_val = inf_engine.prob_sufficiency(instance.cause, instance.effect,
                                                              true_false_cause=(1, 0), true_false_effect=(1, 0))
                    elif instance.query == "PN":
                        res_val = inf_engine.prob_necessity(instance.cause, instance.effect,
                                                            true_false_cause=(1, 0), true_false_effect=(1, 0))

                    if res_val is not None:
                        row = model_meta.copy()

                        exact_res = None
                        if hasattr(instance, 'low') and hasattr(instance, 'upp'):
                            exact_res = [instance.low, instance.upp]

                        row.update({
                            "Iteration": iter,
                            "cause": instance.cause,
                            "effect": instance.effect,
                            "query": instance.query,
                            "Method": method_name,
                            "Estimate": res_val,
                            "exact_result": exact_res,
                            "Time": total_time
                        })

                        if track_acceptance:
                            row["Acceptance_Rate"] = getattr(sampler_instance, 'acceptance_rate', None)

                        results_list.append(row)

        return results_list

    except KeyboardInterrupt:
        return []
    except Exception as e:
        print(f"\n[!] Error in model {model_info['index']}: {e}")
        return []


# =====================================================================
# 3. GENERALIZED PARALLEL EVALUATION FUNCTION (WITH RESUME CAPABILITY)
# =====================================================================
def evaluate_sampler_parallel(sampler_class, method_name, sets_to_run, N_steps, download_dir, num_workers=4,
                              track_acceptance=False, outliers_removal=False):
    """
    Runs any sampler class in PARALLEL and saves a uniform CSV file.
    Automatically resumes from where it left off if a CSV already exists.
    Saves progress safely on KeyboardInterrupt.
    """
    print(f"\n========================================================")
    print(f"[*] Preparing to run {method_name}")

    keys_sorted = sorted(sets_to_run.keys())

    file_name = f"{method_name.replace(' ', '_')}_results.csv"
    download_file = os.path.join(download_dir, file_name)

    existing_results = []
    completed_models = set()

    if os.path.exists(download_file):
        try:
            df_existing = pd.read_csv(download_file)

            # Check for the unique_id column instead of Unique_Query_ID
            if not df_existing.empty and "unique_id" in df_existing.columns and "Iteration" in df_existing.columns:
                # 1. Find EXACTLY which unique_ids reached the final iteration
                completed_df = df_existing[df_existing['Iteration'] == N_steps]
                completed_models = set(completed_df["unique_id"].unique())

                # 2. Keep ONLY the data for models that successfully finished all iterations
                df_clean = df_existing[df_existing['unique_id'].isin(completed_models)]
                existing_results = df_clean.to_dict('records')

                print(f"\n[*] Found existing file for {method_name}.")
                print(f"[*] Loaded data for {len(completed_models)} fully completed unique models.")

        except Exception as e:
            print(f"\n[!] Could not read existing file for {method_name}: {e}. Starting fresh.")

        # CHANGED: Build pending_sets correctly by looping through the sorted dictionary keys
    pending_sets = []
    for key in keys_sorted:
        model_id, nparents, nzr, zdr, cardinality_children = key

        # Skip this model if its exact parameter tuple is fully completed
        # Note: key is a tuple, so we convert to string to match the CSV unique_id
        if str(key) in completed_models:
            continue

        val = sets_to_run[key]
        pending_sets.append({
            'unique_id': str(key),
            'index': model_id,
            'nparents': nparents,
            'nzr': nzr,
            'zdr': zdr,
            'cardinality_children': cardinality_children,
            'model': val['files']['model'],
            'data': val['files']['data'],
            'query': val['files']['query']
        })

    if not pending_sets:
        print(f"\n[*] {method_name} has already processed all {len(sets_to_run)} models. Skipping entirely!")
        return pd.DataFrame(existing_results)

    print(f"\n========================================================")
    print(
        f"[{method_name}] Starting Parallel Evaluation on {num_workers} cores for {len(pending_sets)} pending models...")

    tasks = [(model_info, sampler_class, method_name, N_steps, track_acceptance, outliers_removal) for model_info in
             pending_sets]
    new_results = []

    try:
        # The 'with' statement safely manages the pool's lifecycle
        with multiprocessing.Pool(processes=num_workers) as pool:
            results_iterator = pool.imap_unordered(process_single_model_task, tasks)

            for res in tqdm(results_iterator, total=len(tasks), desc=f"Processing"):
                if res:
                    new_results.extend(res)

    except KeyboardInterrupt:
        # The 'with' block has already automatically terminated the pool at this point!
        print(f"\n[!] Manual stop detected during {method_name}!")

    except Exception as e:
        print(f"\n[!] Fatal Error encountered during {method_name}: {e}")

    finally:
        all_results = existing_results + new_results

        if all_results:
            df_results = pd.DataFrame(all_results)

            fully_completed_ids = set(df_results[df_results['Iteration'] == N_steps]['unique_id'].unique())
            df_results = df_results[df_results['unique_id'].isin(fully_completed_ids)]

            if not df_results.empty:
                df_results.to_csv(download_file, index=False)
                print(f"[*] Saved {len(df_results)} total rows successfully to: {download_file}")
                return df_results
            else:
                print(f"[*] No fully completed models to save for {method_name}.")
                return pd.DataFrame()
        else:
            print(f"[*] No results to save for {method_name}.")
            return pd.DataFrame()


# =====================================================================
# 4. MAIN EXECUTION BLOCK
# =====================================================================

if __name__ == '__main__':
    # Build the list of models to run
    file_map = {}
    for f in os.listdir(DIRECTORY_PATH):
        if not (f.endswith('.uai') or f.endswith('.csv')):
            continue

        file_type = 'model' if f.endswith('.uai') else ('query' if '_query' in f else 'data')

        info = extract_file_info(f)
        if info:
            key = info['unique_key']
            if key not in file_map:
                file_map[key] = {'files': {}, 'meta': info['meta']}
            file_map[key]['files'][file_type] = os.path.join(DIRECTORY_PATH, f)

    # Exclude problematic indices
    excluded_indices = {2, 7, 9, 23, 24, 27, 28, 41, 48}

    # CHANGED: The tuple format unpacked here is now (model_id, nparents, nzr, zdr, cardinality)
    file_map = {
        (model_id, nparents, nzr, zdr, cardinality): val
        for (model_id, nparents, nzr, zdr, cardinality), val in file_map.items()
        if nparents is not None
           and cardinality <= 2
           and model_id not in excluded_indices
           and nparents ==2
           and model_id <= 15
           and all(k in val['files'] for k in ['model', 'data', 'query'])
    }

    # Run the tests

    from dev_ig.Metropolis_Hastings.ParallelTemperingMCMC import MetropolisHastingsSampling as ParallelMH
    from dev.Metropolis_Hastings.Algorithms.ZanellaMCMC import MetropolisHastingsSampling as ZanellaMH
    from dev_ig.Metropolis_Hastings.SwandsenWangMCMC import MetropolisHastingsSampling as SwandsenWangMH
    from dev_ig.Metropolis_Hastings.MCMCAlwaysTrue import MetropolisHastingsSampling as MHAlwaysTrue

    # df_mh_always = evaluate_sampler_parallel(MHAlwaysTrue, "Metropolis_Hastings_AlwaysTrue", file_map, N_ITERATIONS,
    #                                          DOWNLOAD_PATH, NUM_WORKERS, track_acceptance=True)
    df_exclude_outliers = evaluate_sampler_parallel(MetropolisHastingsSampling, "Metropolis_Hastings_Exclude_Outliers", file_map, N_ITERATIONS,
                                                    DOWNLOAD_PATH, NUM_WORKERS, track_acceptance=True, outliers_removal=True)
    df_mh_parallel = evaluate_sampler_parallel(ParallelMH, "Metropolis_Hastings_Parallel_Tempering", file_map, N_ITERATIONS,
                                             DOWNLOAD_PATH, NUM_WORKERS, track_acceptance=True)
    df_mh_parallel_wo_outliers = evaluate_sampler_parallel(ParallelMH, "Metropolis_Hastings_Parallel_Tempering_wo_outliers", file_map, N_ITERATIONS,
                                             DOWNLOAD_PATH, NUM_WORKERS, track_acceptance=True, outliers_removal=True)
    df_mh_zanella = evaluate_sampler_parallel(ZanellaMH, "Metropolis_Hastings_Zanella", file_map, N_ITERATIONS,
                                             DOWNLOAD_PATH, NUM_WORKERS, track_acceptance=True)
    df_zanella_wo_outliers = evaluate_sampler_parallel(ZanellaMH, "Metropolis_Hastings_Zanella_wo_outliers", file_map, N_ITERATIONS,
                                             DOWNLOAD_PATH, NUM_WORKERS, track_acceptance=True,outliers_removal=True)
    # df_swandsen = evaluate_sampler_parallel(SwandsenWangMH, "Metropolis_Hastings_Swandsen_Wang", file_map, N_ITERATIONS,
    #                                          DOWNLOAD_PATH, NUM_WORKERS, track_acceptance=True)