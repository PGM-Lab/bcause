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
# from dev_ig.Metropolis_Hastings.liftedMCMC import MetropolisHastingsSampling as MHv3


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
        "unique_key": (nparents, nzr, zdr, cardinality_children, model_id),
        "meta": {"nparents": nparents, "nzr": nzr, "zdr": zdr, "cardinality_children": cardinality_children,
                 "index": model_id}
    }


# =====================================================================
# 2. THE WORKER FUNCTION (Must be top-level for Multiprocessing)
# =====================================================================

def process_single_model_task(task_args):
    """
    Executes a specific sampler algorithm for a single model.
    task_args is a tuple: (model_info, sampler_class, method_name, n_steps, track_acceptance)
    """
    model_info, sampler_class, method_name, n_steps, track_acceptance, outliers_removal = task_args
    results_list = []

    try:
        model = StructuralCausalModel.read(model_info["model"])
        data_int = pd.read_csv(model_info["data"], index_col=0).add_prefix('V')
        query = pd.read_csv(model_info["query"])

        model_meta = {
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
        return results_list
    except Exception as e:
        print(f"\n[!] Error in model {model_info['index']}: {e}")
        return results_list


# =====================================================================
# 3. GENERALIZED PARALLEL EVALUATION FUNCTION (WITH RESUME CAPABILITY)
# =====================================================================
def evaluate_sampler_parallel(sampler_class, method_name, sets_to_run, N_steps, download_dir, num_workers=4,
                              track_acceptance=False,outliers_removal=False):
    """
    Runs any sampler class in PARALLEL and saves a uniform CSV file.
    Automatically resumes from where it left off if a CSV already exists.
    Saves progress safely on KeyboardInterrupt.
    """
    file_name = f"{method_name.replace(' ', '_')}_results.csv"
    download_file = os.path.join(download_dir, file_name)

    existing_results = []
    completed_models = set()

    # --- IMPROVED RESUME LOGIC ---
    if os.path.exists(download_file):
        try:
            df_existing = pd.read_csv(download_file)
            if not df_existing.empty and "Model_Index" in df_existing.columns and "Iteration" in df_existing.columns:
                # We only consider a model "completed" if it reached N_steps!
                # If a model crashed halfway through, we want to run it again.
                completed_df = df_existing[df_existing['Iteration'] == N_steps]
                completed_models = set(completed_df["Model_Index"].unique())

                # Keep only the results of fully completed models to avoid duplicate data
                # from partially finished runs that we are restarting.
                df_clean = df_existing[df_existing['Model_Index'].isin(completed_models)]
                existing_results = df_clean.to_dict('records')

                print(
                    f"\n[*] Found existing file for {method_name}. Skipping {len(completed_models)} FULLY completed models.")
        except Exception as e:
            print(f"\n[!] Could not read existing file for {method_name}: {e}. Starting fresh.")

    # Filter out the fully completed models
    pending_sets = [s for s in sets_to_run if s["index"] not in completed_models]

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
        with multiprocessing.Pool(processes=num_workers) as pool:
            results_iterator = pool.imap_unordered(process_single_model_task, tasks)

            for res in tqdm(results_iterator, total=len(tasks), desc=f"Processing"):
                if res:
                    new_results.extend(res)

    except KeyboardInterrupt:
        print(f"\n[!] Manual stop detected during {method_name}! Terminating pool and saving computed results...")
        pool.terminate()
        pool.join()
    except Exception as e:
        print(f"\n[!] Fatal Error encountered during {method_name}: {e}")

    finally:
        all_results = existing_results + new_results

        if all_results:
            df_results = pd.DataFrame(all_results)
            df_results.to_csv(download_file, index=False)
            print(f"[*] Saved {len(df_results)} total rows successfully to: {download_file}")
            return df_results
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
        # Slightly more efficient parsing logic
        if not (f.endswith('.uai') or f.endswith('.csv')):
            continue

        file_type = 'model' if f.endswith('.uai') else ('query' if '_query' in f else 'data')

        info = extract_file_info(f)
        if info:
            key = info['unique_key']
            if key not in file_map:
                file_map[key] = {'files': {}, 'meta': info['meta']}
            file_map[key]['files'][file_type] = os.path.join(DIRECTORY_PATH, f)

    sets = []
    for key, val in file_map.items():
        if all(k in val['files'] for k in ['model', 'data', 'query']):
            sets.append({
                'unique_id': str(key),
                'index': val['meta']['index'],
                'nparents': val['meta']['nparents'],
                'nzr': val['meta']['nzr'],
                'zdr': val['meta']['zdr'],
                'cardinality_children': val['meta']['cardinality_children'],
                'model': val['files']['model'],
                'data': val['files']['data'],
                'query': val['files']['query']
            })

    # Filter sets
    sets.sort(key=lambda x: (x['index'], x['nparents'], x['cardinality_children']))
    sets = [s for s in sets if s['nparents'] is not None and s['cardinality_children'] <= 2]

    # Exclude problematic indices
    excluded_indices = {2, 7, 9, 23, 24, 27, 28, 41, 48}
    sets = [s for s in sets if s['index'] not in excluded_indices]

    # Run the tests
    # (Added track_acceptance=True where appropriate)

    # df_gibbs = evaluate_sampler_parallel(GibbsSampling, "Gibbs_Sampling", sets, N_ITERATIONS, DOWNLOAD_PATH, NUM_WORKERS, track_acceptance=False)
    # df_mh = evaluate_sampler_parallel(MetropolisHastingsSampling, "Metropolis_Hastings", sets, N_ITERATIONS, DOWNLOAD_PATH, NUM_WORKERS, track_acceptance=True)
    from dev_ig.Metropolis_Hastings.ParallelTemperingMCMC import MetropolisHastingsSampling as ParallelMH
    from dev_ig.Metropolis_Hastings.ZanellaMCMC import MetropolisHastingsSampling as ZanellaMH
    from dev_ig.Metropolis_Hastings.SwandsenWangMCMC import MetropolisHastingsSampling as SwandsenWangMH
    from dev_ig.Metropolis_Hastings.MCMCAlwaysTrue import MetropolisHastingsSampling as MHAlwaysTrue


    df_mh_always = evaluate_sampler_parallel(MHAlwaysTrue, "Metropolis_Hastings_AlwaysTrue", sets, N_ITERATIONS,
                                               DOWNLOAD_PATH, NUM_WORKERS, track_acceptance=True)
    df_mh_remove_outliers = evaluate_sampler_parallel(MetropolisHastingsSampling, "Metropolis_Hastings_RemoveOutliers", sets, N_ITERATIONS,
                                               DOWNLOAD_PATH, NUM_WORKERS, track_acceptance=True,outliers_removal=True)
    df_mh_parallel = evaluate_sampler_parallel(ParallelMH, "Metropolis_Hastings_Parallel", sets, N_ITERATIONS,
                                               DOWNLOAD_PATH, NUM_WORKERS, track_acceptance=True)
    df_mh_zanella = evaluate_sampler_parallel(ZanellaMH, "Metropolis_Hastings_Zanella", sets, N_ITERATIONS,
                                              DOWNLOAD_PATH, NUM_WORKERS, track_acceptance=True)
    df_mh_swandsenwang = evaluate_sampler_parallel(SwandsenWangMH, "Metropolis_Hastings_SwandsenWang", sets,
                                                   N_ITERATIONS, DOWNLOAD_PATH, NUM_WORKERS, track_acceptance=True)