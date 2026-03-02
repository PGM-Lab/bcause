import pandas as pd
import numpy as np
from sympy.codegen.fnodes import merge

from bcause.models.cmodel import StructuralCausalModel
from dev_ig.expectation_maximization import ExpectationMaximizationPrecomputed
from bcause.util.watch import Watch
from bcause.factors import MultinomialFactor
from bcause.util import randomUtil
from bcause.inference.causal.multi import CausalMultiInference
from bcause.learning.parameter.gibbs import GibbsSampling
from bcause.learning.parameter.metropolis_hastings import MetropolisHastingsSampling

import warnings
import os
import re
import time
import random

warnings.filterwarnings("ignore")

# Define models and data paths
directory_path = "/Users/antoniogonzalezalves/Documents/s23/"
download_path = "/Users/antoniogonzalezalves/Documents/prueba_mh/"


def extract_file_info(filename):
    """
    Parses the filename to extract model parameters and ID.
    Returns a dictionary of attributes to serve as a unique key.
    """
    # 1. Extract Model ID (the number just before the extension)
    #    Matches pattern: _123.uai or _123_query.csv or _123.csv
    id_match = re.search(r'_(\d+)(?:_query)?\.(?:uai|csv)$', filename)
    if not id_match:
        return None
    model_id = int(id_match.group(1))

    # 2. Extract nparents (e.g., _nparents2_)
    nparents_match = re.search(r'_nparents(\d+)', filename)
    nparents = int(nparents_match.group(1)) if nparents_match else None

    # 3. Extract nzr (e.g., _nzr06_)
    nzr_match = re.search(r'_nzr(\d+)', filename)
    nzr = int(nzr_match.group(1)) if nzr_match else None

    # 4. Extract zdr (e.g., _zdr05_)
    zdr_match = re.search(r'_zdr(\d+)', filename)
    zdr = int(zdr_match.group(1)) if zdr_match else None

    # 5. Extract ysize/cardinality (e.g., _ysize3_)
    #    If 'ysize3' exists -> 3, else defaults to 2 (as per your logic)
    if 'ysize3' in filename:
        cardinality_children = 3
    else:
        cardinality_children = 2

    # Return a unique key tuple + metadata
    return {
        "unique_key": (nparents, nzr, zdr, cardinality_children, model_id),
        "meta": {
            "nparents": nparents,
            "nzr": nzr,
            "zdr": zdr,
            "cardinality_children": cardinality_children,
            "index": model_id
        }
    }


file_map = {}

for f in os.listdir(directory_path):
    # Determine file type
    file_type = None
    if f.endswith('.uai'):
        file_type = 'model'
    elif f.endswith('.csv') and '_query' in f:
        file_type = 'query'
    elif f.endswith('.csv'):
        file_type = 'data'

    if not file_type:
        continue

    # Extract info
    info = extract_file_info(f)
    if info:
        key = info['unique_key']

        if key not in file_map:
            file_map[key] = {
                'files': {},
                'meta': info['meta']
            }

        # Store full path
        full_path = os.path.join(directory_path, f)
        file_map[key]['files'][file_type] = full_path

sets = []

for key, val in file_map.items():
    files = val['files']
    meta = val['meta']

    # Ensure we have all three components for this specific configuration
    if all(k in files for k in ['model', 'data', 'query']):
        sets.append({
            'unique_id': str(key),  # String representation for reference
            'index': meta['index'],
            'nparents': meta['nparents'],
            'nzr': meta['nzr'],
            'zdr': meta['zdr'],
            'cardinality_children': meta['cardinality_children'],
            'model': files['model'],
            'data': files['data'],
            'query': files['query']
        })

# Optional: Sort by index (and maybe other params) for consistent order
sets.sort(key=lambda x: (x['index'], x['nparents'], x['cardinality_children']))

dataframe_models_pre = pd.DataFrame(sets)


# filter out models with nparents > 2
sets = [s for s in sets if s['nparents'] is not None and s['cardinality_children'] <= 2]
sets = [s for s in sets if s['index'] not in [2,7,9,23,24,27,28,41,48]]
sets = [s for s in sets if s['index'] < 3]

dataframe_models = pd.DataFrame(sets)

# sets = sets[0:50]
# Filter sets to keep only those with 2 or fewer parents
# sets = [s for s in sets if s['nparents'] is not None and s['nparents'] <= 2]

# sets = random.sample(sets, 1)

# dataframe_results_EM = pd.DataFrame(columns=["Model_Index","Iteration","cause","effect","query", "EM_Precomp"])
dataframe_results_gibbs = pd.DataFrame(columns=[
    "Model_Index", "nparents", "nzr", "zdr", "cardinality",
    "Iteration", "cause", "effect", "query", "Gibbs_Sampling", "Time"
])
dataframe_results_mh = pd.DataFrame(columns=[
    "Model_Index", "nparents", "nzr", "zdr", "cardinality",
    "Iteration", "cause", "effect", "query", "Metropolis_Hastings", "Time"
])
dataframe_results_mh_alt = pd.DataFrame(columns=[
    "Model_Index", "nparents", "nzr", "zdr", "cardinality",
    "Iteration", "cause", "effect", "query", "Metropolis_Hastings_ALT", "Time"
])

N=5000
# N=5
for n,i in enumerate(sets):
    print(
        f"Processing model {n + 1} of {len(sets)} | ID: {i['index']} | Parents: {i['nparents']} | Card: {i['cardinality_children']}")

    model = StructuralCausalModel.read(i["model"])
    data = pd.read_csv(i["data"], dtype='str', index_col=0).add_prefix('V')
    query = pd.read_csv(i["query"])
    data_int = pd.read_csv(i["data"], index_col=0).add_prefix('V')

    # Common dict for result rows (to avoid repetition)
    model_meta = {
        "Model_Index": i["index"],
        "nparents": i["nparents"],
        "nzr": i["nzr"],
        "zdr": i["zdr"],
        "cardinality": i["cardinality_children"]
    }

    # Run Precomputed EM

    # for var in model_em.endogenous:
    #     factor = model_em.factors[var]
    #     # Ensure all probabilities are non-zero
    #     probs = factor.values
    #     probs[probs == 0] = 1e-4
    # model_em = model.copy()
    # models_em_precomp = []
    # for iter in range(N+1):
    #     em_precomp= ExpectationMaximizationPrecomputed(model_em.randomize_factors(model_em.exogenous, allow_zero=False), ignore_convergence=True,as_list=False)
    #     em_precomp.run(data_int[model_em.endogenous],max_iter=100)
    #     models_em_precomp.append(em_precomp.model)
    #     if (iter % 100 == 0) and (iter > 0):
    #         inf_em_precomp = CausalMultiInference(models_em_precomp)
    #         for instance in query.itertuples(index=False):
    #             if instance.query == "PS":
    #                 dataframe_results_EM = pd.concat([dataframe_results_EM, pd.DataFrame([{
    #                     "Model_Index": i["index"],
    #                     "Iteration": iter,
    #                     "cause": instance.cause,
    #                     "effect": instance.effect,
    #                     "query": instance.query,
    #                     "EM_Precomp": inf_em_precomp.prob_sufficiency(instance.cause, instance.effect, true_false_cause=(1, 0), true_false_effect=(1, 0))
    #                 }])], ignore_index=True)
    #             elif instance.query == "PN":
    #                 dataframe_results_EM = pd.concat([dataframe_results_EM, pd.DataFrame([{
    #                     "Model_Index": i["index"],
    #                     "Iteration": iter,
    #                     "cause": instance.cause,
    #                     "effect": instance.effect,
    #                     "query": instance.query,
    #                     "EM_Precomp": inf_em_precomp.prob_necessity(instance.cause, instance.effect, true_false_cause=(1, 0), true_false_effect=(1, 0))
    #                 }])], ignore_index=True)
    #
    # # Download csv results for EM
    # download_file_em = os.path.join(download_path, f"EM_Precomp_results.csv")
    # # download csv results for EM
    # em_precomp_export = dataframe_results_EM.merge(query, left_on=["cause","effect","query"], right_on=["cause","effect","query"], how="left")
    # em_precomp_export["exact_result"] = [list(x) for x in zip(em_precomp_export['low'], em_precomp_export['upp'])]
    # em_precomp_export = em_precomp_export.drop(columns=["low","tinfer","tlearn","upp"])
    # em_precomp_export.to_csv(download_file_em, index=False)


    # Gibbs Sampling
    model_gs = StructuralCausalModel.read(i["model"])
    randomUtil.seed(n)
    gs = GibbsSampling(model_gs.randomize_factors(model_gs.exogenous, allow_zero=False))

    # --- Time Initialization ---
    start_time_gs_init = time.time()
    gs.initialize(data_int[model_gs.endogenous])
    end_time_gs_init = time.time()
    time_gs_learn_total = end_time_gs_init - start_time_gs_init
    # ----------------------------------------------------

    for iter in range(N + 1):
        start_time_gs_step = time.time()
        gs.step()
        end_time_gs_step = time.time()
        time_gs_learn_total += (end_time_gs_step - start_time_gs_step)

        if (iter % 1000 == 0) and (iter > 0):
            print(f"    Model {n + 1} - Gibbs: Iter {iter}, Time {time_gs_learn_total:.2f}s")
            inf_gs = CausalMultiInference(gs.model_evolution[int(iter / 5):])

            for instance in query.itertuples(index=False):
                res_val = None
                if instance.query == "PS":
                    res_val = inf_gs.prob_sufficiency(instance.cause, instance.effect, true_false_cause=(1, 0),
                                                      true_false_effect=(1, 0))
                elif instance.query == "PN":
                    res_val = inf_gs.prob_necessity(instance.cause, instance.effect, true_false_cause=(1, 0),
                                                    true_false_effect=(1, 0))

                if res_val is not None:
                    row = model_meta.copy()
                    row.update({
                        "Iteration": iter,
                        "cause": instance.cause,
                        "effect": instance.effect,
                        "query": instance.query,
                        "Gibbs_Sampling": res_val,
                        "Time": time_gs_learn_total
                    })
                    dataframe_results_gibbs = pd.concat([dataframe_results_gibbs, pd.DataFrame([row])],
                                                        ignore_index=True)

    # Download csv results for Gibbs
    download_file_gibbs = os.path.join(download_path, f"Gibbs_Sampling_results.csv")
    gb_precomp_export = dataframe_results_gibbs.merge(query, left_on=["cause","effect","query"], right_on=["cause","effect","query"], how="left")
    gb_precomp_export["exact_result"] = [list(x) for x in zip(gb_precomp_export['low'], gb_precomp_export['upp'])]
    gb_precomp_export = gb_precomp_export.drop(columns=["low","tinfer","tlearn","upp"])
    gb_precomp_export.to_csv(download_file_gibbs, index=False)

    # Metropolis_Hastings
    model_mh = StructuralCausalModel.read(i["model"])
    randomUtil.seed(n)
    mh = MetropolisHastingsSampling(model_mh.randomize_factors(model_mh.exogenous, allow_zero=False))
    # --- Time Initialization  ---
    start_time_mh_init = time.time()
    mh.initialize(data_int[model_mh.endogenous])
    end_time_mh_init = time.time()
    time_mh_learn_total = end_time_mh_init - start_time_mh_init
    # ---------------------------------------------------
    for iter in range(N + 1):
        start_time_mh_step = time.time()
        mh.step()
        end_time_mh_step = time.time()
        time_mh_learn_total += (end_time_mh_step - start_time_mh_step)

        if (iter % 1000 == 0) and (iter > 0):
            print(f"    Model {n + 1} - MH: Iter {iter}, Time {time_mh_learn_total:.2f}s")
            inf_mh = CausalMultiInference(mh.model_evolution[int(iter / 5):])

            for instance in query.itertuples(index=False):
                res_val = None
                if instance.query == "PS":
                    res_val = inf_mh.prob_sufficiency(instance.cause, instance.effect, true_false_cause=(1, 0),
                                                      true_false_effect=(1, 0))
                elif instance.query == "PN":
                    res_val = inf_mh.prob_necessity(instance.cause, instance.effect, true_false_cause=(1, 0),
                                                    true_false_effect=(1, 0))

                if res_val is not None:
                    row = model_meta.copy()
                    row.update({
                        "Iteration": iter,
                        "cause": instance.cause,
                        "effect": instance.effect,
                        "query": instance.query,
                        "Metropolis_Hastings": res_val,
                        "Time": time_mh_learn_total
                    })
                    dataframe_results_mh = pd.concat([dataframe_results_mh, pd.DataFrame([row])], ignore_index=True)

    # Download csv results for MH
    download_file_mh = os.path.join(download_path, f"Metropolis_Hastings_results.csv")
    mh_precomp_export = dataframe_results_mh.merge(query, left_on=["cause","effect","query"], right_on=["cause","effect","query"], how="left")
    mh_precomp_export["exact_result"] =[list(x) for x in zip(mh_precomp_export['low'], mh_precomp_export['upp'])]
    mh_precomp_export = mh_precomp_export.drop(columns=["low","tinfer","tlearn","upp"])
    dataframe_results_mh.to_csv(download_file_mh, index=False)

    # Metropolis_Hastings
    model_mh_alt = StructuralCausalModel.read(i["model"])
    randomUtil.seed(n)
    mh_alt = MHv3(model_mh_alt.randomize_factors(model_mh_alt.exogenous, allow_zero=False))
    # --- Time Initialization  ---
    start_time_mh_alt_init = time.time()
    mh_alt.initialize(data_int[model_mh_alt.endogenous])
    end_time_mh_alt_init = time.time()
    time_mh_alt_learn_total = end_time_mh_alt_init - start_time_mh_alt_init
    # ---------------------------------------------------
    for iter in range(N + 1):
        start_time_mh_alt_step = time.time()
        mh_alt.step()
        end_time_mh_alt_step = time.time()
        time_mh_alt_learn_total += (end_time_mh_alt_step - start_time_mh_alt_step)

        if (iter % 1000 == 0) and (iter > 0):
            print(f"    Model {n + 1} - MH_ALT: Iter {iter}, Time {time_mh_alt_learn_total:.2f}s")
            inf_mh_alt = CausalMultiInference(mh_alt.model_evolution[int(iter / 5):])

            for instance in query.itertuples(index=False):
                res_val = None
                if instance.query == "PS":
                    res_val = inf_mh_alt.prob_sufficiency(instance.cause, instance.effect, true_false_cause=(1, 0),
                                                      true_false_effect=(1, 0))
                elif instance.query == "PN":
                    res_val = inf_mh_alt.prob_necessity(instance.cause, instance.effect, true_false_cause=(1, 0),
                                                    true_false_effect=(1, 0))

                if res_val is not None:
                    row = model_meta.copy()
                    row.update({
                        "Iteration": iter,
                        "cause": instance.cause,
                        "effect": instance.effect,
                        "query": instance.query,
                        "Metropolis_Hastings": res_val,
                        "Time": time_mh_alt_learn_total
                    })
                    dataframe_results_mh_alt = pd.concat([dataframe_results_mh_alt, pd.DataFrame([row])], ignore_index=True)

    # Download csv results for MH
    download_file_mh_alt = os.path.join(download_path, f"Metropolis_Hastings_alt_results.csv")
    mh_alt_precomp_export = dataframe_results_mh_alt.merge(query, left_on=["cause", "effect", "query"],
                                                   right_on=["cause", "effect", "query"], how="left")
    mh_alt_precomp_export["exact_result"] = [list(x) for x in zip(mh_alt_precomp_export['low'], mh_alt_precomp_export['upp'])]
    mh_alt_precomp_export = mh_alt_precomp_export.drop(columns=["low", "tinfer", "tlearn", "upp"])
    dataframe_results_mh_alt.to_csv(download_file_mh_alt, index=False)

gb_precomp_export = pd.read_csv(os.path.join(download_path, f"Gibbs_Sampling_results.csv"))
mh_precomp_export = pd.read_csv(os.path.join(download_path, f"Metropolis_Hastings_results.csv"))
mh_alt_precomp_export = pd.read_csv(os.path.join(download_path, f"Metropolis_Hastings_alt_results.csv"))

# merge all results
merge_keys = ["Model_Index", "nparents", "nzr", "zdr", "cardinality", "Iteration", "cause", "effect", "query"]

final_results = gb_precomp_export.merge(mh_precomp_export, on=merge_keys, how="outer",suffixes=("_gibbs", "_mh"))
final_results = final_results.merge(mh_alt_precomp_export, on=merge_keys, how="outer", suffixes=("", "_mh_alt"))

# Clean up exact result columns if they are duplicated
if "exact_result_gibbs" in final_results.columns:
    final_results.rename(columns={"exact_result_gibbs": "exact_result"}, inplace=True)
if "exact_result_mh" in final_results.columns:
    final_results.drop(columns=["exact_result_mh"], inplace=True)
if "exact_result_mh_alt" in final_results.columns:
    final_results.drop(columns=["exact_result_mh_alt"], inplace=True)

final_results.to_csv(os.path.join(download_path, "Final_Comparison_Results.csv"), index=False)




