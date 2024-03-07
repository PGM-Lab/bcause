from pathlib import Path
import itertools
from multiprocessing import Pool, cpu_count

import pandas as pd

from bcause.inference.causal.multi import GDCC, EMCC
from bcause.models.cmodel import StructuralCausalModel
from bcause.util import randomUtil
from bcause.util.mathutils import rrmse, rmse
from bcause.util.runningutils import get_logger
from bcause.util.watch import Watch


### Set parameters ###

# Single parameters
num_runs = 100
modelpath = "./papers/gradient_journal/models/synthetic/s123/random_mc2_n5_mid3_d1000_05_mr098_r10_8.uai"
resfolder = "./papers/gradient_journal/results/synthetic/s123/"
run_step = 5

# Multi parameters
USE_FULL_PARAMETERS = True
if USE_FULL_PARAMETERS:
    seed_values = [1,2,3,4]
    remove_outliers_values = [True, False]
    method_values = ["EMCC", "GDCC"]
    max_iter_values_emcc = [25, 50, 100, 150, 200]  # Relevant for EMCC
    tol_values_gdcc = [1e-3, 1e-5, 1e-7, 1e-9]     # Relevant for GDCC
else: # subset of full parameters used for debugg
    seed_values = [1]
    remove_outliers_values = [True]
    method_values = ["GDCC"]
    max_iter_values_emcc = [25]  # Relevant for EMCC
    tol_values_gdcc = [1e-3, 1e-5, 1e-7, 1e-9]     # Relevant for GDCC

### Set parameters End ###

def process_parameters(params):
    num_runs, modelpath, resfolder, run_step, seed, remove_outliers, method, max_iter, tol = params
    # Processing logic here
    print(f"Processing: {seed=}, {remove_outliers=}, {method=}, {max_iter=}, {tol=}") # Single parameters are omitted here
    # Define the logger
    log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'
    log = get_logger(__name__, fmt=log_format)

    # Set the random seed
    randomUtil.seed(seed)

    # Load the model
    model = StructuralCausalModel.read(modelpath)
    #model.factors["V5"].values

    # Load data
    data = pd.read_csv(modelpath.replace(".uai",".csv"), index_col=0)
    data = data.rename(columns={c:"V"+c for c in data.columns})

    # Load the information about the query and the model
    info_query = pd.read_csv(modelpath.replace(".uai","_uai_ccve.csv"))
    pns_exact = (info_query.pns_l.values[0], info_query.pns_u.values[0])
    cause, effect = [f"V{i}" for i in list(info_query[["cause","effect"]].values.flatten())]
    modelname = modelpath.removesuffix(".uai").split("/")[-1]

    log.info(f"PNS exact: {pns_exact}")

    # Set the results
    resfilepath = Path(resfolder, f"{modelname}_uai_{method}_x{num_runs}_iter{max_iter}_tol{tol}_s{seed}.csv")
    results = pd.DataFrame()

    # Determine the method
    if method == "GDCC":
        inf = GDCC(model, data, num_runs=num_runs, tol = tol, outliers_removal=remove_outliers)
    elif method == "EMCC":
        inf = EMCC(model, data, num_runs=num_runs, max_iter=max_iter, outliers_removal=remove_outliers)
    else:
        raise ValueError("Wrong learning method")

    ### Start processing ###
    tlearn = 0
    t0 = 0
    Watch.start()

    # Learning loop
    for _ in inf.compile_incremental(run_step): # The learning is done here at each iteration

        t1 = Watch.get_time()
        # Run the query
        p = inf.prob_necessity_sufficiency(cause,effect, true_false_cause=(0,1), true_false_effect=(0,1))
        t2 = Watch.get_time()

        tlearn = tlearn + t1-t0 # JAN2RAFA: it is unclear to me why tlearn is cummulative but tinfer is measured each loop just for one iteration?
        tinfer = t2-t1
        err = rrmse(pns_exact[0], p[0], pns_exact[1], p[1])
        err2 = rmse(pns_exact[0], p[0], pns_exact[1], p[1])

        nruns = len(inf.models)

        msg = f"[{p[0]:.4f},{p[1]:.4f}]\t {nruns} runs\t rrmse={err:.5f}\t T_learn={tlearn:.0f} ms. \t T_infer={tinfer:.0f} ms."
        log.info(msg)

        # Save the results
        r = pd.DataFrame(dict(modelname=modelname, method=method, cause=cause, effect=effect, tol=tol,
                        num_runs_param=num_runs, max_iter_param=max_iter, seed=seed, tlearn=tlearn, tinfer=tinfer,
                        datasize=len(data),
                        pns_low_exact =pns_exact[0], pns_upp_exact = pns_exact[1],
                        pns_low = p[0], pns_upp = p[1], rrmse = err, rmse = err2, nruns=nruns
                        ), index=[0])

        results = pd.concat([results, r], ignore_index=True)
        results.to_csv(resfilepath)

        t0 = Watch.get_time()


def generate_parameter_combinations():
    # Generate combinations for each method 
    parameter_combinations = []
    for seed, remove_outliers in itertools.product(seed_values, remove_outliers_values):
        for method in method_values:
            if method == "EMCC":
                for max_iter in max_iter_values_emcc:
                    parameter_combinations.append((num_runs, modelpath, resfolder, run_step, seed, remove_outliers, method, max_iter, None))
            elif method == "GDCC":
                for tol in tol_values_gdcc:
                    parameter_combinations.append((num_runs, modelpath, resfolder, run_step, seed, remove_outliers, method, None, tol))
    return parameter_combinations


if __name__ == "__main__":
    # Display the number of available worker processes
    available_workers = cpu_count()
    print(f"Number of available workers: {available_workers}")
    parameter_combinations = generate_parameter_combinations()
    if False: # set to True to test in non-parallel setting
        process_parameters(parameter_combinations[0])  
        quit()
    # Parallel approach
    with Pool() as pool:
        pool.map(process_parameters, parameter_combinations)





