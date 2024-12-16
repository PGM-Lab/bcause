from pathlib import Path
import itertools
from multiprocessing import Pool, cpu_count
import os
import glob
from collections import defaultdict

from concurrent_log_handler import ConcurrentRotatingFileHandler
#pip install concurrent-log-handler

import logging
import sys

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
run_step = 5
resfolder = "./papers/gradient_journal/results/synthetic/simple/"
REWRITE = False
RUN_IN_PARALLEL = True

# Multi parameters
USE_FULL_PARAMETERS = True
if USE_FULL_PARAMETERS:
    seed_values = [1] # after our discussion, we keep only one value for the moment
    remove_outliers_values = [True, False]
    method_values = ["EMCC", "GDCC"]
    max_iter_values_emcc = [25, 50, 100, 150, 200]  # Relevant for EMCC
    tol_values_gdcc = [10 ** -i for i in range(1,11)]     # Relevant for GDCC
else: # subset of full parameters used for debugg
    seed_values = [1]
    remove_outliers_values = [True, False]
    method_values = ["GDCC"] #["EMCC", "GDCC"]
    max_iter_values_emcc = [25, 50, 100, 150]  # Relevant for EMCC
    tol_values_gdcc = [1e-3, 1e-5, 1e-7, 1e-9]  # Relevant for GDCC
    # this settings renders 16 independent process, as we have 16 available workers at our machine

### Set parameters End ###

log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'
logfile = Path(Path(resfolder).parent, f"experiments.log")


def get_clogger(logname, level=logging.INFO, stream=sys.stdout, fmt=None, filename=None):
    # ConcurrentRotatingFileHandler
    log = logging.getLogger(logname)
    log.setLevel(level)
    stdout_handler = logging.StreamHandler(stream)
    if fmt is not None:
        stdout_handler.setFormatter(logging.Formatter(fmt))
    log.addHandler(stdout_handler)

    if filename is not None:
        # Use ConcurrentRotatingFileHandler instead of FileHandler
        file_handler = ConcurrentRotatingFileHandler(filename, "a", 512*1024, 5)
        if fmt is not None:
            file_handler.setFormatter(logging.Formatter(fmt))
        log.addHandler(file_handler)
    return log


def process_parameters(params, log):
    num_runs, modelpath, resfolder, run_step, seed, remove_outliers, method, max_iter, tol = params

    # Check that the experiments with these parameters have not been run yet
    modelname = os.path.basename(modelpath).split(".")[0]
    resfilepath = Path(resfolder, f"{modelname}_uai_{method}_x{num_runs}_iter{max_iter}_tol{tol}_ro{remove_outliers}_s{seed}.csv")    
    if resfilepath.exists() and not REWRITE:
        log.info(f'{resfilepath} already exists. Skipping ...')
        return

    # Processing logic here
    print(f"Processing: {seed=}, {remove_outliers=}, {method=}, {max_iter=}, {tol=}") # Single parameters are omitted here

    # Set the random seed
    randomUtil.seed(seed)

    # Load the model
    model = StructuralCausalModel.read(modelpath)
    if 0:
        # plot the model
        import networkx as nx
        import matplotlib.pyplot as plt
        nx.draw(model.graph, with_labels=True, font_weight='bold')
        plt.show() 
    #model.factors["V5"].values

    # Load data
    data = pd.read_csv(modelpath.replace(".uai",".csv"), index_col=0)
    data = data.rename(columns={c:"V"+c for c in data.columns})

    # Load the information about the query and the model
    modelpath_ccve = modelpath.replace(".uai","_query.csv")
    if os.path.exists(modelpath_ccve):
        info_query = pd.read_csv(modelpath_ccve)
    else:
        print(f'{modelpath_ccve} does not exists, skipping ...') # JAN2RAFA: some of these modelpath_ccve were missing, so I skipped them
        return
    
    nV = int(len(info_query) / 2)
    causes = [f'V{i}' for i in range(1, nV + 1)] # V1, V2, V3
    effect = 'V0'

    exact = defaultdict(lambda : {})
    for query in ['PS', 'PN']: # 'PNS'
        for cause in causes:  
            interval = (info_query[(info_query['cause'] == cause) & (info_query['query'] == query)]['low'].values[0],
                        info_query[(info_query['cause'] == cause) & (info_query['query'] == query)]['upp'].values[0])
            exact[query][cause] = interval

    # Set the results
    results = pd.DataFrame()

    # Determine the method
    if method == "GDCC":
        inf = GDCC(model, data, num_runs=num_runs, tol = tol, outliers_removal=remove_outliers)
    elif method == "EMCC":
        inf = EMCC(model, data, num_runs=num_runs, max_iter=max_iter, outliers_removal=remove_outliers)
    else:
        raise ValueError("Wrong learning method")
    
    # Queries and inference functions
    queries_f = { 
        'PNS': inf.prob_necessity_sufficiency,
        'PS' : inf.prob_sufficiency,
        'PN' : inf.prob_necessity
    }

    ### Start processing ###
    tlearn = 0
    t0 = 0
    Watch.start()

    # Learning loop
    for _ in inf.compile_incremental(run_step): # The learning is done here at each iteration

        nruns = len(inf.models)

        t1 = Watch.get_time()
        tlearn = tlearn + t1-t0 

        # NOTE: When `compile_incremental()` is called, learning occurs only for the new iterations. 
        # However, when `inf.prob_necessity_sufficiency` is called, inference is indeed performed again with all the learned models.

        # Run the queries
        for query, infer_f in queries_f.items():
            if query not in exact:
                continue
            for cause in causes:
                exact_low = exact[query][cause][0]
                exact_upp = exact[query][cause][1]

                t2 = Watch.get_time()     
                approx_low, approx_upp = infer_f(cause, effect, true_false_cause=(0,1), true_false_effect=(0,1))
                tinfer = Watch.get_time()-t2 # TODO: Rafa, check if the times are computed correctly

                err = rrmse(exact_low, approx_low, exact_upp, approx_upp)
                err2 = rmse(exact_low, approx_low, exact_upp, approx_upp)

                # Store the results
                r = pd.DataFrame(dict(modelname=modelname, method=method, query = query, cause=cause, effect=effect, tol=tol,
                                num_runs_param=num_runs, max_iter_param=max_iter, seed=seed, tlearn=tlearn, tinfer=tinfer, remove_outliers=remove_outliers,
                                datasize=len(data),
                                exact_low = exact_low, exact_upp = exact_upp,
                                approx_low = approx_low, approx_upp = approx_upp,
                                rrmse = err, rmse = err2, nruns=nruns
                                ), index=[0])
                results = pd.concat([results, r], ignore_index=True)

        msg = f"[{nruns} runs\t T_learn={tlearn:.0f} ms."
        log.info(msg)
        t0 = Watch.get_time()
    # End Learning loop

    # Check if the directory for the file at resfilepath exists, and create it if it does not
    if not resfilepath.parent.exists():
        resfilepath.parent.mkdir(parents=True, exist_ok=True) 
    # Save results   
    results.to_csv(resfilepath)

def process_parameters_wrapper(params):
    # Define the logger
    log = get_clogger(__name__, fmt=log_format, filename=logfile)
    try:
        process_parameters(params, log)
    except Exception as e:
        log.exception(e)

def generate_parameter_combinations(modelpath):
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
    log = get_clogger(__name__, fmt=log_format, filename=logfile)
    available_workers = cpu_count()
    log.info(f"Number of available workers: {available_workers}")
    modelpaths = glob.glob(os.path.join('./papers/gradient_journal/models/synthetic/simple/', '*.uai'))
    n = len(modelpaths)
    for i, modelpath in enumerate(modelpaths):
        # e.g. modelpath = "./papers/gradient_journal/models/synthetic/s123/random_mc2_n5_mid3_d1000_05_mr098_r10_8.uai"

        # TODO: the line below is just for debug, comment for production
        modelpath = "./papers/gradient_journal/models/synthetic/simple/simple_nparents2_nzr02_zdr05_2.uai"

        model_name = os.path.basename(modelpath) # e.g. model_name = 'random_mc2_n5_mid3_d1000_05_mr098_r10_8.uai'
        log.info(f'Processing {model_name} ({i} out of {n-1}) ...')
        parameter_combinations = generate_parameter_combinations(modelpath)
        if not RUN_IN_PARALLEL: # set to True to test in non-parallel settings
            log.info(parameter_combinations[0])
            process_parameters_wrapper(parameter_combinations[1])
            quit()  
        else:
            # Parallel approach
            with Pool() as pool:
                pool.map(process_parameters_wrapper, parameter_combinations)





