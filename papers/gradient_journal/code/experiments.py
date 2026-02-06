import logging
import math
import os
from pathlib import Path
import argparse

import networkx as nx
import pandas as pd
import sys

from bcause.inference.probabilistic.elimination import VariableElimination
from bcause.models.cmodel import StructuralCausalModel
from bcause.util.mathutils import rrmse, rmse

sys.path.insert(0, "../../../")

from bcause.inference.causal.multi import EMCC, GibbsCausal, GDCC
from bcause.util.runningutils import get_logger
from bcause.util.watch import Watch

'''

# model with 
# -m EMCC_5 -n 3 -s 1234 -rw -ro  ./papers/gradient_journal/models/synthetic/s123/random_mc2_n5_mid3_d1000_05_mr098_r10_2.uai
-m EMCC_5 -n 10 -s 1234 -rw -ro ./papers/gradient_journal/models/synthetic/s123/random_mc2_n5_mid3_d1000_05_mr098_r10_4.uai

-m GDCC_1e-02 -n 3 -s 1234 -rw -ro  ./papers/gradient_journal/models/synthetic/s123/random_mc2_n5_mid3_d1000_05_mr098_r10_4.uai

'''


# CL arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', help="Specify the method.")
parser.add_argument('-n', '--numruns', help="Number of runs.", default=10)
parser.add_argument('-s', '--seed', help="Random seed.", default=0)
parser.add_argument('-o', '--output', help="Results folder.", default=".")
parser.add_argument('-rw', '--rewrite', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('-ro', '--removeoutliers', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('filepath', metavar='N', type=str, nargs='+', help='UAI model.')

args = parser.parse_args()
variables = vars(args)

#variables = dict(method="EMCC_100", filepath=["/Users/rcabanas/GoogleDrive/UAL/research/causality/dev/bcause/papers/gradient_journal/models/synthetic/s123/random_mc2_n5_mid3_d1000_05_mr098_r10_2.uai"],
#                 output=".", numruns=2, seed=0, rewrite=True, removeoutliers=True)

method = variables["method"]
modelpath = variables["filepath"][0]
resfolder = variables["output"]
num_runs = int(variables["numruns"])
seed = int(variables["seed"])
rewrite = variables["rewrite"]
remove_outliers = variables["removeoutliers"]







#constant
modelname = modelpath.split("/")[-1].replace(".uai","")
label = f"{modelname}_uai_{method}_n{num_runs}_ro{remove_outliers}_s{seed}"

# Define the logger
fmt = f'%(asctime)s|%(levelname)s|{label}: %(message)s'
log = get_logger("experiments", fmt=fmt)
log.propagate = 0
#log
log.setLevel("INFO")


log.info(f"Arguments: {args}")

log.info(f"Loading model from {modelpath}")

# load the model
model = StructuralCausalModel.read(modelpath)

# Load data
datapath =  modelpath.replace(".uai", ".csv")
log.info(f"Loading data from {datapath}")
data = pd.read_csv(datapath, index_col=0)
data = data.rename(columns={c: "V" + c for c in data.columns})

# Load the information about the query and the model
modelpath_ccve = modelpath.replace(".uai", "_uai_ccve.csv")

info_query = pd.read_csv(modelpath_ccve)

pns_exact = (info_query.pns_l.values[0], info_query.pns_u.values[0])
cause, effect = [f"V{i}" for i in list(info_query[["cause", "effect"]].values.flatten())]
modelname = os.path.basename(modelpath).split(".")[0]
log.info(f"PNS({cause},{effect}) exact: {pns_exact}")


if not nx.has_path(model.graph, cause, effect):
    log.error("Model does not have any cause-effect path.")
    sys.exit(1)



max_iter = int(method.split("_")[-1]) if method.startswith("EMCC") else None
burn_iter = 100#int(method.split("_")[-1]) if method.startswith("GSCC") else None
tol = float(method.split("_")[-1]) if method.startswith("GDCC") else None


# Set the results
results = pd.DataFrame()
if not os.path.exists(resfolder):
    os.makedirs(resfolder)

resfilepath = Path(resfolder, f"{label}.csv")
if (not rewrite) and os.path.exists(resfilepath):
    log.error(f"File exists, not rewriting: {resfilepath}")
    sys.exit(1)
    #raise ValueError("File exists, not rewriting.")



# Determine the method
if method.startswith("GDCC"):
    inf = GDCC(model, data, num_runs=num_runs, tol = tol, outliers_removal=remove_outliers)
elif method.startswith("EMCC"):
    inf = EMCC(model, data, num_runs=num_runs, max_iter=max_iter, outliers_removal=remove_outliers)
elif method.startswith("GSCC"):
    inf = GibbsCausal(model, data, num_runs=num_runs, burnin_iter=100, outliers_removal=remove_outliers)
else:
    raise ValueError("Wrong learning method")





### Start processing ###
tlearn = 0
t0 = 0
Watch.start()

print(model)
print(data)
# Learning loop
for _ in inf.compile_incremental(1): # The learning is done here at each iteration

    t1 = Watch.get_time()
    # Run the query
    p = inf.prob_necessity_sufficiency(cause,effect, true_false_cause=(0,1), true_false_effect=(0,1))
    t2 = Watch.get_time()

    tlearn = tlearn + t1-t0
    tinfer = t2-t1

    err = rrmse(pns_exact[0], p[0], pns_exact[1], p[1])
    err2 = rmse(pns_exact[0], p[0], pns_exact[1], p[1])

    nruns = len(inf.models)

    msg = f"[{p[0]:.4f},{p[1]:.4f}]\t {nruns} runs\t rmse={err2:.5f}\t T_learn={tlearn:.0f} ms. \t T_infer={tinfer:.0f} ms."
    log.info(msg)

    # Save the results
    r = pd.DataFrame(dict(modelname=modelname, method=method, cause=cause, effect=effect, tol=tol,
                    num_runs_param=num_runs, max_iter_param=max_iter, seed=seed, tlearn=tlearn, tinfer=tinfer, remove_outliers=remove_outliers,
                    datasize=len(data),
                    pns_low_exact =pns_exact[0], pns_upp_exact = pns_exact[1],
                    pns_low = p[0], pns_upp = p[1], rrmse = err, rmse = err2, nruns=nruns
                    ), index=[0])

    results = pd.concat([results, r], ignore_index=True)
    results.to_csv(resfilepath)
    log.info(f"Saving results to {resfilepath}")

    t0 = Watch.get_time()




