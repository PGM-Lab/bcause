import networkx as nx
import pandas as pd

from bcause.inference.causal.multi import EMCC, GibbsCausal
from bcause.models.cmodel import StructuralCausalModel

cause, effect = "V1","V2"
modelpath = "./papers/gradient_journal/models/synthetic/s123/random_mc2_n5_mid3_d1000_05_mr098_r10_2.uai"
modelpath = "./papers/gradient_journal/models/synthetic/s123/random_mc2_n5_mid3_d1000_05_mr098_r10_4.uai"

# load the model
model = StructuralCausalModel.read(modelpath)

# Load data
datapath =  modelpath.replace(".uai", ".csv")
data = pd.read_csv(datapath, index_col=0)
data = data.rename(columns={c: "V" + c for c in data.columns})

# Load the information about the query and the model
modelpath_ccve = modelpath.replace(".uai", "_uai_ccve.csv")
info_query = pd.read_csv(modelpath_ccve)

pns_exact = (info_query.pns_l.values[0], info_query.pns_u.values[0])
cause, effect = [f"V{i}" for i in list(info_query[["cause", "effect"]].values.flatten())]


#inf = EMCC(model, data, num_runs=2, max_iter=5)
inf = GibbsCausal(model, data, num_runs=2, burnin_iter=2)

inf.compile()
p = inf.prob_necessity_sufficiency(cause, effect, true_false_cause=(0, 1), true_false_effect=(0, 1))

