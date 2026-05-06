import random
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import dirichlet
from sympy.physics.units import length

import bcause.util.domainutils as dutils
from bcause.factors.multinomial import MultinomialFactor, canonical_multinomial, random_multinomial
from bcause.inference.causal.multi import CausalMultiInference
from bcause.inference.probabilistic.datainference import LaplaceInference
from bcause.inference.probabilistic.elimination import VariableElimination
from bcause.models.cmodel import StructuralCausalModel
from bcause.util import graphutils
from bcause.util.datautils import to_counts
from bcause.util.equtils import seq_to_pandas
from bcause.util.runningutils import get_logger

log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'

import logging


logging.disable(logging.CRITICAL)


####### Model definition #########

def get_eq(m, domains, x):
    exovar = m.get_exogenous_parents(x)[0]
    right_endoVars = m.get_edogenous_parents(x)
    endoVars = right_endoVars + [x]
    dom = dutils.subdomain(domains, *endoVars)
    return canonical_multinomial(dom, exovar, right_endoVars).reorder(*endoVars)


modelfile = "./models/literature/pearl.uai"
full_model = StructuralCausalModel.read(modelfile)
model_names = dict(V0="T",V1="S", V2="G", V3="V", V4="U", V5="W")
full_model = full_model.rename_vars(model_names)
data_names = {k.replace("V", ""): v for k, v in model_names.items() if v in full_model.endogenous}
full_data = pd.read_csv(modelfile.replace(".uai", ".csv"), index_col=0)
full_data = full_data.rename(columns=data_names)

dag = graphutils.remove_nodes(full_model.graph, ["G","W"])
m = StructuralCausalModel(dag)
endo_dom = {k:v for k,v in full_model.domains.items() if k in ["T","S"]}
u = "U"
for x in m.endogenous: m.set_factor(x,  get_eq(m,endo_dom, x))
for u in m.exogenous:
    d = m.factors[m.get_edogenous_children(u)[0]].domain[u]
    f = random_multinomial({u:d})
    m.set_factor(u,f)

print("\nSEs")
print(seq_to_pandas(m.factors["T"], exovar="V"))
print(seq_to_pandas(m.factors["S"], exovar="U"))

data = full_data[["T","S"]]
to_counts({x:d for x,d in m.domains.items() if x in m.endogenous}, data).values_dict

datainf = LaplaceInference(data, endo_dom)
for x in m.endogenous:
    print(datainf.query(x, conditioning=m.get_edogenous_parents(x)).values_dict)


qtrue = [0.3,1.0]

pv = MultinomialFactor(dict(V=m.domains["V"]), datainf.query("T").values)
m.set_factor("V",pv)





############### Sampling algorithm ##############

max_iter = 50000

model = m.copy()
cardU = len(model.domains["U"])
domU = dict(U=model.domains["U"])



Q = []
theta_samples = []
total_counts_u = []

# sample theta
alpha = np.ones(cardU)
theta = dirichlet.rvs(alpha)[0]
pu = MultinomialFactor(domU, theta)
model.set_factor("U", pu)
#max_iter = 5


# Sample an initial valid configuration
ps = model.get_factors("S")[0]

def sample_valid_u(data):
    samples_u = []
    for i in range(len(data)):
        u = np.random.choice(domU["U"], p = pu.values)
        while ps.get_value(S = data.loc[i,"S"], T = data.loc[i,"T"], U = u) == 0:
            u = np.random.choice(domU["U"], p = pu.values)
        samples_u.append(u)
    return samples_u

samples_u = np.array(sample_valid_u(data))


fs = model.factors["S"]

for i in range(max_iter):
    for n in range(len(data)):
        u_n = samples_u[n] #last u value in position n
        t_n = data.loc[n, "T"]
        s_n = data.loc[n, "S"]


        samples_no_n = np.concatenate([samples_u[:n],samples_u[n+1:]])

        # Calculate a weight for each possible u
        weights = [fs.get_value(U=u, T=t_n, S=s_n) * (alpha[u] + sum(samples_no_n == u)) for u in domU["U"]]

        # Sample a new value for u in position n
        sampling_probs =  weights / sum(weights)
        samples_u[n] = np.random.choice(domU["U"], p = sampling_probs)

    counts_u = [sum(samples_u==u) for u in model.domains["U"]]
    total_counts_u += [counts_u]
    beta = [int(a + c) for a, c in zip(alpha, counts_u)]
    #alpha = beta

    # sample the theta and set it to the model
    theta = dirichlet.rvs(beta)[0]
    theta_samples.append(list(theta))
    model.set_factor("U", MultinomialFactor(domU, theta))

    # run the query
    inf = CausalMultiInference([model])
    inf._model = model
    Q.append(inf.prob_sufficiency("T", "S")[0])




    if i % 10 == 0:
        msg = f"{i}., current query = {Q[-1]}, , theta= {theta}, true interval = {qtrue}"
        if i>100:
            qapprox = [min(Q[100:]), max(Q[100:])]
            msg += f"approx interval = {qapprox}"
        print(msg)

    if i%100==0 and i>100:
        plt.hist(Q[100:],density=True)
        plt.xlim(0, 1)
        plt.show()



