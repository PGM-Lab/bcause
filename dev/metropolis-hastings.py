import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import dirichlet
from scipy.stats import uniform

import bcause.util.domainutils as dutils
from bcause.factors.mulitnomial import MultinomialFactor, canonical_multinomial, random_multinomial
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
import time

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
model_names = dict(V0="T", V1="S", V2="G", V3="V", V4="U", V5="W")
full_model = full_model.rename_vars(model_names)
data_names = {k.replace("V", ""): v for k, v in model_names.items() if v in full_model.endogenous}
full_data = pd.read_csv(modelfile.replace(".uai", ".csv"), index_col=0)
full_data = full_data.rename(columns=data_names)

dag = graphutils.remove_nodes(full_model.graph, ["G", "W"])
m = StructuralCausalModel(dag)
endo_dom = {k: v for k, v in full_model.domains.items() if k in ["T", "S"]}
u = "U"
for x in m.endogenous: m.set_factor(x, get_eq(m, endo_dom, x))
for u in m.exogenous:
    d = m.factors[m.get_edogenous_children(u)[0]].domain[u]
    f = random_multinomial({u: d})
    m.set_factor(u, f)

print("\nSEs")
print(seq_to_pandas(m.factors["T"], exovar="V"))
print(seq_to_pandas(m.factors["S"], exovar="U"))

data = full_data[["T", "S"]]
to_counts({x: d for x, d in m.domains.items() if x in m.endogenous}, data).values_dict

datainf = LaplaceInference(data, endo_dom)
for x in m.endogenous:
    print(datainf.query(x, conditioning=m.get_edogenous_parents(x)).values_dict)

qtrue = [0.3, 1.0]

pv = MultinomialFactor(dict(V=m.domains["V"]), datainf.query("T").values)
m.set_factor("V", pv)

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

sigma = 1


def logit(p):
    return np.log(p/(1-p))

def inv_logit(p):
    return 1/(1 + np.exp(-p))
def get_post(model, theta, data, alpha, domU):
    model.set_factor("U", MultinomialFactor(domU, theta))
    ve = VariableElimination(model)
    pu_ts = ve.query("U", conditioning=model.endogenous)

    samples_u = [pu_ts.R(**obs).sample(1, "U")[0]["U"] for obs in data.to_dict(orient="records")]
    counts_u = [samples_u.count(u) for u in model.domains["U"]]
    beta = [int(a + c) for a, c in zip(alpha, counts_u)]

    return dirichlet.pdf(theta, beta)

thetas = []
#theta_current = dirichlet.rvs(alpha)[0]
theta_current = np.full(cardU,1/cardU)
acceptances = 0
target_accept_rate = 0.3

post_current = get_post(model, theta_current,data,alpha,domU)

for i in range(max_iter):
    #theta_prop = theta_current + np.random.normal(0, 0.05, size=cardU)
    #theta_prop = theta_prop / np.sum(theta_prop)

    #theta_prop = inv_logit(np.random.normal(logit(theta_current), 1, size=cardU))
    #if all(theta_prop == 0):
    #    theta_prop = theta_current
    #theta_prop = np.maximum(theta_prop, 1e-10)

    theta_prop = uniform.rvs(size=4)
    theta_prop = theta_prop / np.sum(theta_prop)


    if any(theta_prop < 0):
        theta_prop = theta_current

    post_prop = get_post(model, theta_prop, data, alpha, domU)

    alpha_ratio = min(1, post_prop / post_current )

    if np.random.uniform(0,1) < alpha_ratio:
        thetas.append(theta_prop)
        theta_current = theta_prop
        post_current = post_prop
        acceptances +=1

        model.set_factor("U", MultinomialFactor(domU, theta_current))
        # run the query
        inf = CausalMultiInference([model])
        Q.append(inf.prob_sufficiency("T", "S")[0])

        if acceptances % 10 == 0:
            print(f"{i}., current query = {Q[-1]}, true interval = {qtrue}, theta= {theta_current}")
            print(alpha_ratio)

        if (acceptances > 100) and (acceptances % 10 == 0):
            plt.hist(Q, density=True)
            plt.xlim(0, 1)
            plt.show()

    if (i + 1) % 100 == 0:  # Adjust every 100 iterations
        accept_rate = acceptances / 100
        if accept_rate > target_accept_rate:
            sigma *= 1.1  # Increase sigma
        else:
            sigma *= 0.9  # Decrease sigma
        acceptances = 0  # Reset counter

    if (i + 1) % 1000 == 0:
        print('check')
