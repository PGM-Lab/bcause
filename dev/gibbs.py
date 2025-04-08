import sys
import logging
import logging
import sys

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import dirichlet

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

import time


for i in range(max_iter):
    # sample U
    ve = VariableElimination(model)
    pu_ts = ve.query("U", conditioning= model.endogenous)

    #data = pd.DataFrame({'T':np.random.randint(2,size = 100000),'S':np.random.randint(2,size = 100000)} )
    #### Solution 1: default
    start = time.time()
    samples_u = [pu_ts.R(**obs).sample(1,"U")[0]["U"] for obs in data.to_dict(orient="records")]

    pu.get_value(U=0)
    fs = model.factors["S"]

    t = data.loc[0,"T"]
    s = data.loc[0, "S"]

    fs.get_value(U=0, T=t, S=s)


    # get the counts of U and update the parameters of the U
<<<<<<< HEAD

    w =random.uniform(0, 100)
    counts_u = [samples_u.count(u)*1 for u in model.domains["U"]]
=======
    counts_u = [samples_u.count(u) for u in model.domains["U"]]
    print(f'Time: {time.time() - start}')
>>>>>>> e095e74 (Update)
    total_counts_u += [counts_u]
    beta = [int(a + c) for a, c in zip(alpha, counts_u)]
    #alpha = beta

    #### Solution 2: SQL approach / alternativa agrupar datos y generar todos a la vez
    ## Extract Prob df from tuple dict
    start = time.time()
    pu_ts_df = pd.DataFrame(list(itertools.product(*pu_ts.domain.values())) , columns = pu_ts.domain.keys())
    pu_ts_df['Probability'] = pu_ts.values
    pu_ts_df = pu_ts_df.groupby(pu_ts.right_vars).agg({'Probability': list}).reset_index()

    ## Join assign U and prob to each data
    join_df = pu_ts_df.merge(data, on=pu_ts.right_vars, how='right')
    join_df['samples_U'] = join_df.apply(lambda row: np.random.choice(pu_ts.left_domain['U'],1, p = row['Probability'])[0],axis=1)
    counts_u = join_df['samples_U'].value_counts().to_dict()
    print(f'Time: {time.time() - start}')

    ### Solution 3: Multindex Series
    start = time.time()
    probability_table = pd.Series(pu_ts.values, index = pd.MultiIndex.from_product(pu_ts.domain.values(), names = pu_ts.domain.keys()),
                                  name='Probabilities').reorder_levels(pu_ts.right_vars + pu_ts.left_vars)
    counts_u = data.apply(lambda row: np.random.choice(pu_ts.left_domain['U'], size = 1, p = probability_table[tuple(row)].values)[0], axis=1).value_counts().to_dict()
    print(f'Time: {time.time() - start}')

    # sample the theta and set it to the model
    theta = dirichlet.rvs(beta)[0]
    theta_samples.append(list(theta))
    model.set_factor("U", MultinomialFactor(domU, theta))

    # run the query
    inf = CausalMultiInference([model])
    Q.append(inf.prob_sufficiency("T", "S")[0])

    if i % 10 == 0:
        print(f"{i}., current query = {Q[-1]}, true interval = {qtrue}, theta= {theta}")

    if i%100==0 and i>100:
        plt.hist(Q[100:],density=True)
        plt.xlim(0, 1)
        plt.show()
<<<<<<< HEAD




dirichlet.rvs([1,1,0.1])[0]
=======
>>>>>>> e095e74 (Update)
