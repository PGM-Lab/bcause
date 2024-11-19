import itertools
import logging
import sys

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

max_iter = 1000

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

for i in range(max_iter):
    # sample U
    ve = VariableElimination(model)
    pu_ts = ve.query("U", conditioning= model.endogenous)

    #data = pd.DataFrame({'T':np.random.randint(2,size = 100000),'S':np.random.randint(2,size = 100000)} )
    #### Solution 1: default
    #samples_u = [pu_ts.R(**obs).sample(1,"U")[0]["U"] for obs in data.to_dict(orient="records")]

    # get the counts of U and update the parameters of the U
    #counts_u = [samples_u.count(u) for u in model.domains["U"]]
    #total_counts_u += [counts_u]
    #beta = [int(a + c) for a, c in zip(alpha, counts_u)]

    #### Solution 2: SQL approach / alternativa agrupar datos y generar todos a la vez
    ## Extract Prob df from tuple dict

    pu_ts_df = pd.DataFrame(list(itertools.product(*pu_ts.right_domain.values())), columns=pu_ts.right_domain.keys())
    pu_ts_df['Probability'] = list(map(lambda obs: pu_ts.R(**obs).values, pu_ts_df.to_dict(orient='records')))
    count_df =data.groupby(pu_ts.right_vars).size().reset_index(name = 'count')
    join_df =  pu_ts_df.merge(count_df, on=pu_ts.right_vars, how='left')

    samples_u = np.concatenate(join_df.apply(lambda row: np.random.choice(pu_ts.left_domain['U'],
                                     size = row['count'], p = row['Probability']),axis=1).to_list())

    counts_u = np.bincount(samples_u)
    beta = [int(a + c) for a, c in zip(alpha, counts_u)]

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
[]