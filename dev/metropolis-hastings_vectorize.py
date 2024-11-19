import itertools
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import dirichlet
import random

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


modelfile = "/Users/antoniogonzalezalves/PycharmProjects/bcause/models/literature/pearl.uai"
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

# Uniform perturbation
def uniform_proposal(num_states):
    return np.full((num_states, num_states), 1 / num_states)

# Random asymmetric proposal for matrix Q(U' | U)
def generate_random_proposal(alpha):
    return np.array([dirichlet.rvs(alpha)[0] for _ in range(4)])

def perturbation_proposal(u, CondMatrix):
    delta = np.random.choice([0, 1, 2, 3], p = CondMatrix[u])
    return (u + delta) % 4

# Asymmetric proposal adding noise
def add_noise_to_Q(Q, noise_scale=0.1):
    noisy_Q = Q + np.random.normal(0, noise_scale, Q.shape)  # Add Gaussian noise
    noisy_Q = np.clip(noisy_Q, 0, None)  # Clip negative values to zero
    #noisy_Q /= noisy_Q.sum(axis=1, keepdims=True)  # Normalize rows to sum to 1
    for i in range(noisy_Q.shape[0]):  # Iterate over each row
        row_sum = np.sum(noisy_Q[i, :])  # Sum of the i-th row
        if row_sum > 0:  # Avoid division by zero
            noisy_Q[i, :] /= row_sum  # Normalize the i-th row
        else:
            noisy_Q[i, :] = 1 / noisy_Q.shape[1]
    noisy_Q = np.nan_to_num(noisy_Q, nan=0, posinf=0, neginf=0)
    return noisy_Q

# No importa, es por si luego queremos ajustar el ruido en función de la aceptación
def adjust_noise_scale(noise_scale, acceptance_rate, target_range=(0.2, 0.3), adjustment_factor=0.1):
    target_min, target_max = target_range

    if acceptance_rate < target_min:
        # Increase acceptance by reducing noise scale
        noise_scale *= (1 - adjustment_factor)
    elif acceptance_rate > target_max:
        # Decrease acceptance by increasing noise scale
        noise_scale *= (1 + adjustment_factor)

    return max(noise_scale, 1e-5)

max_iter = 10000

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

noise_scale = 0.15 # Only applies to perturbation w/ noise

ps = model.get_factors("S")[0]

# Sampling function to always get compatible u
def sample_valid_u(data):
    samples_u = []
    for i in range(len(data)):
        u = np.random.choice(domU["U"], p = pu.values)
        while ps.get_value(S = data.loc[i,"S"], T = data.loc[i,"T"], U = u) == 0:
            u = np.random.choice(domU["U"], p = pu.values)
        samples_u.append(u)
    return samples_u

# Wrapper function to vectorize ps and pu get value
def get_value_vectorized(S, T, U):
    return ps.get_value(S=S, T=T, U=U)

def get_value_u_vectorized(U):
    return pu.get_value(U = U)

# Vectorize the function
ps_get_value_vec = np.vectorize(get_value_vectorized)
pu_get_value_vec = np.vectorize(get_value_u_vectorized)

# Generate the initial sampling for U
samples_u = np.array(sample_valid_u(data))
for i in range(max_iter):
    n_accept = 0
    pu = model.factors["U"]
    ConditionalMatrix = uniform_proposal(len(domU["U"])) # Uniform Proposal option
    #ConditionalMatrix = add_noise_to_Q(ConditionalMatrix, noise_scale=noise_scale) \
    #    if i != 0 else uniform_proposal(len(domU["U"])) # Add gaussian noise option
    #ConditionalMatrix = generate_random_proposal(alpha) # Random proposal

    # Generas las u de golpe
    u_new_all = np.array([perturbation_proposal(u, ConditionalMatrix) for u in samples_u])

    # Fetch S and T values
    S_values = data["S"].to_numpy()
    T_values = data["T"].to_numpy()

    prob_new = (
            ps_get_value_vec(S=S_values, T=T_values, U=u_new_all)  # Vectorized function
            * pu_get_value_vec(U=u_new_all)  # Vectorized function
            * ConditionalMatrix[samples_u, u_new_all]
    )

    prob_old = (
            ps_get_value_vec(S=S_values, T=T_values, U=samples_u)
            * pu_get_value_vec(U=samples_u)
            * np.maximum(ConditionalMatrix[u_new_all, samples_u], 1e-3)
    )


    # Compute acceptance ratios
    ratios = np.minimum(1, prob_new / prob_old)

     # Generate acceptance decisions
    acceptances = np.random.uniform(0, 1, size=len(ratios)) < ratios

     # Update accepted samples
    samples_u[acceptances] = u_new_all[acceptances]
    n_accept += np.sum(acceptances)

    # get the counts of U and update the parameters of the U
    counts_u = [np.count_nonzero(samples_u == u) for u in model.domains["U"]]

    total_counts_u += [counts_u]
    beta = [int(a + c) for a, c in zip(alpha, counts_u)]

    # sample the theta and set it to the model
    theta = dirichlet.rvs(beta)[0]
    theta_samples.append(list(theta))
    model.set_factor("U", MultinomialFactor(domU, theta))

    # run the query
    inf = CausalMultiInference([model])
    Q.append(inf.prob_sufficiency("T", "S")[0])

    if i % 10 == 0:
        print(f"{i}., current query = {Q[-1]}, true interval = {qtrue}, theta= {theta}, N aceptados = {n_accept}")

    if i%100==0 and i>100:
        plt.hist(Q[100:],density=True)
        plt.xlim(0, 1)
        plt.show()
