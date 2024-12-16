import logging
import sys

import networkx as nx

import bcause.util.domainutils as dutils
import pandas as pd
from bcause.factors import DeterministicFactor
from bcause.factors.mulitnomial import MultinomialFactor, canonical_multinomial, random_multinomial
from bcause.inference.probabilistic.datainference import LaplaceInference
from bcause.inference.probabilistic.elimination import VariableElimination
from bcause.inference.causal.elimination import CausalVariableElimination
from bcause.models.cmodel import StructuralCausalModel
from bcause.util import graphutils
from bcause.util.datautils import to_counts
from bcause.util.equtils import seq_to_pandas

modelfile = "papers/gradient_journal/models/literature/pearl.uai"
datapath = "papers/gradient_journal/models/literature/pearl.csv"


def get_eq(m, domains, x):
    exovar = m.get_exogenous_parents(x)[0]
    right_endoVars = m.get_edogenous_parents(x)
    endoVars = right_endoVars + [x]
    dom = dutils.subdomain(domains, *endoVars)
    return canonical_multinomial(dom, exovar, right_endoVars).reorder(*endoVars)

full_model = StructuralCausalModel.read(modelfile)
model_names = dict(V0="T",V1="S", V2="G", V3="V", V4="U", V5="W")
full_model = full_model.rename_vars(model_names)
data_names = {k.replace("V", ""): v for k, v in model_names.items() if v in full_model.endogenous}
full_data = pd.read_csv(datapath, index_col=0)
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

data = full_data[["T","S"]]

modelfile = "papers/gradient_journal/models/literature/pearl_small.bif"
m.save(modelfile)


data.to_csv("papers/gradient_journal/models/literature/pearl_small.csv")



m = StructuralCausalModel.read(modelfile)



seq_to_pandas(m.factors["S"], exovar="U")
seq_to_pandas(m.factors["T"], exovar="V")

