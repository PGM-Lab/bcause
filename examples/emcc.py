import networkx as nx
import bcause as bc
from bcause.factors import DeterministicFactor
from bcause.factors.factor import Factor
from bcause.inference.causal.credal import EMCC
from bcause.inference.causal.elimination import CausalVariableElimination
from bcause.models import BayesianNetwork

from bcause.models.cmodel import StructuralCausalModel



dag = nx.DiGraph([("V0", "V1"), ("V1", "V2"),("V2", "V3"),("U0", "V0"),("U1", "V1"),("U1", "V3"),("U2", "V2")])
model = StructuralCausalModel(dag)
domains = dict(V0=[0,1],V1=[0,1],V2=[0,1],V3=[0,1], U0=[0,1],U1=[0,1,2,3,4,5,6,7,8,9,10,11],U2=[0,1,2,3,4,5,6,7])
bc.randomUtil.seed(1)
model.fill_random_factors(domains)




#
# P([4]) [0.4352219037002618, 0.5647780962997382]
# P([5]) [0.04591629162655201, 0.09990496095808442, 0.12146122722240116, 0.0026614525347146392, 0.10916684228031584, 0.1439067661770072, 0.10887889218653227, 0.04081239361575789, 0.1148620742003469, 0.015022252325027626, 0.06518095379288001, 0.13222589308038002]
# P([6]) [0.1234519168422001, 0.12099695251464455, 0.054671331613850155, 0.008142959342161707, 0.2854206727646181, 0.08898037346844956, 0.11165076442032522, 0.2066850290337506]
#


for u in model.exogenous:
    print(model.factors[u])

#
# P([0, 4]) [0.0, 1.0, 0.0, 1.0]
# P([1, 0, 5]) [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
# P([2, 1, 6]) [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
# P([3, 2, 5]) [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
# ---


model.get_qbnet().factors["V0"]


for x in model.exogenous:
    print(model.factors[x])



model.save("./models/cofounded4.uai")


model._writer.to_uai(model, "./models/cofounded4_rf.uai", reverse_values=False)



cve = CausalVariableElimination(model)


#cve.prob_necessity_sufficiency(f"V0",f"V2", true_false_cause=(0,1), true_false_effect=(0,1))

#
# cve.causal_query("V3", do=dict(V1=0))
# cve.causal_query("V3", do=dict(V1=1))
# cve.query("V3", evidence=dict(V1=0), do=dict())
#
for x in range(0,3):
    for y in range(x+1, 4):
        ##r = cve.prob_necessity_sufficiency(f"V{x}",f"V{y}", true_false_cause=(0,1), true_false_effect=(0,1))
        r = cve.prob_sufficiency(f"V{x}", f"V{y}", true_false_cause=(1, 0), true_false_effect=(1, 0))
        print(f"P(V{x},{y}) = {r}")
#         r = cve.causal_query(f"V{y}", do={f"V{x}":0})
#         print(f"p({y}, | do{y}) = {r}")
#


#
# P(V0,1) = 0.0
# P(V0,2) = 0.0
# WARNING:root:invalid value encountered in true_divide: P(V3_1)/P()
# P(V0,3) = 0.0
# P(V1,2) = 0.3898750354752074
# P(V1,3) = 0.2693966041543566
# P(V2,3) = 1.0



#
# r = cve.causal_query("V2", do=dict(V0=1))
# print(r)
# r = cve.counterfactual_query("V2", evidence=dict(V3=0), do=dict(V0=1))
# print(r)
# r = cve.causal_query("V2", do=dict(V1=1))
# print(r)
# r = cve.counterfactual_query("V2", evidence=dict(V3=0), do=dict(V1=1))
# print(r)
# r = cve.causal_query("V2", evidence=dict(V3=0), do=dict(V1=1))
#
# print(r)
#
#
#
#
#
#
# # 01:P([1]) [0.36667667419362127, 0.6333233258063787]
# # 02:P([2]) [0.4721777040013881, 0.5278222959986117]
# # 03:P([3]) [0.3978210492710605, 0.6021789507289395]
# # 12:P([2]) [0.626992874932074, 0.373007125067926]
# # 13:P([3]) [0.3907133004653417, 0.6092866995346583]
# # 23:P([3]) [0.4622481569696807, 0.5377518430303193]
#
#
#
# model.get_qbnet().factors["V3"].values
#
# #[
# # P([0, 1, 2, 3]) [0.0, 0.4213894344145197, 0.0, 0.6857255540613334, 0.0, 0.3167716504819647, 0.0, 0.017218307172987786, 0.0, 0.5786105655854802, 0.0, 0.3142744459386666, 0.0, 0.6832283495180353, 0.0, 0.9827816928270122],
# # P([1, 2]) [0.626992874932074, 0.38254400557522933, 0.373007125067926, 0.6174559944247706],
# # P([0, 1]) [0.0, 0.8454289084300954, 0.0, 0.15457109156990456], P([0]) [0.0, 1.0]]
#

#
# data = model.sampleEndogenous(1000)
#
#
#
#
# inf = EMCC(model, data, max_iter=100, num_runs=20)
# inf.compile()
#
#
#
#
#
#
# # model.intervention(**dict(V1=0))
# #
# # model.endogenous
# inf.causal_query("V3", do=dict(V1=0))    # same prob regardles of intervention
# inf.causal_query("V3", do=dict(V1=1))
#
#
# inf.prob_necessity("V0","V3", true_false_cause=(0,1), true_false_effect=(0,1))
# inf.prob_necessity("V1","V3", true_false_cause=(0,1), true_false_effect=(0,1))
#
# inf.prob_necessity_sufficiency("V0","V3", true_false_cause=(0,1), true_false_effect=(0,1))
# inf.prob_necessity_sufficiency("V1","V3", true_false_cause=(0,1), true_false_effect=(0,1))
#
#
#
# '''
# PN(0,3):K(vars[]|[])
# K(vars[]|[]) [0.7167128645208563]
#              [0.9999630294545191]
#
# PN(1,3):K(vars[]|[])
# K(vars[]|[]) [0.0]
#              [0.0]
#
# PNS(0,3):K(vars[]|[])
# K(vars[]|[]) [6.287869002822805E-4]
#              [0.017130735211889268]
#
# PNS(1,3):K(vars[]|[])
# K(vars[]|[]) [3.7897215616601986E-20]
#              [1.8611718226896523E-17]
#
# '''