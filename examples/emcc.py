import networkx as nx
import bcause as bc
from bcause.inference.causal.credal import EMCC

from bcause.models.cmodel import StructuralCausalModel

dag = nx.DiGraph([("V1", "V2"), ("V2", "V3"),("V3", "V4"),("U1", "V1"),("U2", "V2"),("U2", "V4"),("U3", "V3")])
model = StructuralCausalModel(dag)
domains = dict(V1=[0,1],V2=[0,1],V3=[0,1],V4=[0,1], U1=[0,1,3],U2=[0,1,2,3],U3=[0,1,2,3])
bc.randomUtil.seed(1)
model.fill_random_factors(domains)
data = model.sampleEndogenous(1000)




inf = EMCC(model, data, max_iter=10, num_runs=10)
inf.compile()


inf.prob_necessity_sufficiency(cause="V2", effect="V4")
inf.set_interval_result(True)
inf.prob_necessity(cause="V2", effect="V4")

inf.prob_sufficiency(cause="V2", effect="V4")



inf.causal_query("V3", do=dict(V2=1))


for m in inf.models:
    print(m.get_factors(*m.exogenous))

[len(inf.get_model_evolution(i)) for i in range(len(inf.models))]


for m in inf.get_model_evolution(0):
    print(m.get_factors(*m.exogenous))
