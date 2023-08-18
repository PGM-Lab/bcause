import networkx as nx
import bcause as bc
from bcause.inference.causal.credal import EMCC
from bcause.models import BayesianNetwork

from bcause.models.cmodel import StructuralCausalModel

# dag = nx.DiGraph([("V1", "V2"), ("V2", "V3"),("V3", "V4"),("U1", "V1"),("U2", "V2"),("U2", "V4"),("U3", "V3")])
# model = StructuralCausalModel(dag)
# domains = dict(V1=[0,1],V2=[0,1],V3=[0,1],V4=[0,1], U1=[0,1,3],U2=[0,1,2,3],U3=[0,1,2,3])
# bc.randomUtil.seed(1)
# model.fill_random_factors(domains)

from bcause.readwrite import bnread

bnet = BayesianNetwork.read("./models/cofounded4_bnet.uai")
bnet = bnread.from_uai("./models/cofounded4_bnet.uai", reverse_values=False)



model = StructuralCausalModel.from_model(bnet)

model.get_qbnet().factors["V3"] # [0.983,0.017,0.921,0.07900000000000001], [0.15699999999999995, 0.979, 0.843, 0.021000000000000008]

data = model.sampleEndogenous(1000)




inf = EMCC(model, data, max_iter=10, num_runs=2)
inf.compile()








model.endogenous
inf.causal_query("V3", do=dict(V1=0))    # same prob regardles of intervention
inf.causal_query("V3", do=dict(V1=0))

'''
Causal query:K(vars[3]|[])
3: 0,1,
K(vars[3]|[]) [0.2111370079565275, 0.7888629920434724]
              [0.20595916954829108, 0.7940408304517088]


'''

