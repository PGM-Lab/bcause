import itertools

import networkx as nx
import bcause as bc
from bcause.models.cmodel import StructuralCausalModel

dag = nx.DiGraph([("V1", "V2"), ("V2", "V3"),("V3", "V4"),("U1", "V1"),("U2", "V2"),("U2", "V4"),("U3", "V3")])
model = StructuralCausalModel(dag)
domains = dict(V1=[0,1],V2=[0,1],V3=[0,1],V4=[0,1], U1=[0,1,3],U2=[0,1,2,3],U3=[0,1,2,3])
bc.randomUtil.seed(1)
model.fill_random_factors(domains)
data = model.sample(1000, as_pandas=True)



U="U2"

# endogenous children
V = model.get_edogenous_children(U)

# get the endogenous parents
Y = list(itertools.chain(*[model.get_edogenous_parents(v) for v in V]))

# remove the variables in V
Y = [v for v in Y if v not in V]

# get the joint domains of Y union V
domYV = model.get_domains(Y+V)


from bcause.util.domainutils import assingment_space, state_space

# get all possible assignments in a domain
assingment_space(domYV)

# get all possible states in a domain
state_space(domYV)



# maximum log-likelihood for a model wrt a dataset
model.max_log_likelihood(data)

# log-likelihood of a model wrt a dataset
model.log_likelihood(data)


# decomposition of the max-log-likelihood for each endogenous c-component
model.max_log_likelihood(data, variables=["V1"]) + \
model.max_log_likelihood(data, variables=["V2","V4"]) + \
model.max_log_likelihood(data, variables=["V3"])

# decomposition of the log-likelihood for each endogenous c-component
model.log_likelihood(data, variables=["V1"]) + \
model.log_likelihood(data, variables=["V2","V4"]) + \
model.log_likelihood(data, variables=["V3"])

