import pyAgrum as gum
import networkx as nx
import bcause.factors as bfd

from pgmpy.models import BayesianNetwork
from itertools import starmap
from bcause.models import BayesianNetwork
from bcause.models.cmodel import StructuralCausalModel
from bcause.util import randomUtil
import bcause.util.domainutils as dutils
import bcause.util.graphutils as gutils

def addGraph(bn: gum.BayesNet, nodes: list[str], card: dict ,  edges: list[tuple[str, str]]):
    # add Nodes
    list(map(lambda node: bn.add(node, card[node]), nodes))
    # add Edges
    list(starmap(bn.addArc, edges))

# Add CPTs
def setCPT(f: bfd.MultinomialFactor, bn: gum.BayesNet):
    v = list(f.left_domain.keys())
    rv = list(f.right_domain.keys())

    # Check nodes
    [bn.add(i) for i in v+rv if not bn.exists(i)]
    # Check edges
    [bn.addArc(i, *v) for i in rv if not bn.existsArc(i, *v)]

    # Set the CPT
    values = f.values_array()
    shape = [len(f.domain[var]) for var in bn.cpt(*v).names[::-1]]
    values = values.reshape(*shape)
    bn.cpt(f.left_vars[0])[:] = values

def toAgrum(bn: BayesianNetwork) -> gum.BayesNet:
    gumbn = gum.BayesNet()
    card = {v: len(d) for v,d in bn.domains.items()}

    addGraph(gumbn, bn.variables, card,list(bn.graph.edges))
    [setCPT(f,gumbn) for f in bn.factors.values()]
    return gumbn

def createFactor(bn: gum.BayesNet,domains, dag, node: str) -> bfd.MultinomialFactor:
    parents = [bn.variable(i).name() for i in bn.parents(bn.idFromName(node))]
    dom = dutils.subdomain(domains, *gutils.relevat_vars(dag, node))
    values = bn.cpt(node).toarray().flatten()
    return bfd.MultinomialFactor(dom, values, right_vars=parents)

def value_from_var_domain(var) -> list:
    return list(range(var.minVal(), var.maxVal() + 1))

def fromAgrum(bn: gum.BayesNet) -> BayesianNetwork:
    # Create a DiGraph from the arcs
    dag = nx.DiGraph(bn.arcs())
    # if the graph is empty, create a DiGraph with the nodes
    if len(dag.edges) == 0:
        dag = nx.DiGraph()
        dag.add_nodes_from(bn.nodes())
    # create a dictionary of keys nodes id and value its name
    nodes = {i: bn.variable(i).name() for i in range(bn.size())}
    # Change name ot the nodes in dag
    dag = nx.relabel_nodes(dag, nodes)
    # Get the domains with keys the name of the nodes and the values a list of the possible values of the node
    domains = {bn.variable(i).name(): value_from_var_domain(bn.variable(i)) for i in range(bn.size())}
    # Base on the cpt tables, create the factors to then use it in the MultinomialFactor function
    factors = {}
    for i in range(bn.size()):
        node = bn.variable(i).name()
        factors[node] = createFactor(bn,domains,dag, node)
    # Return the BayesianNetwork with the dag and the factors
    return BayesianNetwork(dag, factors)


## Example of use

# Define a DAG and the domains
dag = nx.DiGraph([("V1", "V2"), ("V2", "V3"),("V3", "V4"),("U1", "V1"),("U2", "V2"),("U2", "V4"),("U3", "V3")])
model = StructuralCausalModel(dag)
domains = dict(V1=[0,1],V2=[0,1],V3=[0,1],V4=[0,1], U1=[0,1],U2=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],U3=[0,1,2,3])
randomUtil.seed(1)
model.fill_random_factors(domains)
data = model.sample(1000, as_pandas=True)

bn = BayesianNetwork(dag,model.factor_list)

gumbn = toAgrum(bn)
newbn = fromAgrum(gumbn)