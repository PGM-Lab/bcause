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

def _get_node(var,values):
    if any(isinstance(x,str) for x in values):
        node = gum.LabelizedVariable(var, var, len(values))
        for index, label in enumerate(values):
            node.changeLabel(index, label)
    if all(isinstance(x,int) for x in values):
        node = gum.IntegerVariable(var, var, values)
    return node

def _addGraph(bn: gum.BayesNet, domain: dict,  edges: list[tuple[str, str]]):
    # add Nodes
    list(map(lambda node: bn.add(_get_node(node, domain[node])), domain.keys()))
    # add Edges
    list(starmap(bn.addArc, edges))

# Add CPTs
def _setCPT(f: bfd.MultinomialFactor, bn: gum.BayesNet):
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

    _addGraph(gumbn, bn.domains,list(bn.graph.edges))
    [_setCPT(f,gumbn) for f in bn.factors.values()]
    return gumbn

def _value_from_var_domain(var) -> list:
    if var.varType() == 3:
        return list(range(var.minVal(), var.maxVal() + 1))
    elif var.varType() in [0,2]:
        return list(var.integerDomain())
    elif var.varType() == 1:
        return list(var.labels())

def potential_to_factor(potential: gum.Potential) -> bfd.MultinomialFactor:
    vars = potential.variablesSequence()
    left_vars = [vars[0].name()]
    right_vars = [var.name() for var in vars[1:]]
    domain = {var.name(): _value_from_var_domain(var) for var in vars}
    values = potential.toarray().flatten()
    return bfd.MultinomialFactor(domain, values, left_vars=left_vars, right_vars=right_vars)

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
    domains = {bn.variable(i).name(): _value_from_var_domain(bn.variable(i)) for i in range(bn.size())}
    # Base on the cpt tables, create the factors to then use it in the MultinomialFactor function
    factors = {}
    for i in range(bn.size()):
        node = bn.variable(i).name()
        dom = dutils.subdomain(domains, *gutils.relevat_vars(dag, node))
        value = bn.cpt(node).toarray().flatten()
        parents = [bn.variable(j).name() for j in bn.parents(i)]
        factors[node] = bfd.MultinomialFactor(dom, value, left_vars=[node], right_vars=parents)
    # Return the BayesianNetwork with the dag and the factors
    return BayesianNetwork(dag, factors)


if __name__=="__main__":

    dag = nx.DiGraph([("V1", "V2"), ("V2", "V3"),("V3", "V4"),("U1", "V1"),("U2", "V2"),("U2", "V4"),("U3", "V3")])
    model = StructuralCausalModel(dag)
    domains = dict(V1=[0,1],V2=[0,1],V3=[0,1],V4=[0,1], U1=[0,1],U2=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],U3=[0,1,2,3])
    randomUtil.seed(1)
    model.fill_random_factors(domains)
    data = model.sample(1000, as_pandas=True)

    bn = BayesianNetwork(dag,model.factor_list)

    gumbn = toAgrum(bn)
    newbn = fromAgrum(gumbn)