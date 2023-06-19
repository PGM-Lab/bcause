from bcause.factors import MultinomialFactor
from bcause.factors.values.btreestore import BTreeNode, BTreeStore

domain = dict(A=["a1", "a2"], B=["b1", "b2", "b3" ,"b4"])
values = [[0.2, .2, 0.5, 0.1], [0.2, 0.2, 0.6 ,0.0]]
# P(B|A) as a numpy table
f = MultinomialFactor(domain, values, left_vars=["B"])


# Operaciones

f * f
f + f
f.R(A="a1", B=["b1","b2"])
f.marginalize("A")


# data store

f.store
type(f.store)

# each store will have a builder function
f.store.builder(domain=domain, data=values)


# P(B|A) as a binary tree
f = MultinomialFactor(domain, values, left_vars=["B"], vtype="btree")

f.store
type(f.store)

print(f.store.data.summary())





# Use of the BTreeNode class:


variable = "U"
var_domain = ["u1", "u2", "u3"]

# different ways of building a node
n1 = BTreeNode.build(variable, var_domain, 0.2, 0.4, left_states=["u1"])
n2 = BTreeNode.build(variable, var_domain, 0.2, 0.4, right_states=["u2", "u3"])
n3 = BTreeNode.build(variable, var_domain, 0.4, 0.3, end_left_exclusive=1)

# a non terminal node
nested_nodes = BTreeNode.build("X", ["x1", "x2"], 0.33, n1)

print(nested_nodes)

domain = dict(A=["a1", "a2"], B=["b1", "b2", "b3", "b4"])
new_var_order = ["B", "A"]
# complete vars
new_dom = dict([(v, domain[v]) for v in new_var_order])
data = [[0.2, .2, 0.5, 0.1], [0.2, 0.2, 0.6, 0.0]]

bt = BTreeStore(domain, data)
print(bt.data.summary())






