from bcause.factors import MultinomialFactor

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

f.store.builder(domain=domain, data=values)


# P(B|A) as a binary tree
f = MultinomialFactor(domain, values, left_vars=["B"], vtype="btree")

f.store
type(f.store)

print(f.store.data.summary())






