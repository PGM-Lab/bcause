import math
import random

from bcause.factors.values.btreestore import BTreeNode

states = [f"u{i}" for i in range(0,16)]
total_prob=1
min_prob=0

def _build_random_marginal_tree(v, states, total_prob=1, threshold=0, min_prob= 1e-5):

    # number of probability values
    n = len(states)

    # check if it is a leave or it should be pruned
    if n == 1 or total_prob<=threshold:
        return total_prob/n

    # select a random partition of the states
    s = random.choice(range(1,len(states)))
    left_states = states[:s]
    right_states = states[s:]

    # randomly distribute the probability mass
    p =  (random.random() * (1- min_prob)) + min_prob
    total_prob_left = p * total_prob
    total_prob_right = (1-p) * total_prob

    # build both children
    tree_left = _build_random_marginal_tree(v, states=left_states, total_prob=total_prob_left, threshold=threshold, min_prob=min_prob)
    tree_right = _build_random_marginal_tree(v, states=right_states, total_prob=total_prob_right, threshold=threshold, min_prob=min_prob)

    # build the tree
    return BTreeNode.build(v, states, left_child=tree_left, right_child=tree_right, left_states=left_states)


t = _build_random_marginal_tree("U", states)

print(t.summary())

t = _build_random_marginal_tree("U", states, threshold=0.1)

print(t.summary())
