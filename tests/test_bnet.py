import networkx as nx
import pytest

from bcause.models import BayesianNetwork
from bcause.factors.mulitnomial import random_multinomial
from bcause.util.arrayutils import powerset

import numpy as np


np.random.seed(0)

dag = nx.DiGraph([("A", "B"), ("C", "B"), ("C", "D")])
domains = dict(A=["a1", "a2"], B=[0, 1, 3], C=["c1", "c2", "c3", "c4"], D=[True, False])
factors = dict()

for v in dag.nodes:
    parents = list(dag.predecessors(v))
    dom = {x: d for x, d in domains.items() if x == v or x in parents}
    factors[v] = random_multinomial(dom, right_vars=parents)

bnet = BayesianNetwork(dag, factors)

def test_nodes():
    actual = bnet.variables
    expected =  ['A', 'B', 'C', 'D']
    assert expected==actual

def test_edges():
    actual = list(bnet.graph.edges)
    expected = [('A', 'B'), ('C', 'B'), ('C', 'D')]
    assert expected==actual

def test_domains():
    actual = bnet.domains
    expected = {'A': ['a1', 'a2'], 'B': [0, 1, 3], 'C': ['c1', 'c2', 'c3', 'c4'], 'D': [True, False]}
    assert expected==actual

    actual = bnet.get_domains([])
    expected = dict()
    assert expected==actual

    actual = bnet.get_domains(["D", "C"])
    expected = {'C': ['c1', 'c2', 'c3', 'c4'], 'D': [True, False]}
    assert expected==actual


def test_varsizes():
    # varsizes
    actual = bnet.varsizes
    expected =  {'A': 2, 'B': 3, 'C': 4, 'D': 2}
    assert expected==actual

    actual = bnet.get_varsizes(["D", "C"])
    expected = {'C': 4, 'D': 2}
    assert expected==actual

def test_parents_children():
    inputs = powerset(bnet.variables)
    expected_ch = [[], ['B'], [], ['B', 'D'], [], ['B'], ['B', 'D'], ['B'], ['B', 'D'], [], ['B', 'D'],
                ['B', 'D'], ['B'], ['B', 'D'], ['B', 'D'], ['B', 'D']]

    expected_pa = [[], [], ['A', 'C'], [], ['C'], ['A', 'C'], [], ['C'], ['A', 'C'], ['A', 'C'], ['C'],
                   ['A', 'C'], ['A', 'C'], ['C'], ['A', 'C'], ['A', 'C']]

    for i in range(0, len(inputs)):
        actual = bnet.get_children(*inputs[i])
        assert actual == expected_ch[i]
        actual = bnet.get_parents(*inputs[i])
        assert actual == expected_pa[i]


def test_markov_blanket():
    inputs = bnet.variables
    expected = [['B', 'C'], ['A', 'C'], ['B', 'A', 'D'], ['C']]
    for i in range(0, len(inputs)):
        actual = bnet.markov_blanket(inputs[i])
        assert set(actual) == set(expected[i])


### Adversarial tests

def test_wrong_dag():
    with pytest.raises(ValueError):
        BayesianNetwork("[A|E][B|A:C][C][D|C][E|B]")

def test_wrong_domains():
    with pytest.raises(ValueError):
        bnet.set_factor("A", random_multinomial(dict(A=["a0", "a1"])))
    with pytest.raises(ValueError):
        bnet.set_factor("A", random_multinomial(dict(A=["a0", "a1", "a2"])))
    with pytest.raises(ValueError):
        bnet.set_factor("A", random_multinomial(dict(A=[0, 1])))

    with pytest.raises(ValueError):
        bnet.set_factor("A", random_multinomial(dict(A=["a1", "a2"], C=["c1", "c2", "c3", "c4"]), right_vars=["C"]))