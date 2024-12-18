import pytest
from numpy.testing import assert_array_almost_equal, assert_almost_equal

import bcause.readwrite.bnread as bnread
from bcause import BayesianNetwork
from bcause.inference.probabilistic.datainference import LaplaceInference
from bcause.inference.probabilistic.elimination import VariableElimination
from bcause.inference.probabilistic.infpgmpy import VariableEliminationPGMPY
from bcause.models.transform.simplification import minimalize

model = bnread.from_bif("models/asia.bif")


def test_variable_elimination():
    args = [dict(target="dysp", evidence=None),
            dict(target="dysp", evidence=dict(smoke="yes")),
            dict(target="smoke", evidence=dict(dysp="yes")),
            dict(target="either", evidence=None),
            dict(target="smoke", conditioning="dysp"),
            dict(target=["smoke"], conditioning="dysp", evidence=dict(asia="yes")),
            ]

    expected = [0.43597059999999993, 0.552808, 0.6339968796061018, 0.06482800000000001, 0.633997, 0.62592 ]

    inf = VariableElimination(model)
    actual = [inf.query(**arg).values[0] for arg in args]
    assert_array_almost_equal(actual, expected)


def test_laplace_inference():
    args = [dict(target="dysp", evidence=None),
            dict(target="dysp", evidence=dict(smoke="yes")),
            dict(target="smoke", evidence=dict(dysp="yes")),
            dict(target="either", evidence=None),
            dict(target="smoke", conditioning="dysp"),
            dict(target=["smoke"], conditioning="dysp", evidence=dict(asia="yes")),
            ]

    from bcause import randomUtil
    randomUtil.seed(1)
    data = model.sample(5000, as_pandas=True)

    inf = LaplaceInference(data, model.domains)
    actual = [inf.query(**arg).values[0] for arg in args]
    expected = [0.4212, 0.537625754527163, 0.6343779677113011, 0.0616, 0.634377967711301, 0.619047619047619]
    assert_array_almost_equal(actual, expected)


#list(map(lambda x: tuple(x[0]+[x[1]]), list(zip([list(d.values()) for d in dt], ex))))




#list(zip([list(d.values()) for d in dt], ex))

@pytest.mark.parametrize("target,evidence,expected",
                         [('dysp', None, {'xray'}),
                          ('dysp', {'smoke': 'yes'}, {'xray','smoke'}),
                          ('smoke', {'dysp': 'yes'}, {'xray'}),
                          ('either', None, {'bronc', 'dysp', 'xray'}),
                          ('xray', {'tub': 'yes'}, {'dysp', 'asia', 'tub', 'bronc'}),
                          ('lung', {'asia': 'yes'}, {'bronc', 'dysp', 'either', 'tub', 'xray', 'asia'}),
                          ('lung', {'asia': 'yes', 'either': 'yes'}, {'dysp', 'asia', 'xray', 'bronc'}),
                          ('smoke',
                           {'asia': 'yes'},
                           {'bronc', 'dysp', 'either', 'lung', 'tub', 'xray', 'asia'})]
                         )
def test_minimalize(target, evidence, expected):

    def determine_dropped(target, evidence):
        return set(model.variables).difference(set(minimalize(model, target, evidence).variables))

    assert determine_dropped(target, evidence) == expected


def test_multi_evidence():

    new_factors = {v: f.copy_with_dummy_state(v, "?") if model.is_leaf(v) else f for v, f in model.factors.items()}
    model2 = BayesianNetwork(model.graph, new_factors)

    inf1 = VariableElimination(model)
    inf2 = VariableElimination(model2)

    for x in set(model.variables).difference(model.leaf_nodes):
        for y in model.leaf_nodes:
            p0 = inf1.query(x, evidence={y: "yes"}).values[0]
            p1 = inf2.query(x, evidence={y: "yes"}).values[0]
            p2 = inf2.query(x, evidence={y: ["yes", "?"]}).values[0]
            p3 = inf1.query(x, evidence={y: ["yes", "no"]}).values[0]
            p4 = inf1.query(x).values[0]

            assert_almost_equal(p0,p1)
            assert_almost_equal(p1,p2)
            assert_almost_equal(p3,p4)



def test_VariableEliminationPGMPY():
    args = [dict(target="dysp", evidence=None),
            dict(target="dysp", evidence=dict(smoke="yes")),
            dict(target="smoke", evidence=dict(dysp="yes")),
            dict(target="either", evidence=None),
            dict(target="smoke", conditioning="dysp"),
            dict(target=["smoke"], conditioning="dysp", evidence=dict(asia="yes")),
            ]

    expected = [
        0.43597060000000004,
        0.552808,
        0.6339968796061018,
        0.06482799999999998,
        0.6339968796061018,
        0.6259198578212214]


    inf = VariableEliminationPGMPY(model)
    actual = [inf.query(**arg).values[0] for arg in args]

    assert_array_almost_equal(actual, expected)

