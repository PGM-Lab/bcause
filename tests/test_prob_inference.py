import pytest
from numpy.testing import assert_array_almost_equal

import bcause.readwrite.bnread as bnread
from bcause.inference.probabilistic.datainference import LaplaceInference
from bcause.inference.probabilistic.elimination import VariableElimination
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
                          ('dysp', {'smoke': 'yes'}, {'xray'}),
                          ('smoke', {'dysp': 'yes'}, {'xray'}),
                          ('either', None, {'bronc', 'dysp', 'xray'}),
                          ('xray', {'tub': 'yes'}, {'asia', 'bronc', 'dysp'}),
                          ('lung', {'asia': 'yes'}, {'asia','bronc', 'dysp', 'either', 'tub', 'xray'}),
                          ('lung', {'asia': 'yes', 'either': 'yes'}, {'bronc', 'dysp', 'xray'}),
                          ('smoke',
                           {'asia': 'yes'},
                           {'asia','bronc', 'dysp', 'either', 'lung', 'tub', 'xray'})]
                         )
def test_minimalize(target, evidence, expected):

    def determine_dropped(target, evidence):
        return set(model.variables).difference(set(minimalize(model, target, evidence).variables))

    assert determine_dropped(target, evidence) == expected




