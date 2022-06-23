from numpy.testing import assert_array_almost_equal

import bcause.readwrite.bnread as bnread
from bcause.inference.elimination.variableelimination import VariableElimination
from bcause.models.transform.simplification import minimalize

model = bnread.fromBIF("models/asia.bif")


def test_variable_elimination():
    args = [dict(target="dysp", evidence=None),
            dict(target="dysp", evidence=dict(smoke="yes")),
            dict(target="smoke", evidence=dict(dysp="yes")),
            dict(target="either", evidence=None)]

    expected = [0.43597059999999993, 0.552808, 0.6339968796061018, 0.06482800000000001]

    ve = VariableElimination(model)
    actual = [ve.query(**arg).values[0] for arg in args]
    assert_array_almost_equal(actual, expected)

def test_minimalize():

    print(model.variables)
    print(model.domains)


    args = [dict(target="dysp", evidence=None),
            dict(target="dysp", evidence=dict(smoke="yes")),
            dict(target="smoke", evidence=dict(dysp="yes")),
            dict(target="either", evidence=None),
            dict(target="xray", evidence=dict(tub="yes")),
            dict(target="lung", evidence=dict(asia="yes")),
            dict(target="lung", evidence=dict(asia="yes", either="yes")),
            dict(target="smoke", evidence=dict(asia="yes"))
            ]

    expected = [{'xray'}, {'xray'}, {'xray'}, {'bronc', 'dysp', 'xray'}, {'bronc', 'asia', 'dysp'},
                {'either', 'bronc', 'dysp', 'tub', 'xray'}, {'bronc', 'dysp', 'xray'},
                {'either', 'bronc', 'dysp', 'tub', 'lung', 'xray'}]

    def determine_dropped(target, evidence):
        return set(model.variables).difference(set(minimalize(model, target, evidence).variables))

    for i in range(len(args)):
        assert determine_dropped(**args[i]) == expected[i]


