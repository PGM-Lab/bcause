from numpy.testing import assert_array_almost_equal

import bcause.readwrite.bnread as bnread
from inference.elimination.variableelimination import VariableElimination

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
