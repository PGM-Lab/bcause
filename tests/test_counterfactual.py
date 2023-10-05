import networkx as nx
import pytest

import bcause as bc
from bcause.inference.causal.elimination import CausalVariableElimination
from bcause.models.cmodel import StructuralCausalModel

from numpy.testing import assert_array_almost_equal


dag = nx.DiGraph([("V0", "V1"), ("V1", "V2"),("V2", "V3"),("U0", "V0"),("U1", "V1"),("U1", "V3"),("U2", "V2")])
model = StructuralCausalModel(dag)
domains = dict(V0=[0,1],V1=[0,1],V2=[0,1],V3=[0,1], U0=[0,1],U1=[0,1,2,3,4,5,6,7,8,9,10,11],U2=[0,1,2,3,4,5,6,7])
bc.randomUtil.seed(1)
model.fill_random_factors(domains)


cve = CausalVariableElimination(model, preprocess_flag=True)


@pytest.mark.parametrize("cause,effect,expected",
                             [  ('V0', 'V2', 0.0),
                                ('V0', 'V3', 0.0),
                                ('V1', 'V2', 0.3898750354752074),
                                ('V1', 'V3', 0.2693966041543566),
                                ('V2', 'V3', 1.0)
                                ]
                         )
def test_prob_necessity(cause, effect, expected):
    actual = cve.prob_necessity(cause, effect, true_false_cause=(0, 1), true_false_effect=(0, 1))
    assert_array_almost_equal(actual, expected)


@pytest.mark.parametrize("cause,effect,expected",
                            [('V0', 'V1', 0.9827816928270122),
                            ('V0', 'V2', 0.09039648468621407),
                            ('V0', 'V3', 0.14400104268505015),
                            ('V1', 'V2', 0.3958968275699974),
                            ('V1', 'V3', 0.23055531145720365),
                            ('V2', 'V3', 0.6413227964557568), ]
                         )
def test_prob_sufficiency(cause, effect, expected):
    actual = cve.prob_sufficiency(cause, effect, true_false_cause=(0, 1), true_false_effect=(0, 1))
    print(f"('{cause}', '{effect}', {actual}), ")
    assert_array_almost_equal(actual, expected)


@pytest.mark.parametrize("cause,effect,expected",
                            [('V0', 'V1', 0.1519096390351899),
                                ('V0', 'V2', 0.03713413950655857),
                                ('V0', 'V3', 0.09137529490602783),
                                ('V1', 'V2', 0.24444886935684465),
                                ('V1', 'V3', 0.1129960393335237),
                                ('V2', 'V3', 0.4622481569696807), ]
                         )
def test_prob_necessity_sufficiency(cause, effect, expected):
    actual = cve.prob_necessity_sufficiency(cause, effect, true_false_cause=(0, 1), true_false_effect=(0, 1))
    print(f"('{cause}', '{effect}', {actual}), ")
    assert_array_almost_equal(actual, expected)