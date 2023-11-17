'''
Tests for the gradient descent optimizer

TODO: add the values for the commented parts
'''

import networkx as nx
import pytest

import bcause as bc
from bcause.factors.imprecise import IntervalProbFactor
from bcause.inference.causal.multi import GDCC
from bcause.inference.causal.elimination import CausalVariableElimination
from bcause.models.cmodel import StructuralCausalModel

from numpy.testing import assert_array_almost_equal

models, datasets, infobjects, causes, effects = dict(), dict(), dict(), dict(), dict()

@pytest.fixture(scope="session", autouse=True)  # For running this sequentialy and avoiding problems with random generator
def learn():
    for l in ["modelTestA_1", "modelTestB_1"]:
        models[l] = StructuralCausalModel.read(f"./models/{l}.uai")
        bc.randomUtil.seed(1)
        datasets[l] = models[l].sampleEndogenous(1000)
        infobjects[l] = GDCC(models[l], datasets[l], num_runs=20, tol=1e-3)    
        causes[l], effects[l] = [models[l].endogenous[i] for i in [0, -1]]


@pytest.mark.parametrize("label,expected",
                             [("modelTestA_1", [0.6040966525755556, 0.3000516555830903, 0.6999483444169097, 0.3959033474244445]), 
                              ("modelTestB_1", [0.3794668912287373, 0.5033763730051801, 0.4966236269948198, 0.6205331087712628]) 
                              ]
                         )
def test_causal_query(label,expected):
    inf = infobjects[label]
    X,Y = causes[label], effects[label]
    p = inf.causal_query(Y, do={X:0})
    actual = p.values
    print(actual)

    assert_array_almost_equal(actual, expected)


#@pytest.mark.parametrize("label,expected",
#                             [("modelTestA_1", [0.5454061540437741, 0.22976323371809068, 0.7702367662819094, 0.45459384595622593]), # todo: cambiar por valores parecidos
#                              ("modelTestB_1", [0.5299653323117544, 0.47003466763566476, 0.5299653323643353, 0.47003466768824564]) # todo: cambiar por valores parecidos
#                              ]
#                         )
#def test_counterfactual_query(label,expected):
#    inf = infobjects[label]
#    X,Y = causes[label], effects[label]
#    p = inf.counterfactual_query(target=Y, do={X:0}, evidence={X:1})
#    actual = p.values
#    print(actual)

#    assert_array_almost_equal(actual, expected)





#@pytest.mark.parametrize("label,expected",
#                             [("modelTestA_1", [0.3177723822046544, 0.042492679529959027, 0.27250968537176334, 0.00442204181161961, 0.4991739879978861, 0.2238942846877377, 0.45391129129598834, 0.18582364710039156]),
#                              ("modelTestB_1", [0.3187933044410842, 0.0025505259961731406, 0.25705787797551244, 0.009037333607844486, 0.5250737834472572, 0.20883100495041307, 0.4633383570006582, 0.2153178125810572]
#)
#                              ]
#                         )
#def test_exo_interval(label,expected):
#    inf = infobjects[label]
#    model = models[label]
#    U = model.exogenous[-1]
#    p = IntervalProbFactor.from_precise([m.factors[U] for m in inf.models])
#    actual = p.values
#    print(actual)
#    assert_array_almost_equal(actual, expected)

#@pytest.mark.parametrize("label,expected",
#                             [("modelTestA_1", -1489.024653062976),("modelTestB_1", -1410.1130382420752)
#                              ]
#                         )
#def test_llk(label, expected):
#    inf = infobjects[label]
#    data = datasets[label]
#    actual = min([m.log_likelihood(data) for m in inf.models])
#    print(actual)
#    assert actual == expected
