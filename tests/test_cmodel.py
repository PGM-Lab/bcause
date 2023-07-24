
import networkx as nx
from networkx.utils import graphs_equal

import bcause as bc
from bcause.inference.elimination.variableelimination import CausalVariableElimination
from bcause.models.cmodel import StructuralCausalModel

from numpy.testing import assert_array_almost_equal

# Define a DAG and the domains
dag = nx.DiGraph([("V1", "V2"), ("V2", "V3"),("V3", "V4"),("U1", "V1"),("U2", "V2"),("U2", "V4"),("U3", "V3")])
model = StructuralCausalModel(dag)
domains = dict(V1=[0,1],V2=[0,1],V3=[0,1],V4=[0,1], U1=[0,1,3],U2=[0,1,2,3],U3=[0,1,2,3])
bc.randomUtil.seed(1)
model.fill_random_factors(domains)
data = model.sample(1000, as_pandas=True)



def test_model():
    f1, p1 = model.get_factors("V1","U1")
    assert_array_almost_equal(((p1*f1)**"U1").values, [0.5446886440624504, 0.45531135593754957])
    assert model.endogenous == ['V1', 'V2', 'V3', 'V4']
    assert model.exogenous ==  ['U1', 'U2', 'U3']


def test_ccomponents():
    assert model.ccomponents == [{'U1', 'V1'}, {'U2', 'V2', 'V4'}, {'U3', 'V3'}]
    assert model.endo_ccomponents == [{'V1'}, {'V2', 'V4'}, {'V3'}]
    assert graphs_equal(model.qgraph, nx.DiGraph([('V1', 'V2'), ('V1', 'V4'), ('V2', 'V3'), ('V2', 'V4'), ('V3', 'V4')]))

def test_qfact_model():
    qfact = model.get_qfactorisation()
    assert_array_almost_equal(qfact["V1"].values, [0.5446886440624504, 0.45531135593754957])
    assert_array_almost_equal(qfact["V2"].values, [0.17548216319464768,0.8245178368053524,0.6592091722110839,0.34079082778891606])
    assert_array_almost_equal(qfact["V3"].values,
                              [0.3769378426382723, 0.6230621573617277, 0.956425026944555, 0.04357497305544491])
    assert_array_almost_equal(qfact["V4"].values[3:7],
                              [0.0, 0.5628335477556848, 0.4371664522443152, 0.7995086859069446])


def test_qfact_data():
    qfact = model.get_qfactorisation(data)
    assert_array_almost_equal(qfact["V1"].values, [0.53, 0.47])
    assert_array_almost_equal(qfact["V2"].values,
                              [0.1320754716981132, 0.8679245283018868, 0.6425531914893617, 0.3574468085106383])
    assert_array_almost_equal(qfact["V3"].values,
                              [0.33870967741935487, 0.6612903225806451, 0.9426751592356687, 0.057324840764331204])
    assert_array_almost_equal(qfact["V4"].values[3:7], [0.0, 0.5227272727272727, 0.47727272727272724, 0.8])


def test_causal_queries():
    # Run causal inference with Variable Elimination
    cve = CausalVariableElimination(model)
    p = cve.causal_query("V4", do=dict(V2=0))

    assert p.values == [0.7611346018757232,0.2388653981242768]

    # Run a counterfactual query
    p = cve.counterfactual_query("V4",do=dict(V1=0), evidence=dict(V2=1))
    assert p.values == [0.6773272316833546,0.32267276831664543]