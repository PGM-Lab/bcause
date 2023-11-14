import tempfile
from pathlib import Path

import networkx as nx
import pytest
from networkx import topological_sort
from numpy.testing import assert_array_almost_equal

from bcause.models.cmodel import StructuralCausalModel

# Define a DAG and the domains
dag = nx.DiGraph([("V1", "V2"), ("V2", "V3"),("V3", "V4"),("U1", "V1"),("U2", "V2"),("U2", "V4"),("U3", "V3")])
model = StructuralCausalModel(dag)
domains = dict(V1=[0,1],V2=[0,1],V3=[False, True], V4=["a","b"], U1=[0,1,2],U2=[0,1,2,3],U3=[0,1,2,3])
from bcause.util.randomUtil import seed
seed(1)
model.fill_random_factors(domains)
data = model.sample(1000, as_pandas=True)




@pytest.mark.parametrize("ftype", [".uai", ".bif"])#, ".xmlbif"])
def test_read_write_scm(ftype):

    with tempfile.TemporaryDirectory() as tmpdirname:
        model2 = StructuralCausalModel.read(model.save(Path(tmpdirname, f"model{ftype}")))


    if ftype == ".uai":
        model2 = model2.rename_vars(dict(V0="V1", V1="V2", V2="V3", V3="V4", V4="U1", V5="U2", V6="U3"))
    # qm1 = model.get_qbnet()
    # qm2 = model2.get_qbnet()
    # v1 = list(topological_sort(qm1.graph))[-1]
    # v2 = list(topological_sort(qm2.graph))[-1]
    # assert_array_almost_equal(qm1.factors[v1].values,  qm2.factors[v2].values)

    for v in model.exogenous:
        assert_array_almost_equal(model.factors[v].values, model2.factors[v].values)


    for v in model.endogenous:
        print(v)
        f1 = model.factors[v]
        f2 = model2.factors[v]
        var_order = f1.variables
        var_order.sort()

        v1 = f1.values_array(var_order).flatten()
        v2 = f2.values_array(var_order).flatten()
        assert_array_almost_equal(v1,v2)