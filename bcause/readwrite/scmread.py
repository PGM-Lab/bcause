
from bcause.models.cmodel import StructuralCausalModel
from bcause.readwrite import bnread


def from_bif(file, vtype=None):
    return StructuralCausalModel.from_model(bnread.from_bif(file, vtype))

def from_xmlbif(file, vtype=None):
    return StructuralCausalModel.from_model(bnread.from_xmlbif(file, vtype))

def from_uai(path, reverse_values=True, var_prefix="V", init_var_id=0, init_var_state=0):
    return StructuralCausalModel.from_model(bnread.from_uai(path, reverse_values, var_prefix, init_var_id, init_var_state, label="CAUSAL"))


