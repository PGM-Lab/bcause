
from bcause.models.cmodel import StructuralCausalModel
from bcause.readwrite import bnread


def from_bif(filepath, vtype=None):
    return StructuralCausalModel.from_model(bnread.from_bif(filepath, vtype))

def from_xmlbif(filepath, vtype=None):
    return StructuralCausalModel.from_model(bnread.from_xmlbif(filepath, vtype))

def from_uai(filepath, reverse_values=True, var_prefix="V", init_var_id=0, init_var_state=0):
    return StructuralCausalModel.from_model(bnread.from_uai(filepath, reverse_values, var_prefix, init_var_id, init_var_state, label="CAUSAL"))


