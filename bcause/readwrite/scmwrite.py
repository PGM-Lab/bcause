
import bcause.readwrite.bnwrite as bnwrite

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bcause.models.cmodel import StructuralCausalModel


def to_bif(model:'StructuralCausalModel', filepath):
    bnwrite.to_bif(model.to_bnet(), filepath)

def to_xmlbif(model:'StructuralCausalModel', filepath):
    bnwrite.to_xmlbif(model.to_bnet(), filepath)

def to_uai(model:'StructuralCausalModel', filepath, reverse_values=False, var_order=None):
    bnwrite.to_uai(model, filepath, reverse_values, "CAUSAL", integer_varlist=model.endogenous, var_order=var_order);