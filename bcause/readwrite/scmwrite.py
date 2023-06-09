
import bcause.readwrite.bnwrite as bnwrite

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bcause.models.cmodel import StructuralCausalModel


def to_bif(model:'StructuralCausalModel', path):
    bnwrite.to_bif(model.to_bnet(), path)

def to_xmlbif(model:'StructuralCausalModel', path):
    bnwrite.to_xmlbif(model.to_bnet(), path)

def to_uai(model:'StructuralCausalModel', path, reverse_values=True):
    bnwrite.to_uai(model, path, reverse_values, "CAUSAL", integer_varlist=model.endogenous);