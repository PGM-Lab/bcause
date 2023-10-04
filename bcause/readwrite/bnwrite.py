import logging
import os
from pathlib import Path

from pgmpy.readwrite import BIFWriter, XMLBIFWriter

from bcause.factors import MultinomialFactor
from bcause.util.assertions import assert_file_exists


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bcause.models import BayesianNetwork

def __write(writer, model, filepath):
    from bcause.conversion.pgmpy import toPgmpyBNet
    filepath = Path(filepath)
    folder = filepath.parent
    assert_file_exists(folder)
    format = writer.__name__.replace('Writer', '')
    logging.info(f"Saving model in {format} to {os.path.abspath(filepath)}")
    getattr(writer(toPgmpyBNet(model)), f"write_{format.lower()}")(filepath)

def to_bif(model:'BayesianNetwork', filepath):
    __write(BIFWriter, model, filepath)

def to_xmlbif(model:'BayesianNetwork', filepath):
    __write(XMLBIFWriter, model, filepath)

def to_uai(model:'BayesianNetwork', filepath, reverse_values=False, label="BAYES", integer_varlist = None):

    integer_varlist = integer_varlist or []

    print(reverse_values)

    out = f"{label}\n"
    out += f"{len(model.variables)}\n"
    out += " ".join([str(i) for i in model.varsizes.values()])
    out += "\n"

    var_idx = {v: model.variables.index(v) for v in model.variables}
    out += f"{len(model.factors)}\n"

    for v in model.variables:
        idx = [var_idx[x] for x in model.get_parents(v) + [v]]
        out += f"{len(idx)}\t" + " ".join([str(p) for p in idx]) + "\n"

    out += "\n"

    for v in model.variables:
        f = model.factors[v]
        var_order = f.variables

        if label=="CAUSAL" and v in model.endogenous and isinstance(f, MultinomialFactor):
            f = f.to_deterministic()
            var_order = [x for x in var_order if x != v]

        if not reverse_values:
            var_order = var_order[::-1]
        values = f.values_array(var_order).flatten()
        if v in integer_varlist:
            values = [int(p) for p in values]
        out += f"{len(values)}\t" + " ".join([str(p) for p in values]) + "\n"

    # Write to file
    filepath = Path(filepath)
    folder = filepath.parent
    assert_file_exists(folder)

    print(out)
    with open(filepath, "w+") as file:
        # Writing data to a file
        file.write(out)

