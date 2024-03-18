import logging
import os
import tempfile
from pathlib import Path

from pgmpy.readwrite import BIFWriter, XMLBIFWriter
from pyAgrum import pyAgrum

from bcause.factors import MultinomialFactor, DeterministicFactor
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
    logging.getLogger( __name__ ).info(f"Saving model in {format} to {os.path.abspath(filepath)}")
    getattr(writer(toPgmpyBNet(model)), f"write_{format.lower()}")(filepath)

def to_bif(model:'BayesianNetwork', filepath):
    __write(BIFWriter, model, filepath)

def to_xmlbif(model:'BayesianNetwork', filepath):
    __write(XMLBIFWriter, model, filepath)

def to_uai(model:'BayesianNetwork', filepath, reverse_values=False, label="BAYES", integer_varlist = None, var_order=None):

    integer_varlist = integer_varlist or []


    variables = model.endogenous + model.exogenous if label == "CAUSAL" else model.variables
    var_order = var_order or variables
    if set(variables) != set(var_order):
        raise ValueError("Wrong variable order")
    variables = var_order

    out = f"{label}\n"
    out += f"{len(variables)}\n"
    out += " ".join([f"{model.varsizes[v]}" for v in variables])
    out += "\n"

    var_idx = {v: variables.index(v) for v in variables}
    out += f"{len(model.factors)}\n"

    var_order = dict()

    for v in variables:
        var_order[v] = [x for x in model.get_parents(v)][::-1] + [v]
        idx = [var_idx[x] for x in var_order[v]]
        out += f"{len(idx)}\t" + " ".join([str(p) for p in idx]) + "\n"

    out += "\n"

    for v in variables:
        f = model.factors[v]
        vorder = var_order[v]

        if label=="CAUSAL" and v in model.endogenous:
            if not isinstance(f, DeterministicFactor):
                f = f.to_deterministic()
            vorder = [x for x in vorder if x != v]

        # if not reverse_values:
        #     var_order = var_order[::-1]
        values = f.values_array(vorder).flatten()
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



def to_hugin(model:'BayesianNetwork', filepath):

    with tempfile.NamedTemporaryFile(suffix='.bifxml') as tmp:
        model.save(tmp.name)
        # Save states as strings
        lines = [l.replace('<OUTCOME>', '<OUTCOME>"').replace('</OUTCOME>', '"</OUTCOME>') for l in
                 open(tmp.name).readlines()]
        with open(tmp.name, mode='wb+') as temp_fp:
            temp_fp.writelines([str.encode(l) for l in lines])

        # Load it with pyagrum
        pamodel = pyAgrum.loadBN(tmp.name)
        pamodel.setProperty("name", '"bcause_bnet"')
        pamodel.saveNET(str(filepath))

