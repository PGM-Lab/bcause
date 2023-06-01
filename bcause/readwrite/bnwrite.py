import logging
import os

from pgmpy.readwrite import BIFWriter, XMLBIFWriter

from bcause.conversion.pgmpy import toPgmpyBNet
from bcause.models import BayesianNetwork
from bcause.util.assertions import assert_file_exists


def __write(writer, model, path):
    folder = os.path.dirname(path)
    assert_file_exists(folder)
    format = writer.__name__.replace('Writer', '')
    logging.info(f"Saving model in {format} to {os.path.abspath(path)}")
    getattr(writer(toPgmpyBNet(model)), f"write_{format.lower()}")(path)

def toBIF(model:BayesianNetwork, path):
    __write(BIFWriter, model, path)

def toXMLBIF(model:BayesianNetwork, path):
    __write(XMLBIFWriter, model, path)


def toUAI(model:BayesianNetwork, path, reverse_values=True):
    out = "BAYES\n"

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
        if not reverse_values:
            var_order = var_order[::-1]
        values = f.to_values_array(var_order).flatten()
        out += f"{len(values)}\t" + " ".join([str(p) for p in values]) + "\n"

    # Write to file
    folder = os.path.dirname(path)
    assert_file_exists(folder)
    with open(path, "w+") as file:
        # Writing data to a file
        file.write(out)
