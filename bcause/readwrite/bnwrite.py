import logging
import os

from pgmpy.readwrite import BIFWriter, XMLBIFWriter
from bcause.util.assertions import assert_file_exists


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bcause.models import BayesianNetwork

def __write(writer, model, path):
    from bcause.conversion.pgmpy import toPgmpyBNet
    folder = os.path.dirname(path)
    assert_file_exists(folder)
    format = writer.__name__.replace('Writer', '')
    logging.info(f"Saving model in {format} to {os.path.abspath(path)}")
    getattr(writer(toPgmpyBNet(model)), f"write_{format.lower()}")(path)

def to_bif(model:'BayesianNetwork', path):
    __write(BIFWriter, model, path)

def to_xmlbif(model:'BayesianNetwork', path):
    __write(XMLBIFWriter, model, path)

def to_uai(model:'BayesianNetwork', path, reverse_values=True, label="BAYES", integer_varlist = None):

    integer_varlist = integer_varlist or []

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
        if not reverse_values:
            var_order = var_order[::-1]
        values = f.values_array(var_order).flatten()
        if v in integer_varlist:
            values = [int(p) for p in values]
        out += f"{len(values)}\t" + " ".join([str(p) for p in values]) + "\n"

    # Write to file
    folder = os.path.dirname(path)
    assert_file_exists(folder)
    with open(path, "w+") as file:
        # Writing data to a file
        file.write(out)


'''

class bnwrite(ABC):
    @staticmethod
    def __write(writer, model, path):
        from bcause.conversion.pgmpy import toPgmpyBNet
        folder = os.path.dirname(path)
        assert_file_exists(folder)
        format = writer.__name__.replace('Writer', '')
        logging.info(f"Saving model in {format} to {os.path.abspath(path)}")
        getattr(writer(toPgmpyBNet(model)), f"write_{format.lower()}")(path)

    @staticmethod
    def to_bif(model:'BayesianNetwork', path):
        bnwrite.__write(BIFWriter, model, path)

    @staticmethod
    def to_xmlbif(model:'BayesianNetwork', path):
        bnwrite.__write(XMLBIFWriter, model, path)

    @staticmethod
    def to_uai(model:'BayesianNetwork', path, reverse_values=True, label="BAYES", integer_varlist = None):

        integer_varlist = integer_varlist or []

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
            if not reverse_values:
                var_order = var_order[::-1]
            values = f.values_array(var_order).flatten()
            if v in integer_varlist:
                values = [int(p) for p in values]
            out += f"{len(values)}\t" + " ".join([str(p) for p in values]) + "\n"

        # Write to file
        folder = os.path.dirname(path)
        assert_file_exists(folder)
        with open(path, "w+") as file:
            # Writing data to a file
            file.write(out)



'''