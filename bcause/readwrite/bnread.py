import logging
import os
import re

import networkx as nx
from pgmpy.readwrite import BIFReader, XMLBIFReader

from bcause.conversion.pgmpy import toBCauseBNet
from bcause.factors import MultinomialFactor
from bcause.factors.values.store import DataStore


from bcause.util.assertions import assert_file_exists
from bcause.util import domainutils as dutils




def __read(reader, filepath, vtype):
    assert_file_exists(filepath)
    logging.info(f"Reading model in {reader.__name__.replace('Reader', '')} format from {os.path.abspath(filepath)}")
    model =  toBCauseBNet(reader(filepath).get_model(), vtype)
    logging.debug(f"Loaded {model}")
    return model

def from_bif(filepath, vtype=None):
    vtype = vtype or DataStore.DEFAULT_STORE
    return __read(BIFReader, filepath, vtype)

def from_xmlbif(filepath, vtype=None):
    vtype = vtype or DataStore.DEFAULT_STORE
    return __read(XMLBIFReader, filepath, vtype)


def from_uai(filepath, reverse_values=True, var_prefix="V", init_var_id=0, init_var_state=0, label="BAYES"):
    from bcause.models import BayesianNetwork

    if not reverse_values:
        raise ValueError("Not implemented UAI with reverse_values flag as False")

    # Check the label
    with open(filepath) as f:
        l = f.readline().replace(" ", "").replace("\n", "")
        if l not in label:
            raise ValueError(f"Wrong label in uai file: {l}")
            # Generator with the numbers in the file

    def get_numbers(path):
        with open(path) as f:
            for line in f:
                for v in re.findall(r"[-+]?(?:\d*\.*\d+)", line):
                    yield v

    numbers = get_numbers(filepath)

    # variables and domains
    num_vars = int(next(numbers))
    varnames = [f"{var_prefix}{i + init_var_id}" for i in range(0, num_vars)]
    card = [int(next(numbers)) for v in varnames]
    dom = {v: list(range(init_var_state, init_var_state + c)) for v, c in zip(varnames, card)}

    num_factors = int(next(numbers))
    assert num_factors == num_vars

    # structure
    arcs = []
    for v in varnames:
        num_parents = int(next(numbers)) - 1
        parents = [varnames[int(next(numbers))] for _ in range(0, num_parents)]
        for p in parents:
            arcs.append((p, v))

        assert varnames[int(next(numbers))] == v

    dag = nx.DiGraph(arcs)

    # factors in the model
    factors = dict()
    for v in varnames:
        num_values = int(next(numbers))
        values = [float(next(numbers)) for _ in range(num_values)]
        factors[v] = MultinomialFactor(dutils.var_parents_domain(dom, dag, v), left_vars=v, values=values)

    return BayesianNetwork(dag, factors)
