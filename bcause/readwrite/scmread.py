import re

import networkx as nx

import bcause.util.domainutils as dutils
from bcause.factors import MultinomialFactor, DeterministicFactor
from bcause.readwrite import bnread

def from_bif(filepath, vtype=None):
    from bcause.models.cmodel import StructuralCausalModel
    return StructuralCausalModel.from_model(bnread.from_bif(filepath, vtype))

def from_xmlbif(filepath, vtype=None):
    from bcause.models.cmodel import StructuralCausalModel
    return StructuralCausalModel.from_model(bnread.from_xmlbif(filepath, vtype))

def from_uai(filepath, reverse_values=True, var_prefix="V", init_var_id=0, init_var_state=0):
    from bcause.models.cmodel import StructuralCausalModel
    return StructuralCausalModel.from_model(bnread.from_uai(filepath, reverse_values, var_prefix, init_var_id, init_var_state, label="CAUSAL"))



def from_uai(filepath, reverse_values=False, var_prefix="V", init_var_id=0, init_var_state=0, label="CAUSAL", cast_multinomial=True):
    from bcause.models import BayesianNetwork

    if reverse_values:
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
                for v in re.findall(r"[-+]?(?:\d*\.*\d+E?[-+]?\d*)", line):
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
        if len(list(dag.predecessors(v))) == 0:
            factors[v] = MultinomialFactor(dutils.var_parents_domain(dom, dag, v), left_vars=v, values=values)
        else:
            factors[v] = DeterministicFactor(dutils.var_parents_domain(dom, dag, v), left_vars=v, values=values)

    from bcause.models.cmodel import StructuralCausalModel

    return StructuralCausalModel(dag, factors, cast_multinomial=cast_multinomial)
