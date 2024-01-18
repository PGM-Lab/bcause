from functools import reduce
from typing import Dict
from itertools import product

import numpy as np

from bcause.util.graphutils import relevat_vars


def state_space(domain:Dict):
    return list(product(*list(domain.values())))

def assingment_space(domain:Dict):
    return [dict(zip(domain.keys(), s)) for s in state_space(domain)]


def create_domain(varsizes:dict, first = 1, str_values = True):
    dom = dict()
    v,size = list(varsizes.items())[0]
    for v,size in varsizes.items():
       states = list(range(first,first+size))
       if str_values:
           states = [f"{v.lower()}{s}" for s in states]
       dom[v] = states
    return dom


def steps(domain):
    card = [len(d) for d in domain.values()]
    return [reduce(lambda x,y:x*y, card[i+1:],1) for i in range(0,len(card))]




def steps_dict(domain):
    return dict(zip(domain.keys(), steps(domain)))


def restrict_domain(domain, observation):
    domain_restr = domain.copy()
    for v,s in observation.items():
        domain_restr[v] = [s]
    return domain_restr


def values_to_coord(domain, observation):
    return {v:np.where(np.array(domain[v]) == s)[0][0] for v,s in observation.items() if v in domain}


def to_numeric_domains(domain):
    return {v:list(range(len(d))) for v,d in domain.items()}


def index_iterator(domain, observation = None):
    observation = observation or dict()
    S = steps(domain)
    observation = {v: observation[v] for v in domain if v in observation}
    C = state_space(restrict_domain(to_numeric_domains(domain), values_to_coord(domain, observation)))
    for c in C:
        yield sum([x * y for x, y in zip(S, c)])

def index_list(domain, observation = None):
    return list(index_iterator(domain, observation))



def numeric_state_space(dom):
    return state_space(to_numeric_domains(dom))

def numeric_assignment_space(dom):
    return assingment_space(to_numeric_domains(dom))

def random_assignment(dom, iterate_vars = None):
    if iterate_vars is None:
        return {v:np.random.choice(d) for v,d in dom.items()}

    iterate_dom = {v: dom[v] for v in iterate_vars}
    select_dom = {v: dom[v] for v in dom.keys() if v not in iterate_vars}
    out = []
    for s in assingment_space(iterate_dom):
        d = {**random_assignment(select_dom), **s}
        d = {v:d[v] for v in dom.keys()}
        out.append(d)
    return out


def subdomain(domains, *variables):
    if np.ndim(variables)==0: variables = [variables]
    return {v:domains[v] for v in variables}

def var_parents_domain(domains, dag, var):
    return subdomain(domains, *relevat_vars(dag, var))


def identify_true_false(varname, dom):

    dtypes = [type(d) for d in dom]

    if len(dom)>2: raise ValueError("Cannot identify true/false states: non binary domains")
    if len(set(dtypes))>1: raise ValueError("Cannot identify true/false states: different data types")

    tf = []
    if dtypes[0] in [int, float]:
        tf = [d for d in dom if d==True] + [d for d in dom if d==False]
    elif dtypes[0] == str:
        tf = [d for d in dom if d.lower() == varname.lower()+"1"] + [d for d in dom if d.lower() == varname.lower()+"0"] + [d for d in dom if d.lower() == varname.lower()+"2"]

    if len(tf)!=2: raise ValueError("Cannot identify true/false states: wrong values or data types")

    return tf[0],tf[1]
