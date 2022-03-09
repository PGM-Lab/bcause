from functools import reduce
from typing import Dict
from itertools import product

import numpy as np


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
