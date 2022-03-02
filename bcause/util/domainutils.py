from typing import Dict
from itertools import product

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