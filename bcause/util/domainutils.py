from typing import Dict
from itertools import product

def state_space(domain:Dict):
    return list(product(*list(domain.values())))

def assingment_space(domain:Dict):
    return [dict(zip(domain.keys(), s)) for s in state_space(domain)]


