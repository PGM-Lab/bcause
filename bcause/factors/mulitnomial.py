from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from factors.factor import DiscreteFactor, ConditionalFactor
from factors.store import store_dict
from util.domainutils import assingment_space, state_space
from util.arrayutils import normalize_array


class MultinomialFactor(DiscreteFactor, ConditionalFactor):

    def __init__(self, domain:Dict, data, right_vars:list=None, vtype="numpy"):

        np.ndim(data)

        self._store = store_dict[vtype](data, domain)
        self._right_vars = right_vars or []
        self._variables = list(domain.keys())

        def builder(*args, **kwargs):
            return MultinomialFactor(*args, **kwargs, vtype=vtype)

        self.builder = builder

    # Properties

    @property
    def values_list(self)->List:
        return self.prob(assingment_space(self.domain))

    # Factor operations
    def restrict(self, **observation) -> MultinomialFactor:
        if len(set(observation.keys()).intersection(self._variables))==0: return self
        new_store = self.store.restrict(**observation)
        new_right_vars = [v for v in new_store.variables if v in self.right_vars]
        return self.builder(new_store.domain, new_store.data, new_right_vars)

    def multiply(self, other):
        new_store = self.store.multiply(other.store)
        new_right_vars = [v for v in new_store.variables
                          if v not in self.left_vars and v not in other.left_vars]
        return self.builder(new_store.domain, new_store.data, new_right_vars)

    def addition(self, other):
        new_store = self.store.addition(other.store)
        new_right_vars = [v for v in new_store.variables
                          if v not in self.left_vars and v not in other.left_vars]
        return self.builder(new_store.domain, new_store.data, new_right_vars)

    def subtract(self, other):
        new_store = self.store.subtract(other.store)
        new_right_vars = [v for v in new_store.variables
                          if v not in self.left_vars and v not in other.left_vars]
        return self.builder(new_store.domain, new_store.data, new_right_vars)

    def divide(self, other):
        new_store = self.store.divide(other.store)
        new_right_vars = [v for v in new_store.variables
                          if v not in self.left_vars and v not in other.left_vars]
        return self.builder(new_store.domain, new_store.data, new_right_vars)

    def marginalize(self, *vars_remove) -> MultinomialFactor:
        if len(set(vars_remove).intersection(self._variables))==0: return self
        new_store = self.store.marginalize(*vars_remove)
        new_right_vars = [v for v in new_store.variables if v in self.right_vars]
        return self.builder(new_store.domain, new_store.data, new_right_vars)


    def maxmarginalize(self, *vars_remove) -> MultinomialFactor:
        if len(set(vars_remove).intersection(self._variables))==0: return self
        new_store = self.store.maxmarginalize(*vars_remove)
        new_right_vars = [v for v in new_store.variables if v in self.right_vars]
        return self.builder(new_store.domain, new_store.data, new_right_vars)


    def get_value(self, **observation) -> float:
        return self.store.get_value(**observation)

    def prob(self, observations:List[Dict]) -> List:
        return [self.get_value(**x) for x in observations]

    def log_prob(self, observations:List[Dict]) -> List:
        return [math.log(self.get_value(**x)) for x in observations]

    def sample(self, size:int = 1, varnames:bool=True) -> List:
        if size < 1:
            raise ValueError("sample size cannot be lower than 1.")
        return [self._sample(varnames=varnames) for _ in range(0,size)]

    def _sample(self, varnames:bool=True) -> tuple:
        # joint or marginal distribution
        if len(self.right_vars) == 0:
            possible_states = state_space(self.left_domain)
            observations = assingment_space(self.left_domain)
            probs = np.array([float(self.store.restrict(**obs).data) for obs in observations])
            idx = np.random.choice(list(range(0, len(possible_states))), p=probs / probs.sum())
            sample = observations[idx] if varnames else possible_states[idx]
            return sample
        ## conditional distribution
        else:
            raise NotImplementedError("Sampling not available for conditional distributions")
            #todo: return [self.restrict(**obs).sample() for obs in assingment_space(self.right_domain)]


    def __mul__(self, other):
        return self.multiply(other)

    def __rmul__(self, other):
        return other.multiply(self)

    def __add__(self, other):
        return self.addition(other)

    def __radd__(self, other):
        return other.addition(self)

    def __sub__(self, other):
        return self.subtract(other)

    def __rsub__(self, other):
        return other.subtract(self)

    def __truediv__(self, other):
        return self.divide(other)

    def __truediv__(self, other):
        return other.divide(self)


    def __xor__(self, vars_remove):
        if isinstance(vars_remove, str):
            return self.maxmarginalize(vars_remove)
        return self.maxmarginalize(*vars_remove)

    def __pow__(self, vars_remove):
        if isinstance(vars_remove, str):
            return self.marginalize(vars_remove)
        return self.marginalize(*vars_remove)



    def __repr__(self):
        cardinality_dict = self.store.cardinality_dict
        card_str = ",".join([f"{v}:{cardinality_dict[v]}" for v in self._variables])
        vars_str = ",".join(self.left_vars)
        if len(self.right_vars)>0:
            vars_str += "|"+",".join(self.right_vars)
        return f"<{self.__class__.__name__} P({vars_str}), cardinality = ({card_str}), " \
               f"values=[{self.store.values_str()}]>"


def random_multinomial(domain:Dict, right_vars:list=None, vtype="numpy"):
    right_vars = right_vars or []
    left_dims = [i for i,v in enumerate(domain.keys()) if v not in right_vars]
    data = normalize_array(np.random.uniform(0,1, size=[len(d) for d in domain.values()]), axis=left_dims)
    return MultinomialFactor(domain=domain, data=data, right_vars=right_vars, vtype=vtype)


def uniform_multinomial(domain:Dict, right_vars:list=None, vtype="numpy"):
    right_vars = right_vars or []
    left_dims = [i for i,v in enumerate(domain.keys()) if v not in right_vars]
    data = normalize_array(np.ones([len(d) for d in domain.values()]), axis=left_dims)
    return MultinomialFactor(domain=domain, data=data, right_vars=right_vars, vtype=vtype)


if __name__ == "__main__":

    # P(B|A)
    left_domain = dict(A=["a1","a2"])
    right_domain = dict(B=[0,1, 3])
    domain = {**left_domain, **right_domain}

    data=np.array([[0.5, .4, 1.0], [0.3, 0.6, 0.1]])
    f = MultinomialFactor(domain, data, right_vars=["A"])

    f2 = MultinomialFactor(left_domain, data = [0.1, 0.9])
    f - f2

