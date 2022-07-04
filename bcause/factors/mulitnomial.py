from __future__ import annotations

import math
from collections import OrderedDict
from functools import reduce
from typing import Dict, List

import numpy as np

from bcause.factors.values.store import store_dict
import bcause.factors.factor as bf
#from . import DiscreteFactor
from bcause.util.domainutils import assingment_space, state_space, steps, random_assignment, to_numeric_domains
from bcause.util.arrayutils import normalize_array, set_value


class MultinomialFactor(bf.DiscreteFactor, bf.ConditionalFactor):

    def __init__(self, domain:Dict, data, left_vars:list=None, right_vars:list=None, vtype="numpy"):

        self._check_domain(domain)

        if np.ndim(data)==1: data = np.reshape(data, [len(d) for d in domain.values()])

        self._store = store_dict[vtype](data=data, domain=domain)
        self.set_variables(list(domain.keys()), left_vars, right_vars)
        self.vtype = vtype

        def builder(**kwargs):
            if "left_vars" not in kwargs and "right_vars" not in kwargs:
                kwargs["left_vars"] = self.left_vars
            return MultinomialFactor(**kwargs, vtype=vtype)

        self.builder = builder


    def constant(self, left_value):

        new_dom = self.left_domain
        states = new_dom[self.left_vars[0]]

        if len(new_dom) != 1:
            raise ValueError("Only one variable on the left is allowed")

        if left_value not in states:
            raise ValueError("Value not in domain")

        new_data = [0.0] * len(states)
        new_data[states.index(left_value)] = 1.0

        return self.builder(domain=new_dom, data=new_data)


    # Factor operations
    def restrict(self, **observation) -> MultinomialFactor:
        if len(set(observation.keys()).intersection(self._variables))==0: return self
        new_store = self.store.restrict(**observation)
        new_right_vars = [v for v in new_store.variables if v in self.right_vars]
        return self.builder(domain=new_store.domain, data=new_store.data, right_vars = new_right_vars)


    def multiply(self, other):
        new_store = self.store.multiply(other.store)
        new_right_vars = [v for v in new_store.variables
                          if v not in self.left_vars and v not in other.left_vars]
        return self.builder(domain=new_store.domain, data=new_store.data, right_vars = new_right_vars)

    def addition(self, other):
        new_store = self.store.addition(other.store)
        new_right_vars = [v for v in new_store.variables
                          if v not in self.left_vars and v not in other.left_vars]
        return self.builder(domain=new_store.domain, data=new_store.data, right_vars = new_right_vars)

    def subtract(self, other):
        new_store = self.store.subtract(other.store)
        new_right_vars = [v for v in new_store.variables
                          if v not in self.left_vars and v not in other.left_vars]
        return self.builder(domain=new_store.domain, data=new_store.data, right_vars = new_right_vars)

    def divide(self, other):
        new_store = self.store.divide(other.store)
        new_right_vars = [v for v in new_store.variables
                          if v not in self.left_vars and v not in other.left_vars]
        return self.builder(domain=new_store.domain, data=new_store.data, right_vars = new_right_vars)

    def marginalize(self, *vars_remove) -> MultinomialFactor:
        if len(set(vars_remove).intersection(self._variables))==0: return self
        new_store = self.store.marginalize(*vars_remove)
        new_right_vars = [v for v in new_store.variables if v in self.right_vars]
        return self.builder(domain=new_store.domain, data=new_store.data, right_vars = new_right_vars)


    def maxmarginalize(self, *vars_remove) -> MultinomialFactor:
        if len(set(vars_remove).intersection(self._variables))==0: return self
        new_store = self.store.maxmarginalize(*vars_remove)
        new_right_vars = [v for v in new_store.variables if v in self.right_vars]
        return self.builder(domain=new_store.domain, data=new_store.data, right_vars = new_right_vars)



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
            probs = np.array([float(self.store.get_value(**obs)) for obs in observations])
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

    def __rtruediv__(self, other):
        return other.divide(self)


    def __xor__(self, vars_remove):
        if isinstance(vars_remove, str):
            return self.maxmarginalize(vars_remove)
        return self.maxmarginalize(*vars_remove)

    def __pow__(self, vars_remove):
        if isinstance(vars_remove, str):
            return self.marginalize(vars_remove)
        return self.marginalize(*vars_remove)

    @property
    def name(self):
        vars_str = ",".join(self.left_vars)
        if len(self.right_vars) > 0:
            vars_str += "|" + ",".join(self.right_vars)
        return f"P({vars_str})"

    def __repr__(self):
        cardinality_dict = self.store.cardinality_dict
        card_str = ",".join([f"{v}:{cardinality_dict[v]}" for v in self._variables])
        return f"<{self.__class__.__name__} {self.name}, cardinality = ({card_str}), " \
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

def random_deterministic(dom:Dict, right_vars:list=None, vtype="numpy"):
    data = np.zeros([len(d) for d in dom.values()])
    for idx in [list(s.values()) for s in random_assignment(to_numeric_domains(dom), right_vars)]:
        set_value(1., data, idx)
    return MultinomialFactor(domain=dom, right_vars=right_vars, data=data, vtype=vtype)


if __name__ == "__main__":

    # P(B|A)
    left_domain = dict(A=["a1","a2"])
    right_domain = dict(B=[0,1, 3])
    domain = {**left_domain, **right_domain}

    data=np.array([[0.5, .4, 1.0], [0.3, 0.6, 0.1]])
    f = MultinomialFactor(domain, data, right_vars=["A"])

    f2 = MultinomialFactor(left_domain, data = [0.1, 0.9])
    f - f2

