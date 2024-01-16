from __future__ import annotations

import logging
import math
from collections import OrderedDict
from functools import reduce
from typing import Dict, List, Iterable, Union, Hashable

import numpy as np
import pandas as pd

from bcause.factors.values.store import  DataStore
from bcause.factors.values import store_dict

import bcause.factors.factor as bf
#from . import DiscreteFactor
from bcause.util.domainutils import assingment_space, state_space, steps, random_assignment, to_numeric_domains
from bcause.util.arrayutils import normalize_array, set_value


class MultinomialFactor(bf.DiscreteFactor, bf.ConditionalFactor):
    def __init__(self, domain:Dict, values, left_vars:list=None, right_vars:list=None, vtype=None):
        vtype = vtype or DataStore.DEFAULT_STORE

        self._check_domain(domain)
        if (isinstance(values, Iterable) and not isinstance(values, dict)) or np.isscalar(values):
            shape = [len(d) for d in domain.values()]
            if np.ndim(values)==0:
                values = [values] * int(np.prod(shape))
            if np.ndim(values)==1: values = np.reshape(values, shape)

        self._store = store_dict[vtype](data=values, domain=domain)
        self.set_variables(list(domain.keys()), left_vars, right_vars)
        self.vtype = vtype

        def builder(**kwargs):
            if "left_vars" not in kwargs and "right_vars" not in kwargs:
                kwargs["left_vars"] = self.left_vars
            return MultinomialFactor(**kwargs, vtype=vtype)

        self.builder = builder

    def to_deterministic(self):
        if len(self.left_vars) != 1:
            raise ValueError("Wrong number of variables on the left")

        v = self.left_vars[0]
        values = self.values_array().argmax(axis=self.variables.index(v))
        from bcause.factors import DeterministicFactor
        return DeterministicFactor(self.domain, left_vars=[v], values=values)

    def constant(self, left_value):

        new_dom = self.left_domain
        states = new_dom[self.left_vars[0]]

        if len(new_dom) != 1:
            raise ValueError("Only one variable on the left is allowed")

        if left_value not in states:
            raise ValueError("Value not in domain")

        new_data = [0.0] * len(states)
        new_data[states.index(left_value)] = 1.0

        return self.builder(domain=new_dom, values=new_data)


    # Factor operations
    def restrict(self, **observation) -> MultinomialFactor:
        if len(set(observation.keys()).intersection(self._variables))==0: return self
        new_store = self.store.restrict(**observation)
        new_right_vars = [v for v in new_store.variables if v in self.right_vars]
        return self.builder(domain=new_store.domain, values=new_store.data, right_vars = new_right_vars)



    def _prepare_operand(self, f):
        # if it's a scalar, transform it in a constant factor
        if type(f) in [int,float]: f = self.builder(domain=dict(), values=[f])
        return f

    def multiply(self, other : Union[MultinomialFactor, int, float]):

        other = self._prepare_operand(other)
        new_store = self.store.multiply(other.store)
        new_right_vars = [v for v in new_store.variables
                          if v not in self.left_vars and v not in other.left_vars]
        return self.builder(domain=new_store.domain, values=new_store.data, right_vars = new_right_vars)

    def addition(self, other):

        other = self._prepare_operand(other)
        new_store = self.store.addition(other.store)
        new_right_vars = [v for v in new_store.variables
                          if v not in self.left_vars and v not in other.left_vars]
        return self.builder(domain=new_store.domain, values=new_store.data, right_vars = new_right_vars)

    def subtract(self, other):
        other = self._prepare_operand(other)
        new_store = self.store.subtract(other.store)
        new_right_vars = [v for v in new_store.variables
                          if v not in self.left_vars and v not in other.left_vars]
        return self.builder(domain=new_store.domain, values=new_store.data, right_vars = new_right_vars)

    def divide(self, other):
        import warnings
        with warnings.catch_warnings(record=True) as W:
            other = self._prepare_operand(other)
            new_store = self.store.divide(other.store)
            new_right_vars = [v for v in new_store.variables
                              if v in self.right_vars or v in other.variables]
            out = self.builder(domain=new_store.domain, values=new_store.data, right_vars=new_right_vars)

            for w in W: logging.warning(f"{w.message}: {self.name}/{other.name}")

        return out

    def marginalize(self, *vars_remove) -> MultinomialFactor:
        if len(set(vars_remove).intersection(self._variables))==0: return self
        new_store = self.store.marginalize(*vars_remove)
        new_right_vars = [v for v in new_store.variables if v in self.right_vars]
        return self.builder(domain=new_store.domain, values=new_store.data, right_vars = new_right_vars)


    def maxmarginalize(self, *vars_remove) -> MultinomialFactor:
        if len(set(vars_remove).intersection(self._variables))==0: return self
        new_store = self.store.maxmarginalize(*vars_remove)
        new_right_vars = [v for v in new_store.variables if v in self.right_vars]
        return self.builder(domain=new_store.domain, values=new_store.data, right_vars = new_right_vars)



    def prob(self, observations:List[Dict]) -> List:
        return [self.get_value(**x) for x in observations]

    def log_prob(self, observations:List[Dict]) -> List:
        return [math.log(self.get_value(**x)) for x in observations]

    def sample(self, size:int = 1, varnames:bool=True) -> List:
        if size < 1:
            raise ValueError("sample size cannot be lower than 1.")
        return [self._sample(varnames=varnames) for _ in range(0,size)]

    def sample_conditional(self, observations : list[Dict], varnames:bool=True) -> List:
        if len(observations) < 1:
            raise ValueError("sample size cannot be lower than 1.")
        df_obs = pd.DataFrame(observations).drop_duplicates()
        factors = {tuple(val_obs):self.R(**dict(list(zip(list(df_obs.keys()), list(val_obs))))) for val_obs in df_obs.values}
        return [factors[tuple(obs.values())]._sample(varnames=varnames) for obs in observations]


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


def random_multinomial(domain:Dict, right_vars:list=None, vtype=None, allow_zero=True):
    vtype = vtype or DataStore.DEFAULT_STORE
    right_vars = right_vars or []
    left_dims = [i for i,v in enumerate(domain.keys()) if v not in right_vars]

    if allow_zero:
        data = normalize_array(np.random.uniform(0,1, size=[len(d) for d in domain.values()]), axis=left_dims)
    else:
        data = normalize_array(-1*np.random.uniform(-1, 0, size=[len(d) for d in domain.values()]), axis=left_dims)
    return MultinomialFactor(domain=domain, values=data, right_vars=right_vars, vtype=vtype)


def uniform_multinomial(domain:Dict, right_vars:list=None, vtype=None):
    vtype = vtype or DataStore.DEFAULT_STORE
    right_vars = right_vars or []
    left_dims = [i for i,v in enumerate(domain.keys()) if v not in right_vars]
    data = normalize_array(np.ones([len(d) for d in domain.values()]), axis=left_dims)
    return MultinomialFactor(domain=domain, values=data, right_vars=right_vars, vtype=vtype)

def random_deterministic(dom:Dict, right_vars:list=None, vtype=None):
    vtype = vtype or DataStore.DEFAULT_STORE
    data = np.zeros([len(d) for d in dom.values()])
    for idx in [list(s.values()) for s in random_assignment(to_numeric_domains(dom), right_vars)]:
        set_value(1., data, idx)
    return MultinomialFactor(domain=dom, right_vars=right_vars, values=data, vtype=vtype)

def canonical_multinomial(domain:Dict, exo_var:Hashable, right_endo_vars:list=None, vtype=None) -> MultinomialFactor:
    from bcause.factors.deterministic import canonical_deterministic
    return canonical_deterministic(domain, exo_var, right_endo_vars, vtype).to_multinomial()

if __name__ == "__main__":

    # P(B|A)
    left_domain = dict(A=["a1","a2"])
    right_domain = dict(B=[0,1, 3])
    domain = {**left_domain, **right_domain}

    data=np.array([[0.5, .4, 1.0], [0.3, 0.6, 0.1]])
    f = MultinomialFactor(domain, data, right_vars=["A"])

    f2 = MultinomialFactor(left_domain, values= [0.1, 0.9])
    f - f2

