from __future__ import annotations

import logging
import math
from collections import OrderedDict
from functools import reduce
from typing import Dict, List, Iterable, Union

import numpy as np
import pandas as pd

from bcause.factors import MultinomialFactor
from bcause.factors.values.store import  DataStore
from bcause.factors.values import store_dict

import bcause.factors.factor as bf
#from . import DiscreteFactor
from bcause.util.domainutils import assingment_space, state_space, steps, random_assignment, to_numeric_domains
from bcause.util.arrayutils import normalize_array, set_value


class IntervalProbFactor(bf.DiscreteFactor, bf.ConditionalFactor):
    def __init__(self, domain:Dict, values_low, values_up, left_vars:list=None, right_vars:list=None, vtype=None):
        vtype = vtype or DataStore.DEFAULT_STORE

        self._check_domain(domain)
        def process_values(values):
            if (isinstance(values, Iterable) and not isinstance(values, dict)) or np.isscalar(values):
                shape = [len(d) for d in domain.values()]
                if np.ndim(values)==0:
                    values = [values] * int(np.prod(shape))
                if np.ndim(values)==1: values = np.reshape(values, shape)
            return np.array(values)

        values_low,values_up = process_values(values_low), process_values(values_up)
        domain = {**dict(_bound=["low","up"]), **domain}


        if values_low.shape != values_up.shape:
            raise ValueError("Data of different dimensions")

        values = np.array([values_low, values_up])


        self._store = store_dict[vtype](data=values, domain=domain)
        self.set_variables(list(domain.keys()), left_vars, right_vars)
        self.vtype = vtype

        def builder(**kwargs):
            if "left_vars" not in kwargs and "right_vars" not in kwargs:
                kwargs["left_vars"] = self.left_vars
            return IntervalProbFactor(**kwargs, vtype=vtype)

        self.builder = builder

    @staticmethod
    def from_precise(list_factors):
        dom = list_factors[0].domain
        left_vars = list_factors[0].left_vars
        if not all(list_factors[0].domain == f.domain for f in list_factors):
            raise ValueError("Domains must be identical for building an IntervalFactor")

        values = np.array([[f.get_value(**obs) for f in list_factors] for obs in assingment_space(dom)])
        return IntervalProbFactor(dom, values.min(axis=1), values.max(axis=1), left_vars=left_vars)

    @property
    def store_low(self):
        return self.store.restrict(_bound="low")

    @property
    def store_up(self):
        return self.store.restrict(_bound="up")


    @property
    def variables(self) -> list:
        return [v for v in self._variables if not v.startswith("_")]

    @property
    def left_vars(self) -> list:
        return [v for v in self._left_vars if not v.startswith("_")]
    @property
    def left_vars(self) -> list:
        return [v for v in self._variables if v not in self.right_vars and not v.startswith("_")]


    @property
    def domain(self) -> Dict:
        return OrderedDict({v:s for v,s in self.store.domain.items() if not v.startswith("_")})


    # Factor operations
    def restrict(self, **observation) -> IntervalProbFactor:
        if len(set(observation.keys()).intersection(self._variables)) == 0: return self
        new_store = self.store.restrict(**observation)
        new_left_vars = [v for v in new_store.variables if v in self.left_vars]
        new_dom = {v:k for v,k in new_store.domain.items() if not v.startswith("_")}
        values_low = new_store.restrict(_bound="low").data
        values_up = new_store.restrict(_bound="up").data
        return self.builder(domain=new_dom, values_low=values_low, values_up=values_up, left_vars=new_left_vars)

    def _prepare_operand(self, f):
        raise NotImplementedError()

    def multiply(self, other : Union[IntervalProbFactor, int, float]):
        raise NotImplementedError()


    def addition(self, other):
        raise NotImplementedError()


    def subtract(self, other):
        raise NotImplementedError()


    def divide(self, other):
        import warnings
        with warnings.catch_warnings(record=True) as W:
            raise NotImplementedError()

            for w in W: logging.warning(f"{w.message}: {self.name}/{other.name}")

        return out

    def marginalize(self, *vars_remove) -> IntervalProbFactor:
        raise NotImplementedError()

    def maxmarginalize(self, *vars_remove) -> IntervalProbFactor:
        raise NotImplementedError()

    def prob(self, observations:List[Dict]) -> List:
        return [self.get_value(**x) for x in observations]

    def log_prob(self, observations:List[Dict]) -> List:
        raise NotImplementedError() # Note: check how log works with a tuple

        #return [math.log(self.get_value(**x)) for x in observations]

    def sample(self, size:int = 1, varnames:bool=True) -> List:
        if size < 1:
            raise ValueError("sample size cannot be lower than 1.")
        return [self._sample(varnames=varnames) for _ in range(0,size)]

    def sample_conditional(self, observations : list[Dict], varnames:bool=True) -> List:
        raise NotImplementedError()

    def _sample(self, varnames:bool=True) -> tuple:
        raise NotImplementedError()



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
        right_vars = [v for v in self.right_vars if not v.startswith("_")]
        vars_str = ",".join(self.left_vars)
        if len(right_vars) > 0:
            vars_str += "|" + ",".join(right_vars)
        return f"P({vars_str})"

    def __repr__(self):
        cardinality_dict = {k:v for k,v in self.store.cardinality_dict.items() if not k.startswith("_")}
        card_str = ",".join([f"{v}:{cardinality_dict[v]}" for v in self.variables])
        return f"<{self.__class__.__name__} {self.name}, cardinality = ({card_str}), " \
               f"values_low=[{self.store.restrict(_bound = 'low').values_str()}], " \
               f"values_up=[{self.store.restrict(_bound = 'up').values_str()}]" \
               f">"


if __name__ == "__main__":

    # P(B|A)
    left_domain = dict(A=["a1","a2"])
    right_domain = dict(B=[0,1, 3])
    domain = {**left_domain, **right_domain}


    list_factors = []
    f = MultinomialFactor(domain, np.array([[0.2, .6, 0.2], [0.3, 0.6, 0.1]]), right_vars=["A"])
    list_factors.append(f)
    f = MultinomialFactor(domain, np.array([[0.1, .7, 0.2], [0.0, 0.8, 0.2]]), right_vars=["A"])
    list_factors.append(f)

    ifactor = IntervalProbFactor.from_precise(list_factors)

    print(ifactor)

    self = ifactor
    ifactor.left_domain
    ifactor.right_domain
