from __future__ import annotations

import logging
from collections import OrderedDict
from itertools import product
from typing import Dict, List, Hashable

import numpy as np
import pandas as pd

import bcause.util.domainutils as dutils

from bcause.factors import MultinomialFactor
from bcause.factors.values.store import DataStore
from bcause.factors.values import store_dict

import bcause.factors.factor as bf


class DeterministicFactor(bf.DiscreteFactor, bf.ConditionalFactor):

    def __init__(self, domain: Dict, values, left_vars:list=None, right_vars:list=None, vtype=None): # TODO: what is left_vars and right_vars related to? P(LEFT|RIGHT)?
                                                                                                     #       If so, left_vars->event and right_vars->evidence(s) or condition(s)
        vtype = vtype or DataStore.DEFAULT_STORE

        self._domain = OrderedDict(domain) # TODO: Rafa, I think dict() also keeps the order of insertion from Python 3.7
        self.set_variables(list(domain.keys()), left_vars, right_vars)

        if len(self.left_vars)!=1: raise ValueError("Wrong number of left variables")

        if np.ndim(values)==1: values = np.reshape(values, [len(d) for d in self.right_domain.values()])

        self._store = store_dict[vtype](data=values, domain=self.right_domain)
        self.vtype = vtype

        def builder(**kwargs):
            if "left_vars" not in kwargs and "right_vars" not in kwargs:
                kwargs["left_vars"] = self.left_vars
            return DeterministicFactor(**kwargs, vtype=vtype)

        self.builder = builder

    @staticmethod
    def from_data_frame(df, domains, left_var=None):
        columns = list(df.columns)
        left_var = left_var or columns[-1]

        for v in columns:
            df[v] = pd.Categorical(df[v], domains[v])

        df = df.sort_values(by=columns)

        domf = dutils.subdomain(domains, *columns)
        values = list(df[left_var].values)
        return DeterministicFactor(domf, left_vars=left_var, values=values)

    @property
    def domain(self) -> Dict:
        return self._domain

    def eval(self, **observation):
        righ_obs = {v: observation[v] for v in self.right_vars}
        return self.get_value(**righ_obs) == observation[self.left_vars[0]]

    def to_multinomial(self, as_int=False):
        logging.debug(f"Casting deterministic function {self.name} to multinomial")
        cast = int if as_int else float
        values = [cast(self.eval(**obs)) for obs in dutils.assingment_space(self.domain)]
        return MultinomialFactor(self.domain, right_vars=self.right_vars, values=values, vtype=self.vtype)

    def values_array(self, var_order = None) -> np.array:
        return super().values_array(var_order or self.right_vars)

    def constant(self, left_value):
        new_dom = self.left_domain
        if len(new_dom) != 1:
            raise ValueError("Only one varible on the left is allowed")
        if left_value not in new_dom[self.left_vars[0]]:
            raise ValueError("Value not in domain")
        return self.builder(domain=new_dom, values=[left_value])

    def sample(self, size:int, varnames:bool) -> float:
        raise NotImplementedError("Method not available")

    def sample_conditional(self, observations:list[Dict], varnames:bool) -> float:
        raise NotImplementedError("Method not available")

    def restrict(self, **observation: Dict) -> DeterministicFactor:
        raise NotImplementedError("Method not available")
        #
        # if not set(observation.keys()).isdisjoint(self.left_vars):
        #     raise ValueError("It is not allowed to restrict to a left-side variable")
        # new_data = self.store.restrict(**observation)
        # new_dom = {v:d for v,d in self.domain.items() if v not in observation}
        # return self.builder(domain=new_dom, left_vars=self.left_vars, data=new_data)


    def multiply(self, other) -> DeterministicFactor:
        raise NotImplementedError("Method not available")

    def addition(self, other) -> DeterministicFactor:
        raise NotImplementedError("Method not available")

    def subtract(self, other) -> DeterministicFactor:
        raise NotImplementedError("Method not available")

    def divide(self, other) -> DeterministicFactor:
        raise NotImplementedError("Method not available")

    def marginalize(self, *vars_remove) -> DeterministicFactor:
        raise NotImplementedError("Method not available")

    def maxmarginalize(self, *vars_remove) -> DeterministicFactor:
        raise NotImplementedError("Method not available")

    @property
    def name(self):
        left_str = "".join(self.left_vars)
        right_str = ""
        if len(self.right_vars) > 0:
            right_str = ",".join(self.right_vars)
        return f"f{left_str}({right_str})"

    def __repr__(self):
        card_str = ",".join([f"{v}:{len(self.domain[v])}" for v in self._variables])
        return f"<{self.__class__.__name__} {self.name}, cardinality = ({card_str}), " \
               f"values=[{self.store.values_str()}]>"




def canonical_deterministic(domain:Dict, exo_var:Hashable, right_endo_vars:list=None, vtype=None) -> MultinomialFactor:
    vtype = vtype or DataStore.DEFAULT_STORE

    left_var = [x for x in domain.keys() if x not in right_endo_vars and x != exo_var]
    if len(left_var) != 1:
        raise ValueError("Canonical for non-markovian")

    left_var = left_var[0]

    domEndoPa = dutils.assingment_space(dutils.subdomain(domain, *right_endo_vars))
    m = len(domEndoPa)

    new_values = list(product(*[domain[left_var]] * m))
    new_dom = {**dutils.subdomain(domain, left_var), **dutils.subdomain(domain, exo_var), **dutils.subdomain(domain, *right_endo_vars)}

    return DeterministicFactor(domain=new_dom, values=new_values, left_vars=[left_var], vtype=vtype)

