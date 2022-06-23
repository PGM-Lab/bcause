from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import numpy as np
import networkx as nx

import bcause.util.domainutils as dutils

from bcause.factors import MultinomialFactor
from bcause.factors.values.store import store_dict



import bcause.factors.factor as bf


class DeterministicFactor(bf.DiscreteFactor, bf.ConditionalFactor):

    def __init__(self, domain: Dict, data, left_vars:list=None, right_vars:list=None, vtype="numpy"):

        self._domain = OrderedDict(domain)
        self.set_variables(list(domain.keys()), left_vars, right_vars)

        if np.ndim(data)==1: data = np.reshape(data, [len(d) for d in self.right_domain.values()])

        self._store = store_dict[vtype](data=data, domain=self.right_domain)
        self.vtype = vtype

        def builder(*args, **kwargs):
            return DeterministicFactor(*args, **kwargs, vtype=vtype)

        self.builder = builder

    @property
    def domain(self) -> Dict:
        return self._domain

    def eval(self, **observation):
        righ_obs = {v: observation[v] for v in self.right_vars}
        return self.get_value(**righ_obs) == observation[self.left_vars[0]]

    def to_multinomial(self, as_int=False):
        cast = int if as_int else float
        data = [cast(self.eval(**obs)) for obs in dutils.assingment_space(self.domain)]
        return MultinomialFactor(self.domain, right_vars=self.right_vars, data=data, vtype=self.vtype)

    def to_values_array(self, var_order = None) -> np.array:
        return super().to_values_array(var_order or self.right_vars)


    def sample(self) -> float:
        raise NotImplementedError("Method not available")

    def restrict(self, **observation: Dict) -> DeterministicFactor:
        return self.store.restrict(**observation)

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
        if len(self.right_vars) > 0:
            right_str = ",".join(self.right_vars)
        return f"f{left_str}({right_str})"

    def __repr__(self):
        card_str = ",".join([f"{v}:{len(self.domain[v])}" for v in self._variables])
        return f"<{self.__class__.__name__} {self.name}, cardinality = ({card_str}), " \
               f"values=[{self.store.values_str()}]>"

