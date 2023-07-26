from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import reduce
from typing import Dict, List

import numpy as np

from bcause.factors.values.store import DataStore
from bcause.util.domainutils import assingment_space


class Factor(ABC):
    @property
    def store(self) -> DataStore:
        return self._store

    @property
    def variables(self) -> list:
        return self._variables

    @property
    @abstractmethod
    def domain(self) -> Dict:
        pass

    @abstractmethod
    def sample(self, size:int, varnames:bool) -> float:
        pass

    @abstractmethod
    def sample_conditional(self, observations:list[Dict], varnames:bool) -> float:
        pass

    @abstractmethod
    def restrict(self, **observation: Dict) -> Factor:
        pass

    def R(self, **observation: Dict) -> Factor:
        return self.restrict(**observation)

    @abstractmethod
    def multiply(self, other) -> Factor:
        pass

    @abstractmethod
    def addition(self, other) -> Factor:
        pass

    @abstractmethod
    def subtract(self, other) -> Factor:
        pass

    @abstractmethod
    def divide(self, other) -> Factor:
        pass

    @abstractmethod
    def marginalize(self, *vars_remove) -> Factor:
        pass

    @abstractmethod
    def maxmarginalize(self, *vars_remove) -> Factor:
        pass

    @staticmethod
    def combine_all(*factors) -> Factor:
        return reduce((lambda f1, f2: f1 * f2), factors)


class DiscreteFactor(Factor):
    def _check_domain(self, domain):
        if not isinstance(domain, dict) \
                or not all([isinstance(v, str) for v in domain.keys()]) \
                or not all([isinstance(v, list) for v in domain.values()]) \
                or not all([len(set([type(v) for v in d]))==1 for d in domain.values()]):
            raise ValueError("Wrong domain format: it must be a dictionary with keys of class str and values of class list.")

        if any(v.startswith("_") for v in domain.keys()):
            raise ValueError("Variables names cannot start with an underscore")

    @property
    def domain(self) -> Dict:
        return self.store.domain

    def get_value(self, **observation) -> float:
        return self.store.get_value(**observation)

    @property
    def values(self)->List:
        return self.store.values_list

    @property
    def values_list(self) -> Dict:
        return self.store.values_list

    @property
    def values_dict(self) -> Dict:
        return self.store.values_dict

    def values_array(self, var_order = None) -> np.array:
        var_order  = var_order or list(self.domain.keys())
        dom_order = OrderedDict([(v,self.domain[v]) for v in var_order])
        shape = [len(dom_order[v]) for v in var_order]
        return np.array([self.store.get_value(**s)
                  for s in assingment_space(dom_order)
                  ]).reshape(shape)

    def is_degenerate(self):
        return len([x for x in self.values_list if x != 0 and x != 1]) == 0

    def rename_vars(self, names_mapping) -> DiscreteFactor:
        kwargs = dict()
        kwargs["values"] = self.values
        kwargs["domain"] = OrderedDict(
            [(v, d) if v not in names_mapping else (names_mapping[v], d) for v, d in self.domain.items()])


        if isinstance(self, ConditionalFactor):
            kwargs["left_vars"] = [v if v not in names_mapping else names_mapping[v] for v in self.left_vars]
            kwargs["right_vars"] = [v if v not in names_mapping else names_mapping[v] for v in self.right_vars]

        return self.builder(**kwargs)



class ConditionalFactor(Factor):
    @property
    def right_vars(self) -> set:
        return self._right_vars

    @property
    def left_vars(self) -> list:
        return [v for v in self._variables if v not in self.right_vars]

    @property
    def right_domain(self) -> Dict:
        return {v:s for v,s in self.domain.items() if v in self.right_vars}

    @property
    def left_domain(self) -> Dict:
        return {v:s for v,s in self.domain.items() if v in self.left_vars}

    def set_variables(self, variables, left_vars, right_vars):
        if left_vars is None and right_vars is None:
            # all the variables are in the left side
            left_vars, right_vars = variables, []
        elif left_vars is not None and right_vars is None:
            right_vars = [v for v in variables if v not in left_vars]
        elif left_vars is not None and right_vars is not None:
            if not (set(right_vars).union(left_vars) == set(variables) and set(right_vars).isdisjoint(left_vars)):
                raise ValueError("Cannot determine left/right side variables")

        self._variables, self._right_vars = variables, right_vars


