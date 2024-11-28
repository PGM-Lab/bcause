from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import reduce
from typing import Dict, List

import numpy as np

from bcause.factors.values.store import DataStore
from bcause.util.domainutils import assingment_space


class Factor(ABC):
    """
    Abstract base class for factors, representing mathematical objects with a defined domain and operations.
    """

    @property
    def store(self) -> DataStore:
        """Returns the data store associated with the factor."""
        return self._store

    @property
    def variables(self) -> list:
        """Returns the list of variables in the factor."""
        return self._variables

    @property
    @abstractmethod
    def domain(self) -> Dict:
        """Returns the domain of the factor."""
        pass

    @abstractmethod
    def sample(self, size: int, varnames: bool) -> float:
        """Samples values from the factor."""
        pass

    @abstractmethod
    def sample_conditional(self, observations: list[Dict], varnames: bool) -> float:
        """Samples values conditionally given observations."""
        pass

    @abstractmethod
    def restrict(self, **observation: Dict) -> Factor:
        """Restricts the factor to a specific set of observations."""
        pass

    def R(self, **observation: Dict) -> Factor:
        """Alias for the restrict method."""
        return self.restrict(**observation)

    @abstractmethod
    def multiply(self, other) -> Factor:
        """Multiplies the factor with another factor."""
        pass

    @abstractmethod
    def addition(self, other) -> Factor:
        """Adds another factor to this factor."""
        pass

    @abstractmethod
    def subtract(self, other) -> Factor:
        """Subtracts another factor from this factor."""
        pass

    @abstractmethod
    def divide(self, other) -> Factor:
        """Divides the factor by another factor."""
        pass

    @abstractmethod
    def marginalize(self, *vars_remove) -> Factor:
        """Marginalizes out specified variables."""
        pass

    @abstractmethod
    def maxmarginalize(self, *vars_remove) -> Factor:
        """Maximizes and marginalizes out specified variables."""
        pass

    @staticmethod
    def combine_all(*factors) -> Factor:
        """Combines multiple factors by multiplication."""
        return reduce((lambda f1, f2: f1 * f2), factors)


class DiscreteFactor(Factor):
    """
    A specific implementation of a factor for discrete variables.
    """

    def _check_domain(self, domain):
        """Validates the format of the domain."""
        if not isinstance(domain, dict) \
                or not all([isinstance(v, str) for v in domain.keys()]) \
                or not all([isinstance(v, list) for v in domain.values()]) \
                or not all([len(set([type(v) for v in d])) == 1 for d in domain.values()]):
            raise ValueError("Domain must be a dictionary with string keys and list values.")

        if any(v.startswith("_") for v in domain.keys()):
            raise ValueError("Variable names cannot start with an underscore.")

    @property
    def domain(self) -> Dict:
        """Returns the domain of the discrete factor."""
        return self.store.domain

    def get_value(self, **observation) -> float:
        """Retrieves the value associated with an observation."""
        return self.store.get_value(**observation)

    @property
    def values(self) -> List:
        """Returns the values of the factor."""
        return self.store.values_list

    @property
    def values_list(self) -> Dict:
        """Returns the values of the factor as a list."""
        return self.store.values_list

    @property
    def values_dict(self) -> Dict:
        """Returns the values of the factor as a dictionary."""
        return self.store.values_dict

    def values_array(self, var_order=None) -> np.array:
        """Returns the factor values as a numpy array."""
        var_order = var_order or list(self.domain.keys())
        dom_order = OrderedDict([(v, self.domain[v]) for v in var_order])
        shape = [len(dom_order[v]) for v in var_order]
        return np.array([self.store.get_value(**s)
                         for s in assingment_space(dom_order)
                         ]).reshape(shape)

    def is_degenerate(self):
        """Checks if the factor is degenerate (only 0 or 1 values)."""
        return len([x for x in self.values_list if x != 0 and x != 1]) == 0

    def rename_vars(self, names_mapping) -> DiscreteFactor:
        """Renames variables in the factor."""
        kwargs = dict()
        kwargs["values"] = self.values
        kwargs["domain"] = OrderedDict(
            [(v, d) if v not in names_mapping else (names_mapping[v], d) for v, d in self.domain.items()])

        if isinstance(self, ConditionalFactor):
            kwargs["left_vars"] = [v if v not in names_mapping else names_mapping[v] for v in self.left_vars]
            kwargs["right_vars"] = [v if v not in names_mapping else names_mapping[v] for v in self.right_vars]

        return self.builder(**kwargs)

    def reorder(self, *var_order):
        """Reorders the variables in the factor."""
        var_order = list(var_order) + [v for v in self.variables if v not in var_order]
        new_dom = {x: self.domain[x] for x in var_order}
        new_vals = self.values_array(var_order)
        return self.builder(domain=new_dom, values=new_vals)

    def change_domains(self, **domains):
        """Changes the domain of specific variables."""
        new_dom = self.domain
        for v in new_dom.keys():
            if v in domains:
                new_dom[v] = domains[v]
        return self.builder(domain=new_dom, values=self.values)


class ConditionalFactor(Factor):
    """
    Represents a conditional factor with left-hand and right-hand variables.
    """

    @property
    def right_vars(self) -> set:
        """Returns the set of right-hand variables."""
        return self._right_vars

    @property
    def left_vars(self) -> list:
        """Returns the list of left-hand variables."""
        return [v for v in self._variables if v not in self.right_vars]

    @property
    def right_domain(self) -> Dict:
        """Returns the domain of the right-hand variables."""
        return {v: s for v, s in self.domain.items() if v in self.right_vars}

    @property
    def left_domain(self) -> Dict:
        """Returns the domain of the left-hand variables."""
        return {v: s for v, s in self.domain.items() if v in self.left_vars}

    def set_variables(self, variables, left_vars, right_vars):
        """
        Sets variables for the conditional factor, splitting into left and right sides.

        Args:
            variables: The full set of variables.
            left_vars: Variables for the left-hand side.
            right_vars: Variables for the right-hand side.
        """
        if left_vars is None and right_vars is None:
            left_vars, right_vars = variables, []
        elif left_vars is not None and right_vars is None:
            right_vars = [v for v in variables if v not in left_vars]
        elif left_vars is not None and right_vars is not None:
            if not (set(right_vars).union(left_vars) == set(variables) and set(right_vars).isdisjoint(left_vars)):
                raise ValueError("Cannot determine left/right side variables")

        self._variables, self._right_vars = variables, right_vars
