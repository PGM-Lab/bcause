from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import math

import numpy as np

from bcause.factors.store import DataStore

from bcause.factors.store import store_dict
from util.domainutils import assingment_space, state_space


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
    def sample(self) -> float:
        pass


class DiscreteFactor(Factor):

    @property
    def domain(self) -> Dict:
        return self.store.domain

    @abstractmethod
    def restrict(self, **observation: Dict) -> DiscreteFactor:
        pass

    def R(self, **observation: Dict) -> DiscreteFactor:
        return self.restrict(**observation)


    @abstractmethod
    def multiply(self, other) -> DiscreteFactor:
        pass

    @abstractmethod
    def addition(self, other) -> DiscreteFactor:
        pass

    @abstractmethod
    def subtract(self, other) -> DiscreteFactor:
        pass

    @abstractmethod
    def divide(self, other) -> DiscreteFactor:
        pass

    @abstractmethod
    def marginalize(self, *vars_remove) -> DiscreteFactor:
        pass

    @abstractmethod
    def maxmarginalize(self, *vars_remove) -> DiscreteFactor:
        pass

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

