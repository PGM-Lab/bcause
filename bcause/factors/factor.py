from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

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
        self._variables

    @property
    @abstractmethod
    def domain(self) -> Dict:
        pass


    @abstractmethod
    def sample(self) -> float:
        pass
    



class DiscreteFactor(Factor):

    @abstractmethod
    def restrict(self, **observation: Dict) -> DiscreteFactor:
        pass



class MultinomialFactor(DiscreteFactor):

    def __init__(self, domain:Dict, data, right_vars:list=None, vtype="numpy"):

        self._store = store_dict[vtype](data, domain)
        self._right_vars = right_vars or []
        self._variables = list(domain.keys())

        def builder(*args, **kwargs):
            return MultinomialFactor(*args, **kwargs, vtype=vtype)

        self.builder = builder


    @property
    def right_vars(self) -> list:
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

    @property
    def domain(self) -> Dict:
        return self.store.domain

    def restrict(self, **observation) -> MultinomialFactor:
        new_store = self.store.restrict(**observation)
        new_right_vars = [v for v in new_store.variables if v in self.right_vars]
        return self.builder(new_store.domain, new_store.data, new_right_vars)

    def sample(self) -> tuple:
        # joint or marginal distribution
        if len(self.right_vars) == 0:
            possible_states = state_space(self.left_domain)
            observations = assingment_space(self.left_domain)
            probs = np.array([float(self.store.restrict(**obs).data) for obs in observations])
            idx = np.random.choice(list(range(0, len(possible_states))), p=probs / probs.sum())
            sample = possible_states[idx]
            return sample
        ## conditional distribution
        else:
            return [self.restrict(**obs).sample() for obs in assingment_space(self.right_domain)]

    def __repr__(self):
        cardinality_dict = self.store.cardinality_dict
        card_str = ",".join([f"{v}:{cardinality_dict[v]}" for v in self._variables])
        vars_str = ",".join(self.left_vars)
        if len(self.right_vars)>0:
            vars_str += "|"+",".join(self.right_vars)
        return f"<{self.__class__.__name__} P({vars_str}), cardinality = ({card_str}), " \
               f"values=[{self.store.values_str()}]>"



if __name__ == "__main__":
    left_domain = dict(A=["a1","a2"])
    right_domain = dict(B=[0,1, 3])
    domain = {**left_domain, **right_domain}

    data=[[0.5, .4, 1.0], [0.3, 0.6, 0.1]]

    f = MultinomialFactor(domain, data, right_vars=["A"])
    print(f)
    f.restrict(A="a1").sample()
    f.sample()
