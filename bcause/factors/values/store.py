from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Iterable, Union

import numpy as np

import bcause.util.domainutils as dutil
from bcause.factors.values.operations import TreeStoreOperations, OperationSet
from bcause.util.treesutil import build_default_tree, treeNode


class DataStore(ABC):
    DEFAULT_STORE = "numpy"
    @classmethod
    def set_default(cls, vtype:str):
        cls.DEFAULT_STORE = vtype

class DiscreteStore(DataStore):

    def __init__(self, domain: Dict, data: Union[Iterable, int, float]):

        if not self._check_consistency(data, domain):
            raise ValueError("Cardinality Error")

        self._data = data
        self._domain = OrderedDict(domain)


        super().__init__()

    def copy(self, deep=True):
        new_data = self._copy_data() if deep else self._data
        new_dom = self.domain.copy()
        return self.builder(data=new_data, domain = new_dom)

    @property
    def variables(self):
        return list(self._domain.keys())

    @property
    def domain(self):
        return self._domain

    @property
    def data(self):
        return self._data

    @property
    def cardinality(self):
        return [len(d) for v,d in self._domain.items()]

    @property
    def cardinality_dict(self):
        return {v:len(d) for v,d in self._domain.items()}

    @staticmethod
    @abstractmethod
    def _check_consistency(data, domain):
        pass

    @abstractmethod
    def _copy_data(self):
        pass

    @abstractmethod
    def set_value(self, value, observation):
        pass


    def __repr__(self):
        cardinality_dict = self.cardinality_dict
        card_str = ",".join([f"{v}:{cardinality_dict[v]}" for v in self.variables])
        vars_str = ",".join([f"{v}" for v in self.variables])
        return f"<{self.__class__.__name__}({vars_str}), cardinality = ({card_str})>"


    @abstractmethod
    def restrict(self, **observation) -> DiscreteStore:
        pass

    @abstractmethod
    def get_value(self, **observation) -> NumpyStore:
        pass

    @abstractmethod
    def set_value(self,value, observation) -> float:
        pass

    def set_operationSet(self, ops:OperationSet):
        for f in [self.set_marginalize, self.set_maxmarginalize, self.set_multiply,
                  self.set_addition, self.set_subtract, self.set_divide]:
            f(ops)

    def set_marginalize(self, ops:OperationSet):
        self._marginalize = ops.marginalize

    def set_maxmarginalize(self, ops:OperationSet):
        self._maxmarginalize = ops.maxmarginalize

    def set_multiply(self, ops:OperationSet):
        self._multiply = ops.multiply

    def set_addition(self, ops:OperationSet):
        self._addition = ops.addition

    def set_subtract(self, ops:OperationSet):
        self._subtract = ops.subtract

    def set_divide(self, ops:OperationSet):
        self._divide = ops.divide

    def marginalize(self, *vars_remove) -> DiscreteStore:
        return self._marginalize(self, vars_remove)

    def maxmarginalize(self, *vars_remove) -> DiscreteStore:
        return self._maxmarginalize(self, vars_remove)

    def multiply(self, other:DiscreteStore) -> DiscreteStore:
        return self._multiply(self, other)

    def addition(self, other:DiscreteStore) -> DiscreteStore:
        return self._addition(self, other)

    def subtract(self, other: DiscreteStore) -> DiscreteStore:
        return self._subtract(self, other)

    def divide(self, other: DiscreteStore) -> DiscreteStore:
        return self._divide(self, other)

    @abstractmethod
    def sum_all(self):
        pass

    @property
    def values_list(self) -> list:
        return [self.get_value(**x) for x in dutil.assingment_space(self.domain)]
    @property
    def values_dict(self) -> dict:
        return {tuple(zip(d.keys(), d.values())): self.get_value(**d) for d in dutil.assingment_space(self.domain)}

    def values_str(self, maxvalues = 4):
        vals = self.values_list
        output = ",".join([str(x) for x in vals[0:maxvalues]])
        if len(vals) > maxvalues:
            output += f",...,{vals[-1]}"

        return output


    def sum_all(self) -> float:
        return sum(self.values_list)


    def get_var_index(self, v):
        if v not in self.variables:
            raise ValueError(f"Error: {v} not present in data store")
        gen = (i for i, e in enumerate(self.variables) if e == v)
        return next(gen)






class TreeStore(DiscreteStore):

    def __init__(self, domain: Dict, data: Union[Iterable, int, float, dict]=None):
        #defualt data
        if data is None:
            data = np.zeros(np.prod([len(d) for d in domain.values()]))
        if len(domain)>0 and type(data) not in [dict, OrderedDict]:
            data = build_default_tree(domain, data)
        def builder(**kwargs):
            return TreeStore(**kwargs)

        self.builder = builder
        self.set_operationSet(TreeStoreOperations)   # set implement
        super(self.__class__, self).__init__(domain=domain, data=data)

    @staticmethod
    def _check_consistency(data, domain):
        return True

    def _copy_data(self):
        return copy.deepcopy(self.data)

    def set_value(self, value, observation):
        def modify_dict(data, observation, value):
            if not isinstance(data, dict) or len(observation) == 0:
                out = value
            elif data["var"] not in observation:
                new_ch = dict()
                for state, ch in data["children"].items():
                    new_ch[state] = value if not isinstance(ch, dict) else modify_dict(ch, observation, value)
                out = treeNode(data["var"], new_ch)

            else:
                other_obs = {v: s for v, s in observation.items() if v != data["var"]}
                ch = data["children"][observation[data["var"]]]
                data["children"][observation[data["var"]]] = modify_dict(ch, other_obs, value)
                out = data
            return out

        relevant_obs = {v: s for v, s in observation.items() if v in self.domain}
        self._data = modify_dict(self.data, relevant_obs, value)

    def get_value(self, **observation):
        return self.restrict(**observation).data

    def restrict(self, **observation) -> TreeStore:
        new_data = TreeStore.restrict_dict(self.data, observation)
        new_dom = OrderedDict([(k, d) for k, d in self.domain.items() if k not in observation])
        return self.builder(data = new_data, domain = new_dom)


if __name__=="__main__":
    left_domain = dict(A=["a1", "a2"])
    right_domain = dict(B=[0, 1, 3])
    domain = {**left_domain, **right_domain}

    new_var_order = ["B", "A"]
    #complete vars

    new_dom = OrderedDict([(v,domain[v]) for v in new_var_order])

    data = [[0.5, .4, 0.1], [0.3, 0.6, 0.1]]

    vars_remove = ["B"]
    f1 = NumpyStore(domain, data)
    f2 = ListStore(domain, data)

    for f in [f1,f2]:
        print(f.multiply(f.restrict(B=1)).marginalize("A").data)




