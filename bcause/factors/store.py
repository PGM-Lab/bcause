from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Iterable, Union

import numpy as np

import util.domainutils as dutil


class DataStore(ABC):
    pass

class DiscreteStore(DataStore):

    def __init__(self, data:Union[Iterable, int, float], domain:Dict):

        if not self._check_consistency(data, domain):
            raise ValueError("Cardinality Error")

        self._data = data
        self._domain = OrderedDict(domain)

        super().__init__()

    def copy(self, deep=True):
        new_data = self._copy_data() if deep else self._data
        new_dom = self.domain.copy()
        return self.builder(new_data, new_dom)

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


    def __repr__(self):
        cardinality_dict = self.cardinality_dict
        card_str = ",".join([f"{v}:{cardinality_dict[v]}" for v in self.variables])
        vars_str = ",".join([f"{v}" for v in self.variables])
        return f"<{self.__class__.__name__}({vars_str}), cardinality = ({card_str})>"


    @abstractmethod
    def restrict(self, **observation) -> DiscreteStore:
        pass

    def get_value(self, **observation) -> float:
        if len([v for v in self.variables if v not in observation.keys()])>0:
            raise ValueError("Missing value for any of the variables in the domain")
        return float(self.restrict(**observation).data)

    @abstractmethod
    def marginalize(self, *vars_remove) -> DiscreteStore:
        pass

    @abstractmethod
    def maxmarginalize(self, *vars_remove) -> DiscreteStore:
        pass

    @abstractmethod
    def multiply(self, other:DiscreteStore) -> DiscreteStore:
        pass

    @abstractmethod
    def addition(self, other:DiscreteStore) -> DiscreteStore:
        pass

    @abstractmethod
    def subtract(self, other: DiscreteStore) -> DiscreteStore:
        pass

    @abstractmethod
    def divide(self, other: DiscreteStore) -> DiscreteStore:
        pass

    @abstractmethod
    def sum_all(self):
        pass


class NumpyStore(DiscreteStore):
    def __init__(self, data:Union[Iterable, int, float], domain:Dict):

        def builder(*args, **kwargs):
            return NumpyStore(*args, **kwargs)

        self.builder = builder
        super(self.__class__, self).__init__(data=np.array(data), domain=domain)

    @property
    def cardinality(self) -> tuple:
        return np.shape(self.data)

    @staticmethod
    def _check_consistency(data, domain):
        return (np.ndim(data) == len(domain)) \
               and (list(np.shape(data)) == [len(d) for v,d in domain.items()])

    def _copy_data(self):
        return self._data.copy()


    def sum_all(self) -> float:
        return sum(self._data.flatten())

    def restrict(self, **observation) -> NumpyStore:
        items = []
        for v in self.variables:
            if v in observation.keys():
                idx = np.where(np.array(self.domain[v]) == observation[v])[0][0]
                items.append(idx)
            else:
                items.append(slice(None))
        new_data = self._data[tuple(items)].copy()
        new_dom = {k:d for k,d in self.domain.items() if k not in observation}
        return NumpyStore(new_data, new_dom)



    def values_str(self, maxvalues = 4):
        vals = self.data.flatten()
        output = ",".join([str(x) for x in vals[0:maxvalues]])
        if self.data.size > maxvalues:
            output += f",...,{vals[-1]}"

        return output

    def get_var_index(self, v):
        if v not in self.variables:
            raise ValueError(f"Error: {v} not present in data store")
        gen = (i for i, e in enumerate(self.variables) if e == v)
        return next(gen)


    def marginalize(self, *vars_remove) -> NumpyStore:
        idx_vars = tuple(self.get_var_index(v) for v in vars_remove)
        new_data = np.sum(self.data, axis=idx_vars)
        new_dom = {v: d for v, d in self.domain.items() if v not in vars_remove}
        return self.builder(new_data, new_dom)


    def maxmarginalize(self, *vars_remove) -> NumpyStore:
        idx_vars = tuple(self.get_var_index(v) for v in vars_remove)
        new_data = np.max(self.data, axis=idx_vars)
        new_dom = {v: d for v, d in self.domain.items() if v not in vars_remove}
        return self.builder(new_data, new_dom)

    def reorder(self, *new_var_order):
        if len(self.variables)<2:
            return self

        # filter variables and add remaining variables
        new_var_order = [v for v in new_var_order if v in self.variables]
        new_var_order += [v for v in self.variables if v not in new_var_order]

        idx_var_order = [new_var_order.index(v) for v in self.variables]

        # set the new order in the values
        new_data = np.moveaxis(self.data, range(np.ndim(self.data)), idx_var_order)
        new_dom = OrderedDict([(v, self.domain[v]) for v in new_var_order])

        # create new object
        return self.builder(data = new_data, domain = new_dom)

    def extend_domain(self, **extra_dom):

        extra_dom = OrderedDict({v:c for v,c in extra_dom.items() if v not in self.variables})
        add_card = tuple([len(d) for d in extra_dom.values()])

        new_data = np.reshape(np.repeat(self.data, np.prod(add_card)), np.shape(self.data) + add_card)
        new_dom = {**self.domain, **extra_dom}

        return self.builder(data=new_data, domain=new_dom)

    def _generic_combine(self, op2: NumpyStore, operation:callable) -> NumpyStore:

        op1 = self

        if op1.__class__.__name__ != op2.__class__.__name__:
            raise ValueError("Combination with non-compatible data structure")

        new_domain = OrderedDict({**op1.domain, **op2.domain})

        # Extend domains if needed
        if len(op1.variables) < len(new_domain):
            op1 = op1.extend_domain(**new_domain)
        if len(op2.variables) < len(new_domain):
            op2 = op2.extend_domain(**new_domain)

        # Set the same variable order
        new_vars = list(new_domain.keys())
        if op1.variables != new_vars:
            op1 = op1.reorder(*new_vars)
        if op2.variables != new_vars:
            op2 = op2.reorder(*new_vars)

        return self.builder(data=operation(op1.data, op2.data), domain=new_domain)


    def multiply(self, op2: NumpyStore) -> NumpyStore:
        return self._generic_combine(op2, lambda x,y : x*y)

    def addition(self, op2: NumpyStore) -> NumpyStore:
        return self._generic_combine(op2, lambda x,y : x+y)

    def divide(self, op2: NumpyStore) -> NumpyStore:
        return self._generic_combine(op2, lambda x,y : x/y)

    def subtract(self, op2: NumpyStore) -> NumpyStore:
        return self._generic_combine(op2, lambda x, y: x - y)


class ListStore(DiscreteStore):

    def __init__(self, data:Union[Iterable, int, float], domain:Dict):

        def builder(*args, **kwargs):
            return ListStore(*args, **kwargs)

        self.builder = builder
        super(self.__class__, self).__init__(data=list(np.ravel(data)), domain=domain)



    @staticmethod
    def _check_consistency(data, domain):
        return len(np.ravel(data)) == np.prod([len(d) for d in domain.values()])

    def _copy_data(self):
        return self._data.copy()


    def sum_all(self) -> float:
        return sum(self.data)

    def restrict(self, **observation) -> ListStore:
        idx = list(dutil.index_iterator(domain, observation))
        new_data = [self._data[i] for i in idx]
        new_dom = {k: d for k, d in self.domain.items() if k not in observation}
        return ListStore(new_data, new_dom)


    def marginalize(self, *vars_remove) -> DiscreteStore:
        space_remove = dutil.assingment_space({v: d for v, d in self.domain.items() if v in vars_remove})
        new_dom = {v: d for v, d in self.domain.items() if v not in vars_remove}
        iterators = [dutil.index_iterator(domain, s) for s in space_remove]
        new_len = np.prod([len(d) for d in new_dom.values()])
        new_data = [sum([self.data[next(it)] for it in iterators]) for i in range(new_len)]
        return ListStore(new_data, new_dom)


    def maxmarginalize(self, *vars_remove) -> DiscreteStore:
        space_remove = dutil.assingment_space({v: d for v, d in self.domain.items() if v in vars_remove})
        new_dom = {v: d for v, d in self.domain.items() if v not in vars_remove}
        iterators = [dutil.index_iterator(domain, s) for s in space_remove]
        new_len = np.prod([len(d) for d in new_dom.values()])
        new_data = [max([self.data[next(it)] for it in iterators]) for i in range(new_len)]
        return ListStore(new_data, new_dom)

    def _generic_combine(self, op2: ListStore, operation:callable) -> ListStore:

        op1 = self

        if op1.__class__.__name__ != op2.__class__.__name__:
            raise ValueError("Combination with non-compatible data structure")

        new_domain = OrderedDict({**op1.domain, **op2.domain})
        new_space = dutil.assingment_space(new_domain)
        new_len = np.prod([len(d) for d in new_domain.values()])
        new_data = [0.0] * len(new_space)

        for k in range(0, len(new_data)):
            i = dutil.index_list(op1.domain, new_space[k])[0]
            j = dutil.index_list(op2.domain, new_space[k])[0]
            new_data[k] = operation(op1.data[i], op2.data[j])

        return ListStore(data=new_data, domain=new_domain)


    def multiply(self, op2: ListStore) -> ListStore:
        return self._generic_combine(op2, lambda x,y : x*y)

    def addition(self, op2: ListStore) -> ListStore:
        return self._generic_combine(op2, lambda x,y : x+y)

    def divide(self, op2: ListStore) -> ListStore:
        return self._generic_combine(op2, lambda x,y : x/y)

    def subtract(self, op2: ListStore) -> ListStore:
        return self._generic_combine(op2, lambda x, y: x - y)



store_dict = {"numpy": NumpyStore, "list":ListStore}

if __name__=="__main__":
    left_domain = dict(A=["a1", "a2"])
    right_domain = dict(B=[0, 1, 3])
    domain = {**left_domain, **right_domain}

    new_var_order = ["B", "A"]
    #complete vars

    new_dom = OrderedDict([(v,domain[v]) for v in new_var_order])

    data = [[0.5, .4, 0.1], [0.3, 0.6, 0.1]]

    vars_remove = ["B"]
    f1 = NumpyStore(data, domain)
    f2 = ListStore(data, domain)

    for f in [f1,f2]:
        print(f.multiply(f.restrict(B=1)).marginalize("A").data)
