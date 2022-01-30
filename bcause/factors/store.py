from abc import ABC, abstractmethod
from typing import Dict, Iterable, Union

import numpy as np

'''
todo: use term store? 
include full domains

'''

class DataStore(ABC):
    pass

class DiscreteStore(DataStore):

    def __init__(self, data:Union[Iterable, int, float], domain:Dict):

        if not self._check_consistency(data, domain):
            raise ValueError("Cardinality Error")

        self._data = data
        self._domain = domain

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
    def restrict(self, **observations):
        pass

    @abstractmethod
    def marginalize(self, operation):
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
    def cardinality(self):
        return np.shape(self.data)

    @staticmethod
    def _check_consistency(data, domain):
        return (np.ndim(data) == len(domain)) \
               and (list(np.shape(data)) == [len(d) for v,d in domain.items()])

    def _copy_data(self):
        return self._data

    def sum_all(self):
        return sum(self._data.flatten())


    def restrict(self, **observation):
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


store_dict = {"numpy": NumpyStore}


