from __future__ import annotations


import numpy as np
from collections import OrderedDict
from typing import Dict, Iterable, Union

from bcause.factors.values.store import DiscreteStore
from bcause.factors.values.operations import  GenericOperations

import bcause.util.domainutils as dutil
from bcause.factors.values.numpyops import NumpyStoreOperations



class NumpyStore(DiscreteStore):
    def __init__(self, domain: Dict, data: Union[Iterable, int, float] = None):

        # default data
        if data is None:
            data = np.zeros([len(d) for d in domain.values()])

        def builder(*args, **kwargs):
            return NumpyStore(**kwargs)

        self.builder = builder
        self.set_operationSet(NumpyStoreOperations)


        super(self.__class__, self).__init__(domain=domain, data=np.array(data))


    @staticmethod
    def _check_consistency(data, domain):
        return (np.ndim(data) == len(domain)) \
               and (list(np.shape(data)) == [len(d) for v,d in domain.items()])

    def _copy_data(self):
        return self._data.copy()

    def set_value(self, value, observation):
        items = []
        for v in self.variables:
            idx = np.where(np.array(self.domain[v]) == observation[v])[0][0]
            items.append(idx)
        self._data[tuple(items)] = value

    def get_value(self, **observation):
        return np.atleast_1d(self.restrict(**observation).data)[0]


class Numpy1DStore(DiscreteStore):

    def __init__(self, domain: Dict, data: Union[Iterable, int, float]=None):

        #defualt data
        if data is None:
            data = np.zeros(int(np.prod([len(d) for d in domain.values()])))

        def builder(**kwargs):
            return Numpy1DStore(**kwargs)

        self.builder = builder
        #self.set_operationSet(ListStoreOperations)
        self.set_operationSet(GenericOperations)
        super(self.__class__, self).__init__(domain=domain, data=np.ravel(data))

    @staticmethod
    def _check_consistency(data, domain):
        return len(np.ravel(data)) == np.prod([len(d) for d in domain.values()])

    def _copy_data(self):
        return self._data.copy()

    def set_value(self, value, observation):
        idx = list(dutil.index_iterator(self.domain, observation))[0]
        self._data[idx] = value

    def get_value(self, **observation):
        return self.restrict(**observation).data[0]

    def restrict(self, **observation) -> Numpy1DStore:
        if any([type(v)==list for v in observation.values()]):
            raise NotImplementedError("Restriction to an extended configuration not implemented")
        idx = list(dutil.index_iterator(self.domain, observation))
        new_data = [self._data[i] for i in idx]
        new_dom = OrderedDict([(k, d) for k, d in self.domain.items() if k not in observation])
        return self.builder(data = new_data, domain = new_dom)



