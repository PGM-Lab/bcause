from __future__ import annotations

import numpy as np

from collections import OrderedDict
from typing import Dict, Iterable, Union

import bcause.util.domainutils as dutil

from bcause.factors.values.listops import  ListStoreOperations
from bcause.factors.values.store import DiscreteStore


class ListStore(DiscreteStore):

    def __init__(self, domain: Dict, data: Union[Iterable, int, float]=None):

        # default data
        if data is None:
            data = [0.0] * int(np.prod([len(d) for d in domain.values()]))

        def builder(**kwargs):
            return ListStore(**kwargs)

        self.builder = builder
        self.set_operationSet(ListStoreOperations)
        #self.set_operationSet(GenericOperations)

        super(self.__class__, self).__init__(domain=domain, data=list(np.ravel(data)))

    @staticmethod
    def _check_consistency(data, domain):
        return len(np.ravel(data)) == np.prod([len(d) for d in domain.values()])

    def _copy_data(self):
        return self._data.copy()

    def set_value(self, value, observation):
        idx = list(dutil.index_iterator(self.domain, observation))[0]
        self._data[idx] = value

    def get_value(self, **observation) -> 'ListStore':
        return self.restrict(**observation).data[0]



