from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np

import bcause.util.domainutils as dutil
from bcause.factors.values.operations import OperationSet


if TYPE_CHECKING:
    from bcause.factors.values import ListStore


class  ListStoreOperations(OperationSet):
    @staticmethod
    def marginalize(store: 'ListStore', vars_remove: list) -> 'ListStore':
        space_remove = dutil.assingment_space({v: d for v, d in store.domain.items() if v in vars_remove})
        new_dom = OrderedDict([(v,d) for v, d in store.domain.items() if v not in vars_remove])
        iterators = [dutil.index_iterator(store.domain, s) for s in space_remove]
        new_len = int(np.prod([len(d) for d in new_dom.values()]))
        new_data = [sum([store.data[next(it)] for it in iterators]) for i in range(new_len)]
        return store.builder(data=new_data, domain=new_dom)

    @staticmethod
    def maxmarginalize(store: 'ListStore', vars_remove: list) -> 'ListStore':
        space_remove = dutil.assingment_space({v: d for v, d in store.domain.items() if v in vars_remove})
        new_dom = OrderedDict([(v,d) for v, d in store.domain.items() if v not in vars_remove])
        iterators = [dutil.index_iterator(store.domain, s) for s in space_remove]
        new_len = int(np.prod([len(d) for d in new_dom.values()]))
        new_data = [max([store.data[next(it)] for it in iterators]) for i in range(new_len)]
        return store.builder(data=new_data, domain=new_dom)


    def _generic_combine(op1:  'ListStore', op2:  'ListStore', operation:callable) ->  'ListStore':

        if op1.__class__.__name__ != op2.__class__.__name__:
            raise ValueError("Combination with non-compatible data structure")

        new_domain = OrderedDict({**op1.domain, **op2.domain})
        new_space = dutil.assingment_space(new_domain)
        new_data = [0.0] * len(new_space)

        for k in range(0, len(new_data)):
            i = dutil.index_list(op1.domain, new_space[k])[0]
            j = dutil.index_list(op2.domain, new_space[k])[0]
            new_data[k] = operation(op1.data[i], op2.data[j])

        return op1.builder(data=new_data, domain=new_domain)

    @staticmethod
    def multiply(store: 'ListStore', other: 'ListStore') -> 'ListStore':
        return  ListStoreOperations._generic_combine(store, other, lambda x, y: x * y)

    @staticmethod
    def addition(store: 'ListStore', other: 'ListStore') -> 'ListStore':
        return  ListStoreOperations._generic_combine(store, other, lambda x, y: x + y)

    @staticmethod
    def subtract(store: 'ListStore', other: 'ListStore') -> 'ListStore':
        return  ListStoreOperations._generic_combine(store, other, lambda x, y: x - y)

    @staticmethod
    def divide(store: 'ListStore', other: 'ListStore') -> 'ListStore':
        return  ListStoreOperations._generic_combine(store, other, lambda x, y: x / y)

    @staticmethod
    def restrict(store : 'ListStore', observation:dict) -> 'ListStore':
        if any([type(v)==list for v in observation.values()]):
            raise NotImplementedError("Restriction to an extended configuration not implemented")
        idx = list(dutil.index_iterator(store.domain, observation))
        new_data = [store._data[i] for i in idx]
        new_dom = OrderedDict([(k, d) for k, d in store.domain.items() if k not in observation])
        return store.builder(data= new_data, domain = new_dom)
