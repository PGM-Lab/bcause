from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np

from bcause.factors.values.operations import OperationSet

if TYPE_CHECKING:
    from bcause.factors.values import DataStore, NumpyStore


class NumpyStoreOperations(OperationSet):

    @staticmethod
    def marginalize(store: 'NumpyStore', vars_remove: list) -> 'NumpyStore':
        idx_vars = tuple(store.get_var_index(v) for v in vars_remove)
        new_data = np.sum(store.data, axis=idx_vars)
        new_dom = OrderedDict([(v,d) for v, d in store.domain.items() if v not in vars_remove])
        return store.builder(data=new_data, domain=new_dom)

    @staticmethod
    def maxmarginalize(store: 'NumpyStore', vars_remove: list) -> 'NumpyStore':
        idx_vars = tuple(store.get_var_index(v) for v in vars_remove)
        new_data = np.max(store.data, axis=idx_vars)
        new_dom = OrderedDict([(v,d) for v, d in store.domain.items() if v not in vars_remove])
        return store.builder(data=new_data, domain=new_dom)

    @staticmethod
    def _generic_combine(op1:'NumpyStore', op2: 'NumpyStore', operation:callable) -> 'NumpyStore':

        if op1.__class__.__name__ != op2.__class__.__name__:
            raise ValueError("Combination with non-compatible data structure")

        new_domain = OrderedDict({**op1.domain, **op2.domain})

        # Extend domains if needed
        if len(op1.variables) < len(new_domain):
            op1 = NumpyStoreOperations._extend_domain(op1, **new_domain)
        if len(op2.variables) < len(new_domain):
            op2 = NumpyStoreOperations._extend_domain(op2, **new_domain)

        # Set the same variable order
        new_vars = list(new_domain.keys())
        if op1.variables != new_vars:
            op1 = NumpyStoreOperations._reorder(op1, *new_vars)
        if op2.variables != new_vars:
            op2 = NumpyStoreOperations._reorder(op2, *new_vars)

        return op1.builder(data=operation(op1.data, op2.data), domain=new_domain)


    @staticmethod
    def _reorder(store, *new_var_order) -> 'NumpyStore':
        if len(store.variables)<2:
            return store

        # filter variables and add remaining variables
        new_var_order = [v for v in new_var_order if v in store.variables]
        new_var_order += [v for v in store.variables if v not in new_var_order]

        idx_var_order = [new_var_order.index(v) for v in store.variables]

        # set the new order in the values
        new_data = np.moveaxis(store.data, range(np.ndim(store.data)), idx_var_order)
        new_dom = OrderedDict([(v, store.domain[v]) for v in new_var_order])

        # create new object
        return store.builder(data = new_data, domain = new_dom)

    @staticmethod
    def _extend_domain(store, **extra_dom) -> 'NumpyStore':

        extra_dom = OrderedDict({v:c for v,c in extra_dom.items() if v not in store.variables})
        add_card = tuple([len(d) for d in extra_dom.values()])

        new_data = np.reshape(np.repeat(store.data, np.prod(add_card)), np.shape(store.data) + add_card)
        new_dom = OrderedDict({**store.domain, **extra_dom})

        return store.builder(data=new_data, domain=new_dom)



    @staticmethod
    def multiply(store: 'NumpyStore', other: 'NumpyStore') -> 'NumpyStore':
        return NumpyStoreOperations._generic_combine(store, other, lambda x, y: x * y)

    @staticmethod
    def addition(store: 'NumpyStore', other: 'NumpyStore') -> 'NumpyStore':
        return NumpyStoreOperations._generic_combine(store, other, lambda x, y: x + y)

    @staticmethod
    def subtract(store: 'NumpyStore', other: 'NumpyStore') -> 'NumpyStore':
        return NumpyStoreOperations._generic_combine(store, other, lambda x, y: x - y)

    @staticmethod
    def divide(store: 'NumpyStore', other: 'NumpyStore') -> 'NumpyStore':
        return NumpyStoreOperations._generic_combine(store, other, lambda x, y: np.nan_to_num(x / y))

    @staticmethod
    def restrict(store : 'NumpyStore', observation:dict) -> 'NumpyStore':
        items = []
        new_dom = OrderedDict()
        for v in store.variables:
            if v in observation.keys():
                obs_v = observation[v]
                if isinstance(obs_v, list) and len(obs_v)==1:
                    obs_v = obs_v[0]

                if not isinstance(obs_v, list):
                    idx = np.where(np.array(store.domain[v]) == obs_v)[0][0]
                    items.append(idx)
                else:
                    idx = [np.where(np.array(store.domain[v]) == x)[0][0] for x in obs_v]
                    items.append(idx)
                    new_dom[v] = [x for x in store.domain[v] if x in obs_v]
            else:
                items.append(slice(None))
                new_dom[v] = store.domain[v]
        new_data = store._data[tuple(items)].copy()
        return store.builder(domain=new_dom, data=new_data)

