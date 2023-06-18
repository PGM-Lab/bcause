from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import reduce
from typing import TYPE_CHECKING


import bcause.util.domainutils as dutil


if TYPE_CHECKING:
    from bcause.factors.values import DataStore, ListStore, NumpyStore

class OperationSet(ABC):

    @staticmethod
    @abstractmethod
    def marginalize(store : 'DataStore', vars_remove:list):
        pass

    @staticmethod
    @abstractmethod
    def maxmarginalize(store : 'DataStore', vars_remove:list):
        pass

    @staticmethod
    @abstractmethod
    def multiply(store : 'DataStore', other:'DataStore') -> 'DataStore':
        pass

    @staticmethod
    @abstractmethod
    def addition(store : 'DataStore', other:'DataStore') -> 'DataStore':
        pass

    @staticmethod
    @abstractmethod
    def subtract(store : 'DataStore', other: 'DataStore') -> 'DataStore':
        pass

    @staticmethod
    @abstractmethod
    def divide(store : 'DataStore', other: 'DataStore') -> 'DataStore':
        pass

    @staticmethod
    @abstractmethod
    def restrict(store : 'DataStore', observation:dict) -> 'DataStore':
        pass


class GenericOperations(OperationSet):
    @staticmethod
    def marginalize(store: 'DataStore', vars_remove: list):
        space_remove = dutil.assingment_space({v: d for v, d in store.domain.items() if v in vars_remove})
        new_dom = OrderedDict([(v,d) for v, d in store.domain.items() if v not in vars_remove])
        restricted = [store.restrict(**obs) for obs in space_remove]
        return reduce(GenericOperations.addition, restricted)

    @staticmethod
    def maxmarginalize(store: 'DataStore', vars_remove: list):
        raise NotImplementedError("Operation not implemented")

    @staticmethod
    def _generic_combine(op1:  'DataStore', op2:  'DataStore', operation:callable) ->  'ListStore':

        if op1.__class__.__name__ != op2.__class__.__name__:
            raise ValueError("Combination with non-compatible data structure")

        new_domain = OrderedDict({**op1.domain, **op2.domain})
        new_space = dutil.assingment_space(new_domain)
        res = op1.builder(domain=new_domain)

        for obs in new_space:
            value = operation(op1.get_value(**obs), op2.get_value(**obs))
            res.set_value(value, obs)

        return res


    @staticmethod
    def multiply(store: 'DataStore', other: 'DataStore') -> 'DataStore':
        return GenericOperations._generic_combine(store, other, lambda x, y: x * y)

    @staticmethod
    def addition(store: 'DataStore', other: 'DataStore') -> 'DataStore':
        return GenericOperations._generic_combine(store, other, lambda x, y: x + y)

    @staticmethod
    def subtract(store: 'DataStore', other: 'DataStore') -> 'DataStore':
        return GenericOperations._generic_combine(store, other, lambda x, y: x - y)

    @staticmethod
    def divide(store: 'DataStore', other: 'DataStore') -> 'DataStore':
        return GenericOperations._generic_combine(store, other, lambda x, y: x / y)

    @staticmethod
    def restrict(store : 'DataStore', observarion:dict) -> 'DataStore':
        raise NotImplementedError("Not implemented")


