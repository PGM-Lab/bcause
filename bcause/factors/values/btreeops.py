from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

from bcause.factors.values.operations import OperationSet

if TYPE_CHECKING:
    from bcause.factors.values.btreestore import BTreeStore


class BTreeStoreOperations(OperationSet):

    @staticmethod
    def marginalize(store: 'BTreeStore', vars_remove: list):
        raise NotImplementedError()
    @staticmethod
    def maxmarginalize(store: 'BTreeStore', vars_remove: list):
        raise NotImplementedError()

    @staticmethod
    def _restrict_btreenode(data, observation):
        raise NotImplementedError()

    @staticmethod
    def _marginalize_btreenode(d, var_to_remove, k, operation: callable):
        raise NotImplementedError()

    @staticmethod
    def _generic_combine(op1: 'BTreeStore', op2: 'BTreeStore', operation: callable) -> 'BTreeStore':

        if op1.__class__.__name__ != op2.__class__.__name__:
            raise ValueError("Combination with non-compatible data structure")

        new_domain = OrderedDict({**op1.domain, **op2.domain})
        new_data = BTreeStoreOperations._combine_btreenode(op1.data, op2.data, operation)
        return op1.builder(domain=new_domain, data=new_data)

    @staticmethod
    def multiply(store: 'BTreeStore', other: 'BTreeStore') -> 'BTreeStore':
        return BTreeStoreOperations._generic_combine(store, other, lambda x, y: x * y)

    @staticmethod
    def addition(store: 'BTreeStore', other: 'BTreeStore') -> 'BTreeStore':
        return BTreeStoreOperations._generic_combine(store, other, lambda x, y: x + y)

    @staticmethod
    def subtract(store: 'BTreeStore', other: 'BTreeStore') -> 'BTreeStore':
        return BTreeStoreOperations._generic_combine(store, other, lambda x, y: x - y)

    @staticmethod
    def divide(store: 'BTreeStore', other: 'BTreeStore') -> 'BTreeStore':
        return BTreeStoreOperations._generic_combine(store, other, lambda x, y: x / y)

    @staticmethod
    def restrict(store : 'BTreeStore', observarion:dict) -> 'BTreeStore':
        raise NotImplementedError("Not implemented")