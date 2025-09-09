from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

from bcause.factors.values.operations import OperationSet

from bcause.factors.values.btreestore import BTreeStore, BTreeNode, BTreeNodeConsecutive, BTreeNodeNonConsecutive


class BTreeStoreOperations(OperationSet):

    @staticmethod
    def marginalize(store: 'BTreeStore', vars_remove: list):
        raise NotImplementedError()
    @staticmethod
    def maxmarginalize(store: 'BTreeStore', vars_remove: list):
        raise NotImplementedError()

    @staticmethod
    def restrict_btreenode(data, observation):
        if not isinstance(data, BTreeNode) or len(observation) == 0:
            out = data
        elif data.variable not in observation:
            new_left = BTreeStoreOperations.restrict_btreenode(data.left_child, observation)
            new_right = BTreeStoreOperations.restrict_btreenode(data.right_child, observation)

            out = BTreeNode.build(variable = data.variable,
                                  var_domain = data.var_domain,
                                  left_child=new_left,
                                  right_child=new_right,
                                  left_states=data.left_states,
                                  right_states=data.right_states,
                                  consecutive=isinstance(data, BTreeNodeConsecutive))

        else:
            other_obs = {v: s for v, s in observation.items() if v != data.variable}
            state = observation[data.variable]
            if state in data.left_states:
                ch = data.left_child
            if state in data.right_states:
                ch = data.right_child
            out = BTreeStoreOperations.restrict_btreenode(ch, other_obs)
        return out

    @staticmethod
    def _marginalize_btreenode(d, var_to_remove, k, operation: callable):
        raise NotImplementedError()
    @staticmethod
    def _combine_btreenode(d1, d2, operation):
        pass


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