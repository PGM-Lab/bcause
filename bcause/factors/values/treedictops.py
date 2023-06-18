from __future__ import annotations

from collections import OrderedDict
from functools import reduce
from typing import TYPE_CHECKING

from bcause.factors.values.operations import OperationSet
from bcause.util.treesutil import build_default_tree, treeNode

if TYPE_CHECKING:
    from bcause.factors.values.treedictstore import TreeDictStore


class TreeDictStoreOperations(OperationSet):

    @staticmethod
    def marginalize(store: 'TreeDictStore', vars_remove: list):
        new_data = store.data
        for v in vars_remove:
            new_data = TreeDictStoreOperations._marginalize_dict(new_data, v, len(store.domain[v]), lambda x, y: x + y)
        new_dom = OrderedDict([(v, d) for v, d in store.domain.items() if v not in vars_remove])
        return store.builder(domain=new_dom, data=new_data)

    @staticmethod
    def maxmarginalize(store: 'TreeDictStore', vars_remove: list):
        new_data = store.data
        for v in vars_remove:
            new_data = TreeDictStoreOperations._marginalize_dict(new_data, v, 1, lambda x, y: max(x, y))
        new_dom = OrderedDict([(v, d) for v, d in store.domain.items() if v not in vars_remove])
        return store.builder(domain=new_dom, data=new_data)
    @staticmethod
    def _combine_dict(d1, d2, operation):
        if not isinstance(d1, dict):
            if not isinstance(d2, dict):
                out = operation(d1, d2)
            else:
                new_var = d2["var"]
                new_ch = {state: TreeDictStoreOperations._combine_dict(d1, ch, operation) for state, ch in d2["children"].items()}
                out = treeNode(new_var, new_ch)
        else:
            new_var = d1["var"]
            new_ch = {state: TreeDictStoreOperations._combine_dict(ch, TreeDictStoreOperations._restrict_dict(d2,
                {new_var: state}), operation) for
                      state, ch in d1["children"].items()}
            out = treeNode(new_var, new_ch)

        return out

    @staticmethod
    def _restrict_dict(data, observation):
        if not isinstance(data, dict) or len(observation) == 0:
            out = data
        elif data["var"] not in observation:
            new_ch = dict()
            for state, ch in data["children"].items():
                new_ch[state] = ch if not isinstance(ch, dict) else TreeDictStoreOperations._restrict_dict(ch,
                                                                                                           observation)
            out = treeNode(data["var"], new_ch)
        else:
            other_obs = {v: s for v, s in observation.items() if v != data["var"]}
            out = TreeDictStoreOperations._restrict_dict(data["children"][observation[data["var"]]], other_obs)
        return out

    @staticmethod
    def _marginalize_dict(d, var_to_remove, k, operation: callable):

        if not isinstance(d, dict):
            out = d * k
        else:
            if d["var"] == var_to_remove:
                out = reduce(lambda ch1, ch2: TreeDictStoreOperations._combine_dict(ch1, ch2, operation),
                             [ch for ch in d["children"].values()])
            else:
                new_var = d["var"]
                new_ch = {state: TreeDictStoreOperations._marginalize_dict(ch, var_to_remove, k, operation) for
                          state, ch in d["children"].items()}
                out = treeNode(new_var, new_ch)

        return out

    @staticmethod
    def _generic_combine(op1: 'TreeDictStore', op2: 'TreeDictStore', operation: callable) -> 'TreeDictStore':

        if op1.__class__.__name__ != op2.__class__.__name__:
            raise ValueError("Combination with non-compatible data structure")

        new_domain = OrderedDict({**op1.domain, **op2.domain})
        new_data = TreeDictStoreOperations._combine_dict(op1.data, op2.data, operation)
        return op1.builder(domain=new_domain, data=new_data)

    @staticmethod
    def multiply(store: 'TreeDictStore', other: 'TreeDictStore') -> 'TreeDictStore':
        return TreeDictStoreOperations._generic_combine(store, other, lambda x, y: x * y)

    @staticmethod
    def addition(store: 'TreeDictStore', other: 'TreeDictStore') -> 'TreeDictStore':
        return TreeDictStoreOperations._generic_combine(store, other, lambda x, y: x + y)

    @staticmethod
    def subtract(store: 'TreeDictStore', other: 'TreeDictStore') -> 'TreeDictStore':
        return TreeDictStoreOperations._generic_combine(store, other, lambda x, y: x - y)

    @staticmethod
    def divide(store: 'TreeDictStore', other: 'TreeDictStore') -> 'TreeDictStore':
        return TreeDictStoreOperations._generic_combine(store, other, lambda x, y: x / y)


    @staticmethod
    def restrict(store : 'TreeDictStore', observation:dict) -> 'TreeDictStore':
        if any([type(v)==list for v in observation.values()]):
            raise NotImplementedError("Restriction to an extended configuration not implemented")
        new_data = TreeDictStoreOperations._restrict_dict(store.data, observation)
        new_dom = OrderedDict([(k, d) for k, d in store.domain.items() if k not in observation])
        return store.builder(data = new_data, domain = new_dom)