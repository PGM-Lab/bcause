from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

from PIL.ImageChops import multiply
from sympy.codegen.cnodes import restrict
import numpy as np
from functools import reduce

from bcause.factors.values.operations import OperationSet

from bcause.factors.values.btreestore import BTreeStore, BTreeNode, BTreeNodeConsecutive, BTreeNodeNonConsecutive


class BTreeStoreOperations(OperationSet):

    @staticmethod
    def marginalize(store: 'BTreeStore', vars_remove: list):
        new_data = store.data
        for v in vars_remove:
            new_data = BTreeStoreOperations._marginalize_btreenode(new_data, v, len(store.domain[v]), lambda x, y: x + y)
        new_dom = OrderedDict([(v, d) for v, d in store.domain.items() if v not in vars_remove])
        return store.builder(domain=new_dom, data=new_data)

    @staticmethod
    def maxmarginalize(store: 'BTreeStore', vars_remove: list):
        new_data = store.data
        for v in vars_remove:
            new_data = BTreeStoreOperations._marginalize_btreenode(new_data, v, 1, lambda x, y: max(x, y))
        new_dom = OrderedDict([(v, d) for v, d in store.domain.items() if v not in vars_remove])
        return store.builder(domain=new_dom, data=new_data)

    # @staticmethod
    # def restrict_btreenode(data, observation):
    #     if not isinstance(data, BTreeNode) or len(observation) == 0:
    #         out = data
    #     elif data.variable not in observation:
    #         new_left = BTreeStoreOperations.restrict_btreenode(data.left_child, observation)
    #         new_right = BTreeStoreOperations.restrict_btreenode(data.right_child, observation)
    #
    #         out = BTreeNode.build(variable = data.variable,
    #                               var_domain = data.var_domain,
    #                               left_child=new_left,
    #                               right_child=new_right,
    #                               left_states=data.left_states,
    #                               right_states=data.right_states,
    #                               consecutive=isinstance(data, BTreeNodeConsecutive))
    #
    #     else:
    #         other_obs = {v: s for v, s in observation.items() if v != data.variable}
    #         state = observation[data.variable]
    #         if state in data.left_states:
    #             ch = data.left_child
    #         elif state in data.right_states:
    #             ch = data.right_child
    #         else:
    #             raise ValueError(f"State {state} not in variable {data.variable} domain")
    #         out = BTreeStoreOperations.restrict_btreenode(ch, other_obs)
    #     return out

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
            obs_val = observation[data.variable]
            obs_set = set(obs_val) if isinstance(obs_val, list) else {obs_val}

            SlX = set(data.left_states) & obs_set
            SrX = set(data.right_states) & obs_set

            new_var_domain = set(data.var_domain) & obs_set

            if not SlX and not SrX:
                raise ValueError(f"State {observation[data.variable]} not in variable {data.variable} domain")
            elif not SlX:
                new_obs = observation.copy()
                new_obs[data.variable] = list(SrX)
                out = BTreeStoreOperations.restrict_btreenode(data.right_child, new_obs)
            elif not SrX:
                new_obs = observation.copy()
                new_obs[data.variable] = list(SlX)
                out = BTreeStoreOperations.restrict_btreenode(data.left_child, new_obs)
            else:

                left_obs = dict(observation);left_obs[data.variable] = list(SlX)
                right_obs = dict(observation);right_obs[data.variable] = list(SrX)
                new_left = BTreeStoreOperations.restrict_btreenode(data.left_child, left_obs)
                new_right = BTreeStoreOperations.restrict_btreenode(data.right_child, right_obs)
                out = BTreeNode.build(variable = data.variable,
                                      var_domain = list(new_var_domain),
                                      left_child=new_left,
                                      right_child=new_right,
                                      left_states=list(SlX) if isinstance(data, BTreeNodeConsecutive) else SlX,
                                      right_states=list(SrX) if isinstance(data, BTreeNodeConsecutive) else SrX,
                                      consecutive=isinstance(data, BTreeNodeConsecutive))

        return out

    @staticmethod
    def _marginalize_btreenode(d, var_to_remove, k, operation: callable):

        if not isinstance(d, BTreeNode):
            out = d * k
        else:
            if d.variable == var_to_remove:
                new_left = BTreeStoreOperations._marginalize_btreenode(d.left_child, var_to_remove, len(d.left_states), operation)
                new_right = BTreeStoreOperations._marginalize_btreenode(d.right_child, var_to_remove, len(d.right_states), operation)
                out = BTreeStoreOperations.combine_btreenode(new_left, new_right, operation)
            else:
                new_left = BTreeStoreOperations._marginalize_btreenode(d.left_child, var_to_remove, k, operation)
                new_right = BTreeStoreOperations._marginalize_btreenode(d.right_child, var_to_remove, k, operation)
                out = BTreeNode.build(variable=d.variable,
                                      var_domain=d.var_domain,
                                      left_child=new_left,
                                      right_child=new_right,
                                      left_states=d.left_states,
                                      right_states=d.right_states,
                                      consecutive=isinstance(d, BTreeNodeConsecutive))

        return out
    @staticmethod
    def combine_btreenode(d1, d2, operation):


        if "multiply" in operation.__qualname__:
            if d1==0 or d2==0:
                return 0
        elif "addition" in operation.__qualname__:
            if d1 == 0: return d2
            if d2 == 0: return d1
        elif "divide" in operation.__qualname__:
            if d1 == 0 : return 0
            if d2 == 0: return 0

        if not isinstance(d1, BTreeNode):
            if not isinstance(d2, BTreeNode):
                out = operation(d1, d2)
            else:
                new_var = d2.variable
                new_ch_left = BTreeStoreOperations.combine_btreenode(d1, d2.left_child, operation)
                new_ch_right = BTreeStoreOperations.combine_btreenode(d1, d2.right_child, operation)
                out = BTreeNode.build(variable=new_var,
                                      var_domain=d2.var_domain,
                                      left_child=new_ch_left,
                                      right_child=new_ch_right,
                                      left_states=d2.left_states,
                                      right_states=d2.right_states,
                                      consecutive=isinstance(d2, BTreeNodeConsecutive))
        else:
            new_var = d1.variable
            obs1 = list(d1.left_states) if len(d1.left_states) > 1 else list(d1.left_states)[0]
            obs2 = list(d1.right_states) if len(d1.right_states) > 1 else list(d1.right_states)[0]
            restrict_d2_left = BTreeStoreOperations.restrict_btreenode(d2, {new_var: obs1})
            restrict_d2_right = BTreeStoreOperations.restrict_btreenode(d2, {new_var: obs2})

            new_ch_left = BTreeStoreOperations.combine_btreenode(d1.left_child, restrict_d2_left, operation)
            new_ch_right = BTreeStoreOperations.combine_btreenode(d1.right_child, restrict_d2_right, operation)
            out = BTreeNode.build(variable=new_var,
                                  var_domain=d1.var_domain,
                                  left_child=new_ch_left,
                                  right_child=new_ch_right,
                                  left_states=d1.left_states,
                                  right_states=d1.right_states,
                                  consecutive=isinstance(d1, BTreeNodeConsecutive))
        return out


    @staticmethod
    def _generic_combine(op1: 'BTreeStore', op2: 'BTreeStore',operation: callable) -> 'BTreeStore':

        if op1.__class__.__name__ != op2.__class__.__name__:
            raise ValueError("Combination with non-compatible data structure")

        new_domain = OrderedDict({**op1.domain, **op2.domain})
        new_data = BTreeStoreOperations.combine_btreenode(op1.data, op2.data, operation)
        return op1.builder(domain=new_domain, data=new_data)

    @staticmethod
    def multiply(store: 'BTreeStore', other: 'BTreeStore') -> 'BTreeStore':
        return BTreeStoreOperations._generic_combine(store, other,lambda x, y: x * y)

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
    def restrict(store : 'BTreeStore', observation:dict) -> 'BTreeStore':
        new_data = BTreeStoreOperations.restrict_btreenode(store.data, observation)
        new_dom = OrderedDict([(v, d) for v, d in store.domain.items() if v not in observation])
        return store.builder(domain=new_dom, data=new_data)

    @staticmethod
    def SE_operation(op1: 'BTreeStore', op2: 'BTreeStore') -> 'BTreeStore':

        if op1.__class__.__name__ != op2.__class__.__name__:
            raise ValueError("Combination with non-compatible data structure")

        new_domain = OrderedDict({**op1.domain, **op2.domain})
        new_data = BTreeStoreOperations.multiply_SE(op1.data, op2.data)
        return op1.builder(domain=new_domain, data=new_data)

    @staticmethod
    def multiply_SE(d1, d2):
        if not isinstance(d1, BTreeNode):
            if not isinstance(d2, BTreeNode):
                out = d1*d2
            else:
                new_var = d2.variable
                new_ch_left = BTreeStoreOperations.multiply_SE(d1, d2.left_child)
                new_ch_right = BTreeStoreOperations.multiply_SE(d1, d2.right_child)
                out = BTreeNode.build(variable=new_var,
                                      var_domain=d2.var_domain,
                                      left_child=new_ch_left,
                                      right_child=new_ch_right,
                                      left_states=d2.left_states,
                                      right_states=d2.right_states,
                                      consecutive=isinstance(d2, BTreeNodeConsecutive))
        else:
            if isinstance(d1, BTreeNodeNonConsecutive):
                if not isinstance(d2, BTreeNodeNonConsecutive):
                    new_var = d2.variable
                    new_lb = d2.left_states
                    new_rb = d2.right_states

                    new_ch_left = BTreeStoreOperations.multiply_SE(d1, d2.left_child)
                    new_ch_right = BTreeStoreOperations.multiply_SE(d1, d2.right_child)
                    out = BTreeNode.build(variable=new_var,
                                          var_domain=d2.var_domain,
                                          left_child=new_ch_left,
                                          right_child=new_ch_right,
                                          left_states=new_lb,
                                          right_states=new_rb,
                                          consecutive=isinstance(d2, BTreeNodeConsecutive))
                else:
                    compatible_states = set(d1.right_states) & set(d2.right_states)
                    non_compatible_states = set(d1.var_domain) - set(compatible_states)
                    out = BTreeNode.build(variable=d1.variable,
                                          var_domain=d1.var_domain,
                                          left_child=0,
                                          right_child=1,
                                          left_states=non_compatible_states,
                                          right_states=compatible_states,
                                          consecutive=False)
            else:
                new_var = d1.variable
                obs1 = list(d1.left_states) if len(d1.left_states) > 1 else list(d1.left_states)[0]
                obs2 = list(d1.right_states) if len(d1.right_states) > 1 else list(d1.right_states)[0]
                restrict_d2_left = BTreeStoreOperations.restrict_btreenode(d2, {new_var: obs1})
                restrict_d2_right = BTreeStoreOperations.restrict_btreenode(d2, {new_var: obs2})

                new_ch_left = BTreeStoreOperations.multiply_SE(d1.left_child, restrict_d2_left)
                new_ch_right = BTreeStoreOperations.multiply_SE(d1.right_child, restrict_d2_right)
                out = BTreeNode.build(variable=new_var,
                                      var_domain=d1.var_domain,
                                      left_child=new_ch_left,
                                      right_child=new_ch_right,
                                      left_states=d1.left_states,
                                      right_states=d1.right_states,
                                      consecutive=isinstance(d1, BTreeNodeConsecutive))
        return out

    @staticmethod
    def multiply_exogenous(d1,d2):
        if not isinstance(d1, BTreeNode):
            out = d1
        else:
            if d1.variable != d2.variable:
                left = BTreeStoreOperations.multiply_exogenous(d1.left_child,d2)
                right = BTreeStoreOperations.multiply_exogenous(d1.right_child,d2)
                out = BTreeNode.build(variable=d1.variable,
                                        var_domain=d1.var_domain,
                                        left_child=left,
                                        right_child=right,
                                        left_states=d1.left_states,
                                        right_states=d1.right_states,
                                        consecutive=isinstance(d1, BTreeNodeConsecutive))

            else:
                #new_right = np.sum([BTreeStoreOperations.restrict_btreenode(d2, {d2.variable: s}) for s in d1.right_states])
                new_right = BTreeStoreOperations.restrict_btreenode(d2, {d2.variable: list(d1.right_states)}) if len(d1.right_states)>1 else BTreeStoreOperations.restrict_btreenode(d2, {d2.variable: list(d1.right_states)[0]})


                out = BTreeNode.build(variable=d1.variable,
                                        var_domain=d1.var_domain,
                                        left_child=d1.left_child,
                                        right_child=new_right,
                                        left_states=d1.left_states,
                                        right_states=d1.right_states,
                                        consecutive=isinstance(d1, BTreeNodeConsecutive))

        return out

    @staticmethod
    def marginalize_endogenous(data, exovar):
        def rec(n):
            if not isinstance(n, BTreeNode):
                pass
            elif n.variable != exovar:
                rec(n.left_child)
                rec(n.right_child)
            else:
                subtrees.append(n)
            return subtrees
        subtrees = []
        subtrees = rec(data)
        return reduce(lambda a,b : BTreeStoreOperations.combine_btreenode(a,b,lambda x, y: x + y), subtrees)


