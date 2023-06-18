import copy
import math
from abc import ABC, abstractmethod
from typing import Hashable, List, OrderedDict, Dict, Union, Iterable

import numpy as np

from bcause.factors.values.btreeops import BTreeStoreOperations
from bcause.factors.values.store import DiscreteStore


class BTreeNode(ABC):

    @property
    def variable(self):
        return self._variable

    @property
    def var_domain(self):
        return self._var_domain

    @property
    @abstractmethod
    def left_states(self):
        pass

    @property
    @abstractmethod
    def right_states(self):
        pass

    @property
    def left_child(self):
        return self._left_child

    @property
    def right_child(self):
        return self._right_child

    @abstractmethod
    def is_on_left(self, state):
        pass

    def is_on_right(self, state):
        return not self.is_on_left(state)

    @staticmethod
    def build(variable, var_domain, left_child, right_child, left_states=None, right_states=None,
              end_left_exclusive=None, consecutive=True):
        if consecutive:
            spoints = []
            if end_left_exclusive is not None:
                if end_left_exclusive < 0: end_left_exclusive = len(var_domain) - end_left_exclusive
                spoints.append(end_left_exclusive)
            if left_states is not None:
                spoints.append(var_domain.index(left_states[-1]) + 1)
            if right_states is not None:
                spoints.append(var_domain.index(right_states[0]))

            # in case of binary variables there is no need of specifying the spliting point
            if len(set(spoints)) == 0 and len(var_domain) == 2:
                spoints = [1]

            if len(set(spoints)) != 1:
                raise ValueError(f"Inconsistent partition of states {left_states}, {right_states}")

            return BTreeNodeConsecutive(variable, var_domain, left_child, right_child, spoints[0])

        else:
            raise NotImplementedError("build bt node not implemented for non consecutive")


class BTreeNodeConsecutive(BTreeNode):

    def __init__(self, variable: Hashable, var_domain: List, left_child, right_child, end_left_exclusive: int):
        if end_left_exclusive < 1 or end_left_exclusive >= len(var_domain):
            raise ValueError(f"Spliting point end_left_exclusive must be in interval [1,len(var_domain)) ")
        self._variable = variable
        self._var_domain = var_domain
        self._end_left_exclusive = end_left_exclusive
        self._left_child = left_child
        self._right_child = right_child

    @property
    def left_states(self):
        return self._var_domain[:self._end_left_exclusive]

    @property
    def right_states(self):
        return self._var_domain[self._end_left_exclusive:]

    def is_on_left(self, state):
        return self._var_domain.index(state) < self._end_left_exclusive

    def summary(self, n=0):
        s = f"<BTNode({self.variable})[...{self.left_states[-1]}|{self.right_states[0]}...]"
        s += "\n"
        s += "  "*n
        s += f"left: <{self.left_child.summary(n+1) if isinstance(self.left_child, BTreeNode) else self.left_child}>,"
        s += "\n"
        s += "  "*n
        s += f"right: <{self.right_child.summary(n+1) if isinstance(self.right_child, BTreeNode) else self.right_child}>>"
        return s


class BTreeStore(DiscreteStore):

    def __init__(self, domain: Dict, data: Union[Iterable, int, float, dict]=None):
        #defualt data
        if data is None:
            data = np.zeros(np.prod([len(d) for d in domain.values()]))

        if len(domain)>0 and not isinstance(data, BTreeNode):
            from bcause.factors.values import NumpyStore
            data = self._build_from_table(NumpyStore(domain, data))

        def builder(**kwargs):
            return BTreeStore(**kwargs)

        self.builder = builder
        self.set_operationSet(BTreeStoreOperations)
        super(self.__class__, self).__init__(domain=domain, data=data)

    @staticmethod
    def _build_from_table(table):

        if table.all_equal():
            return table.values_list[0]

        v,tl,tr, table_left, table_right =  BTreeStore._best_split_point(table)

        tree_left = BTreeStore._build_from_table(table_left)
        tree_right = BTreeStore._build_from_table(table_right)


        return BTreeNode.build(v, table.domain[v], left_child=tree_left, right_child=tree_right, left_states=tl)

    @staticmethod
    def _best_split_point(table):
        info_max = float("-Inf")
        best_var = None
        best_left_states = None
        best_left_table = None
        best_right_table = None
        for v in table.variables:
            sum_At = table.sum_all()
            info_At = sum_At * math.log(len(table.domain[v]) / sum_At)

            for s in range(1, len(table.domain[v])):
                tl = table.domain[v][:s]
                tr = table.domain[v][s:]

                left_table = table.restrict(**{v: tl})
                right_table = table.restrict(**{v: tr})

                sum_left = left_table.sum_all()
                sum_right = right_table.sum_all()
                info = info_At
                if sum_left>0:
                    info += sum_left * math.log(sum_left/len(tl))
                if sum_right>0:
                    info += sum_right * math.log(sum_right/len(tr))
                if info_max < info:
                    best_var, best_left_states, best_right_states = v, tl, tr
                    best_left_table, best_right_table = left_table, right_table

        return best_var, best_left_states, best_right_states, best_left_table, best_right_table

    @staticmethod
    def _check_consistency(data, domain):
        return True

    def _copy_data(self):
        return copy.deepcopy(self.data)

    def set_value(self, value, observation):
        raise NotImplementedError("method not implemented")

    def get_value(self, **observation):
        raise NotImplementedError("method not implemented")



if __name__ == "__main__":

    variable = "U"
    var_domain = ["u1", "u2", "u3"]

    # different ways of building a node
    n1 = BTreeNode.build(variable, var_domain, 0.2, 0.4, left_states=["u1"])
    n2 = BTreeNode.build(variable, var_domain, 0.2, 0.4, right_states=["u2", "u3"])
    n3 = BTreeNode.build(variable, var_domain, 0.4, 0.3, end_left_exclusive=1)

    # a non terminal node
    nested_nodes = BTreeNode.build("X", ["x1", "x2"], 0.33, n1)

    print(nested_nodes)


    domain = dict(A=["a1", "a2"], B=["b1", "b2", "b3","b4"])
    new_var_order = ["B", "A"]
    #complete vars
    new_dom = dict([(v,domain[v]) for v in new_var_order])
    data = [[0.2, .2, 0.5, 0.1], [0.2, 0.2, 0.6,0.0]]

    bt = BTreeStore(domain, data)
    print(bt.data.summary())


