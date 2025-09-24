import copy
import math
from abc import ABC, abstractmethod
from typing import Hashable, List, OrderedDict, Dict, Union, Iterable

import numpy as np
from pandas.core.computation.expr import intersection
from torch.distributions.constraints import multinomial

# from bcause.factors.multinomial import btree_equation
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

            ls = set(left_states) if left_states is not None else set()
            rs = set(right_states) if right_states is not None else set()

            if var_domain is None:
                var_domain = list(ls.union(rs))

            # In case of binary variables there is no need of specifying the states
            if not ls and not rs and len(var_domain) == 2:
                ls = {var_domain[0]}
                rs = {var_domain[1]}
            if not ls:
                ls = set(var_domain) - rs
            if not rs:
                rs = set(var_domain) - ls

            return BTreeNodeNonConsecutive(variable, left_child, right_child, ls, rs, var_domain)


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

    def to_BtreeNodeNonConsecutive(self):
        return BTreeNodeNonConsecutive(self.variable, self.left_child, self.right_child,
                                      set(self.left_states), set(self.right_states), self.var_domain)

    def summary(self, n=0):
        s = f"<BTNode({self.variable})[...{self.left_states[-1]}|{self.right_states[0]}...]"
        s += "\n"
        s += "  "*n
        s += f"left: <{self.left_child.summary(n+1) if isinstance(self.left_child, BTreeNode) else self.left_child}>,"
        s += "\n"
        s += "  "*n
        s += f"right: <{self.right_child.summary(n+1) if isinstance(self.right_child, BTreeNode) else self.right_child}>>"
        return s

class BTreeNodeNonConsecutive(BTreeNode):
    def __init__(self, variable: Hashable, left_child, right_child, left_states: set, right_states: set, var_domain: list):

        # validations
        if not left_states and not right_states:
            raise ValueError(f"Either left_states or right_states must be provided")

        if left_states and right_states:
            if left_states.intersection(right_states)!=set():
                raise ValueError(f"left_states and right_states must be disjoint")

            if left_states.union(right_states)!= set(var_domain):
                raise ValueError(f"left_states and right_states must cover the whole var_domain")

        self._variable = variable
        self._var_domain = var_domain
        self._left_states = left_states
        self._right_states = right_states
        self._left_child = left_child
        self._right_child = right_child

    @property
    def left_states(self):
        return self._left_states

    @property
    def right_states(self):
        return self._right_states

    def is_on_left(self, state):
        return state in self._left_states

    def is_on_right(self, state):
        return state in self._right_states

    def summary(self, n=0):
        s = f"<BTNode({self.variable})[{self.left_states}|{self.right_states}]"
        s += "\n"
        s += "  "*n
        s += f"left: <{self.left_child.summary(n+1) if isinstance(self.left_child, BTreeNode) else self.left_child}>,"
        s += "\n"
        s += "  "*n
        s += f"right: <{self.right_child.summary(n+1) if isinstance(self.right_child, BTreeNode) else self.right_child}>>"
        return s


class BTreeStore(DiscreteStore):

    def __init__(self, domain: Dict, data: Union[Iterable, int, float, dict]=None, is_equation=False, exovar=None):
        from bcause.factors.values import NumpyStore
        #defualt data
        if data is None:
            data = np.zeros(np.prod([len(d) for d in domain.values()]))

        if len(domain)>0 and not isinstance(data, BTreeNode):
            if not is_equation:
                data = self._build_from_table(NumpyStore(domain, data))

            else:
                if exovar is None:
                    raise ValueError("In case of equations the exogenous variable must be provided")
                from bcause.factors.multinomial import btree_equation
                data = self._build_from_equation(NumpyStore(domain,data), exovar)
                #data = btree_equation(domain=domain, values=btree, vtype='btree')

        def builder(**kwargs):
            return BTreeStore(**kwargs)

        self.builder = builder
        from bcause.factors.values.btreeops import BTreeStoreOperations
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
    def _build_from_equation(table, exovar):

        # Select current variable
        #endo_vars = [v for v in table.variables if v != exovar]

        # Select first the left variables and then the right-vars
        vars = table.variables
        endo_vars = [v for v in vars if v != exovar]

        assert all(len(table.domain[var]) == 2 for var in endo_vars), \
            "Only binary variables are supported"
        if len(endo_vars) == 0:
            v = exovar
        else:
            v = endo_vars[0]

        # States for each branch
        if v == exovar:
            table_left = 0
            table_right = 1
            tl = np.array(table.domain[exovar])[np.where(np.array(table.data) == table_left)[0]]
            tr = np.array(table.domain[exovar])[np.where(np.array(table.data) == table_right)[0]]
            return BTreeNode.build(v, table.domain[v], left_child=table_left, right_child=table_right, left_states=tl,
                                   right_states=tr, consecutive=False)
        else:
            tl = [table.domain[v][0]]
            tr = [table.domain[v][1]]
            table_left = table.restrict(**{v: tl})
            table_right = table.restrict(**{v: tr})

            # Build recursively the subtrees
            tree_left = BTreeStore._build_from_equation(table_left, exovar)
            tree_right = BTreeStore._build_from_equation(table_right, exovar)

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
                    info_max = info

        return best_var, best_left_states, best_right_states, best_left_table, best_right_table

    @staticmethod
    def var_to_nonconsecutive(data, var):

        if not isinstance(data, BTreeNode):
            return data

        new_left = BTreeStore.var_to_nonconsecutive(data.left_child,var)
        new_right = BTreeStore.var_to_nonconsecutive(data.right_child,var)

        if data.variable == var and isinstance(data, BTreeNodeConsecutive):
            data = data.to_BtreeNodeNonConsecutive()
            return BTreeNode.build(
                variable=data.variable,
                var_domain=list(data.var_domain),
                left_child=new_left,
                right_child=new_right,
                left_states=data.left_states,
                right_states=data.right_states,
                consecutive=False
            )

        return BTreeNode.build(
            variable=data.variable,
            var_domain=list(data.var_domain),
            left_child=new_left,
            right_child=new_right,
            left_states=data.left_states,
            right_states=data.right_states,
            consecutive=isinstance(data, BTreeNodeConsecutive)
        )


    @staticmethod
    def _check_consistency(data, domain):
        return True

    def _copy_data(self):
        return copy.deepcopy(self.data)

    def set_value(self, value, observation):
        raise NotImplementedError("method not implemented")

    def get_value(self, **observation):
        raise NotImplementedError("method not implemented")

    def set_data(self, data):
        self._data = data

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


