from abc import ABC, abstractmethod
from typing import Hashable, List


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

    def __repr__(self):
        return f"<BTNode[...{self.left_states[-1]}|{self.right_states[0]}...] left: <{self.left_child}>, right: <{self.right_child}>>"


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
