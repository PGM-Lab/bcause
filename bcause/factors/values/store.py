from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import reduce
from typing import Dict, Iterable, Union

import numpy as np

import bcause.util.domainutils as dutil
from bcause.util.treesutil import build_default_tree, treeNode


class DataStore(ABC):
    pass

class DiscreteStore(DataStore):

    def __init__(self, domain: Dict, data: Union[Iterable, int, float]):

        if not self._check_consistency(data, domain):
            raise ValueError("Cardinality Error")

        self._data = data
        self._domain = OrderedDict(domain)


        super().__init__()

    def copy(self, deep=True):
        new_data = self._copy_data() if deep else self._data
        new_dom = self.domain.copy()
        return self.builder(data=new_data, domain = new_dom)

    @property
    def variables(self):
        return list(self._domain.keys())

    @property
    def domain(self):
        return self._domain

    @property
    def data(self):
        return self._data

    @property
    def cardinality(self):
        return [len(d) for v,d in self._domain.items()]

    @property
    def cardinality_dict(self):
        return {v:len(d) for v,d in self._domain.items()}

    @staticmethod
    @abstractmethod
    def _check_consistency(data, domain):
        pass

    @abstractmethod
    def _copy_data(self):
        pass

    @abstractmethod
    def set_value(self, value, observation):
        pass


    def __repr__(self):
        cardinality_dict = self.cardinality_dict
        card_str = ",".join([f"{v}:{cardinality_dict[v]}" for v in self.variables])
        vars_str = ",".join([f"{v}" for v in self.variables])
        return f"<{self.__class__.__name__}({vars_str}), cardinality = ({card_str})>"


    @abstractmethod
    def restrict(self, **observation) -> DiscreteStore:
        pass

    @abstractmethod
    def get_value(self, **observation) -> NumpyStore:
        pass

    @abstractmethod
    def set_value(self,value, observation) -> float:
        pass

    def set_operationSet(self, ops:OperationSet):
        for f in [self.set_marginalize, self.set_maxmarginalize, self.set_multiply,
                  self.set_addition, self.set_subtract, self.set_divide]:
            f(ops)

    def set_marginalize(self, ops:OperationSet):
        self._marginalize = ops.marginalize

    def set_maxmarginalize(self, ops:OperationSet):
        self._maxmarginalize = ops.maxmarginalize

    def set_multiply(self, ops:OperationSet):
        self._multiply = ops.multiply

    def set_addition(self, ops:OperationSet):
        self._addition = ops.addition

    def set_subtract(self, ops:OperationSet):
        self._subtract = ops.subtract

    def set_divide(self, ops:OperationSet):
        self._divide = ops.divide

    def marginalize(self, *vars_remove) -> DiscreteStore:
        return self._marginalize(self, vars_remove)

    def maxmarginalize(self, *vars_remove) -> DiscreteStore:
        return self._maxmarginalize(self, vars_remove)

    def multiply(self, other:DiscreteStore) -> DiscreteStore:
        return self._multiply(self, other)

    def addition(self, other:DiscreteStore) -> DiscreteStore:
        return self._addition(self, other)

    def subtract(self, other: DiscreteStore) -> DiscreteStore:
        return self._subtract(self, other)

    def divide(self, other: DiscreteStore) -> DiscreteStore:
        return self._divide(self, other)

    @abstractmethod
    def sum_all(self):
        pass

    @property
    def values_list(self) -> list:
        return [self.get_value(**x) for x in dutil.assingment_space(self.domain)]
    @property
    def values_dict(self) -> dict:
        return {tuple(zip(d.keys(), d.values())): self.get_value(**d) for d in dutil.assingment_space(self.domain)}

    def values_str(self, maxvalues = 4):
        vals = self.values_list
        output = ",".join([str(x) for x in vals[0:maxvalues]])
        if len(vals) > maxvalues:
            output += f",...,{vals[-1]}"

        return output


    def sum_all(self) -> float:
        return sum(self.values_list)


    def get_var_index(self, v):
        if v not in self.variables:
            raise ValueError(f"Error: {v} not present in data store")
        gen = (i for i, e in enumerate(self.variables) if e == v)
        return next(gen)





class NumpyStore(DiscreteStore):
    def __init__(self, domain: Dict, data: Union[Iterable, int, float] = None):

        # default data
        if data is None:
            data = np.zeros([len(d) for d in domain.values()])

        def builder(*args, **kwargs):
            return NumpyStore(**kwargs)

        self.builder = builder
        self.set_operationSet(NumpyStoreOperations)


        super(self.__class__, self).__init__(domain=domain, data=np.array(data))


    @staticmethod
    def _check_consistency(data, domain):
        return (np.ndim(data) == len(domain)) \
               and (list(np.shape(data)) == [len(d) for v,d in domain.items()])

    def _copy_data(self):
        return self._data.copy()

    def set_value(self, value, observation):
        items = []
        for v in self.variables:
            idx = np.where(np.array(self.domain[v]) == observation[v])[0][0]
            items.append(idx)
        self._data[tuple(items)] = value

    def get_value(self, **observation):
        return np.atleast_1d(self.restrict(**observation).data)[0]

    def restrict(self, **observation) -> NumpyStore:
        items = []
        for v in self.variables:
            if v in observation.keys():
                idx = np.where(np.array(self.domain[v]) == observation[v])[0][0]
                items.append(idx)
            else:
                items.append(slice(None))
        new_data = self._data[tuple(items)].copy()
        new_dom = OrderedDict([(k,d) for k,d in self.domain.items() if k not in observation])
        return NumpyStore(new_dom, new_data)


    def reorder(self, *new_var_order):
        if len(self.variables)<2:
            return self

        # filter variables and add remaining variables
        new_var_order = [v for v in new_var_order if v in self.variables]
        new_var_order += [v for v in self.variables if v not in new_var_order]

        idx_var_order = [new_var_order.index(v) for v in self.variables]

        # set the new order in the values
        new_data = np.moveaxis(self.data, range(np.ndim(self.data)), idx_var_order)
        new_dom = OrderedDict([(v, self.domain[v]) for v in new_var_order])

        # create new object
        return self.builder(data = new_data, domain = new_dom)

    def extend_domain(self, **extra_dom):

        extra_dom = OrderedDict({v:c for v,c in extra_dom.items() if v not in self.variables})
        add_card = tuple([len(d) for d in extra_dom.values()])

        new_data = np.reshape(np.repeat(self.data, np.prod(add_card)), np.shape(self.data) + add_card)
        new_dom = OrderedDict({**self.domain, **extra_dom})

        return self.builder(data=new_data, domain=new_dom)

    def _generic_combine(self, op2: NumpyStore, operation:callable) -> NumpyStore:

        op1 = self

        if op1.__class__.__name__ != op2.__class__.__name__:
            raise ValueError("Combination with non-compatible data structure")

        new_domain = OrderedDict({**op1.domain, **op2.domain})

        # Extend domains if needed
        if len(op1.variables) < len(new_domain):
            op1 = op1.extend_domain(**new_domain)
        if len(op2.variables) < len(new_domain):
            op2 = op2.extend_domain(**new_domain)

        # Set the same variable order
        new_vars = list(new_domain.keys())
        if op1.variables != new_vars:
            op1 = op1.reorder(*new_vars)
        if op2.variables != new_vars:
            op2 = op2.reorder(*new_vars)

        return self.builder(data=operation(op1.data, op2.data), domain=new_domain)


class ListStore(DiscreteStore):

    def __init__(self, domain: Dict, data: Union[Iterable, int, float]=None):

        # default data
        if data is None:
            data = [0.0] * int(np.prod([len(d) for d in domain.values()]))

        def builder(**kwargs):
            return ListStore(**kwargs)

        self.builder = builder
        self.set_operationSet(ListStoreOperations)
        #self.set_operationSet(GenericOperations)

        super(self.__class__, self).__init__(domain=domain, data=list(np.ravel(data)))

    @staticmethod
    def _check_consistency(data, domain):
        return len(np.ravel(data)) == np.prod([len(d) for d in domain.values()])

    def _copy_data(self):
        return self._data.copy()

    def set_value(self, value, observation):
        idx = list(dutil.index_iterator(self.domain, observation))[0]
        self._data[idx] = value

    def get_value(self, **observation):
        return self.restrict(**observation).data[0]

    def restrict(self, **observation) -> ListStore:
        idx = list(dutil.index_iterator(self.domain, observation))
        new_data = [self._data[i] for i in idx]
        new_dom = OrderedDict([(k, d) for k, d in self.domain.items() if k not in observation])
        return self.builder(data= new_data, domain = new_dom)


class Numpy1DStore(DiscreteStore):

    def __init__(self, domain: Dict, data: Union[Iterable, int, float]=None):

        #defualt data
        if data is None:
            data = np.zeros(int(np.prod([len(d) for d in domain.values()])))

        def builder(**kwargs):
            return Numpy1DStore(**kwargs)

        self.builder = builder
        #self.set_operationSet(ListStoreOperations)
        self.set_operationSet(GenericOperations)
        super(self.__class__, self).__init__(domain=domain, data=np.ravel(data))

    @staticmethod
    def _check_consistency(data, domain):
        return len(np.ravel(data)) == np.prod([len(d) for d in domain.values()])

    def _copy_data(self):
        return self._data.copy()

    def set_value(self, value, observation):
        idx = list(dutil.index_iterator(self.domain, observation))[0]
        self._data[idx] = value

    def get_value(self, **observation):
        return self.restrict(**observation).data[0]

    def restrict(self, **observation) -> Numpy1DStore:
        idx = list(dutil.index_iterator(self.domain, observation))
        new_data = [self._data[i] for i in idx]
        new_dom = OrderedDict([(k, d) for k, d in self.domain.items() if k not in observation])
        return self.builder(data = new_data, domain = new_dom)



class TreeStore(DiscreteStore):

    def __init__(self, domain: Dict, data: Union[Iterable, int, float, dict]=None):
        #defualt data
        if data is None:
            data = np.zeros(np.prod([len(d) for d in domain.values()]))
        if len(domain)>0 and type(data) not in [dict, OrderedDict]:
            data = build_default_tree(domain, data)
        def builder(**kwargs):
            return TreeStore(**kwargs)

        self.builder = builder
        self.set_operationSet(TreeStoreOperations)   # set implement
        super(self.__class__, self).__init__(domain=domain, data=data)

    @staticmethod
    def _check_consistency(data, domain):
        return True

    def _copy_data(self):
        return copy.deepcopy(self.data)

    def set_value(self, value, observation):
        def modify_dict(data, observation, value):
            if not isinstance(data, dict) or len(observation) == 0:
                out = value
            elif data["var"] not in observation:
                new_ch = dict()
                for state, ch in data["children"].items():
                    new_ch[state] = value if not isinstance(ch, dict) else modify_dict(ch, observation, value)
                out = treeNode(data["var"], new_ch)

            else:
                other_obs = {v: s for v, s in observation.items() if v != data["var"]}
                ch = data["children"][observation[data["var"]]]
                data["children"][observation[data["var"]]] = modify_dict(ch, other_obs, value)
                out = data
            return out

        relevant_obs = {v: s for v, s in observation.items() if v in self.domain}
        self._data = modify_dict(self.data, relevant_obs, value)

    def get_value(self, **observation):
        return self.restrict(**observation).data

    def restrict(self, **observation) -> TreeStore:
        new_data = TreeStore.restrict_dict(self.data, observation)
        new_dom = OrderedDict([(k, d) for k, d in self.domain.items() if k not in observation])
        return self.builder(data = new_data, domain = new_dom)

    @staticmethod
    def restrict_dict(data, observation):
        if not isinstance(data, dict) or len(observation) == 0:
            out = data
        elif data["var"] not in observation:
            new_ch = dict()
            for state, ch in data["children"].items():
                new_ch[state] = ch if not isinstance(ch, dict) else TreeStore.restrict_dict(ch, observation)
            out = treeNode(data["var"], new_ch)
        else:
            other_obs = {v: s for v, s in observation.items() if v != data["var"]}
            out = TreeStore.restrict_dict(data["children"][observation[data["var"]]], other_obs)
        return out


store_dict = {"numpy": NumpyStore,"numpy1d": Numpy1DStore, "list":ListStore, "tree":TreeStore}
__ALL__ = list(store_dict.keys())

####### operations


class OperationSet(ABC):
    # @staticmethod
    # @abstractmethod
    # def check_store_class():
    #     pass


    @staticmethod
    @abstractmethod
    def marginalize(store : DataStore, vars_remove:list):
        pass

    @staticmethod
    @abstractmethod
    def maxmarginalize(store : DataStore, vars_remove:list):
        pass

    @staticmethod
    @abstractmethod
    def multiply(store : DataStore, other:DataStore) -> DataStore:
        pass

    @staticmethod
    @abstractmethod
    def addition(store : DataStore, other:DataStore) -> DataStore:
        pass

    @staticmethod
    @abstractmethod
    def subtract(store : DataStore, other: DataStore) -> DataStore:
        pass

    @staticmethod
    @abstractmethod
    def divide(store : DataStore, other: DataStore) -> DataStore:
        pass


class GenericOperations(OperationSet):
    @staticmethod
    def marginalize(store: DataStore, vars_remove: list):
        space_remove = dutil.assingment_space({v: d for v, d in store.domain.items() if v in vars_remove})
        new_dom = OrderedDict([(v,d) for v, d in store.domain.items() if v not in vars_remove])
        restricted = [store.restrict(**obs) for obs in space_remove]
        return reduce(GenericOperations.addition, restricted)

    @staticmethod
    def maxmarginalize(store: DataStore, vars_remove: list):
        raise NotImplementedError("Operation not implemented")

    @staticmethod
    def _generic_combine(op1: ListStore, op2: ListStore, operation:callable) -> ListStore:

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
    def multiply(store: DataStore, other: DataStore) -> DataStore:
        return GenericOperations._generic_combine(store, other, lambda x, y: x * y)

    @staticmethod
    def addition(store: DataStore, other: DataStore) -> DataStore:
        return GenericOperations._generic_combine(store, other, lambda x, y: x + y)

    @staticmethod
    def subtract(store: DataStore, other: DataStore) -> DataStore:
        return GenericOperations._generic_combine(store, other, lambda x, y: x - y)

    @staticmethod
    def divide(store: DataStore, other: DataStore) -> DataStore:
        print(store.data)
        print(other.data)
        return GenericOperations._generic_combine(store, other, lambda x, y: x / y)




class NumpyStoreOperations(OperationSet):

    @staticmethod
    def marginalize(store: DataStore, vars_remove: list):
        idx_vars = tuple(store.get_var_index(v) for v in vars_remove)
        new_data = np.sum(store.data, axis=idx_vars)
        new_dom = OrderedDict([(v,d) for v, d in store.domain.items() if v not in vars_remove])
        return store.builder(data=new_data, domain=new_dom)

    @staticmethod
    def maxmarginalize(store: DataStore, vars_remove: list):
        idx_vars = tuple(store.get_var_index(v) for v in vars_remove)
        new_data = np.max(store.data, axis=idx_vars)
        new_dom = OrderedDict([(v,d) for v, d in store.domain.items() if v not in vars_remove])
        return store.builder(data=new_data, domain=new_dom)

    @staticmethod
    def _generic_combine(op1:NumpyStore, op2: NumpyStore, operation:callable) -> NumpyStore:

        if op1.__class__.__name__ != op2.__class__.__name__:
            raise ValueError("Combination with non-compatible data structure")

        new_domain = OrderedDict({**op1.domain, **op2.domain})

        # Extend domains if needed
        if len(op1.variables) < len(new_domain):
            op1 = op1.extend_domain(**new_domain)
        if len(op2.variables) < len(new_domain):
            op2 = op2.extend_domain(**new_domain)

        # Set the same variable order
        new_vars = list(new_domain.keys())
        if op1.variables != new_vars:
            op1 = op1.reorder(*new_vars)
        if op2.variables != new_vars:
            op2 = op2.reorder(*new_vars)

        return op1.builder(data=operation(op1.data, op2.data), domain=new_domain)



    @staticmethod
    def multiply(store: DataStore, other: DataStore) -> DataStore:
        return NumpyStoreOperations._generic_combine(store, other, lambda x, y: x * y)

    @staticmethod
    def addition(store: DataStore, other: DataStore) -> DataStore:
        return NumpyStoreOperations._generic_combine(store, other, lambda x, y: x + y)

    @staticmethod
    def subtract(store: DataStore, other: DataStore) -> DataStore:
        return NumpyStoreOperations._generic_combine(store, other, lambda x, y: x - y)

    @staticmethod
    def divide(store: DataStore, other: DataStore) -> DataStore:
        return NumpyStoreOperations._generic_combine(store, other, lambda x, y: np.nan_to_num(x / y))






class ListStoreOperations(OperationSet):
    @staticmethod
    def marginalize(store: DataStore, vars_remove: list):
        space_remove = dutil.assingment_space({v: d for v, d in store.domain.items() if v in vars_remove})
        new_dom = OrderedDict([(v,d) for v, d in store.domain.items() if v not in vars_remove])
        iterators = [dutil.index_iterator(store.domain, s) for s in space_remove]
        new_len = int(np.prod([len(d) for d in new_dom.values()]))
        new_data = [sum([store.data[next(it)] for it in iterators]) for i in range(new_len)]
        return store.builder(data=new_data, domain=new_dom)

    @staticmethod
    def maxmarginalize(store: DataStore, vars_remove: list):
        space_remove = dutil.assingment_space({v: d for v, d in store.domain.items() if v in vars_remove})
        new_dom = OrderedDict([(v,d) for v, d in store.domain.items() if v not in vars_remove])
        iterators = [dutil.index_iterator(store.domain, s) for s in space_remove]
        new_len = int(np.prod([len(d) for d in new_dom.values()]))
        new_data = [max([store.data[next(it)] for it in iterators]) for i in range(new_len)]
        return store.builder(data=new_data, domain=new_dom)


    def _generic_combine(op1: ListStore, op2: ListStore, operation:callable) -> ListStore:

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
    def multiply(store: DataStore, other: DataStore) -> DataStore:
        return ListStoreOperations._generic_combine(store, other, lambda x, y: x * y)

    @staticmethod
    def addition(store: DataStore, other: DataStore) -> DataStore:
        return ListStoreOperations._generic_combine(store, other, lambda x, y: x + y)

    @staticmethod
    def subtract(store: DataStore, other: DataStore) -> DataStore:
        return ListStoreOperations._generic_combine(store, other, lambda x, y: x - y)

    @staticmethod
    def divide(store: DataStore, other: DataStore) -> DataStore:
        return ListStoreOperations._generic_combine(store, other, lambda x, y: x / y)


class TreeStoreOperations(OperationSet):

    @staticmethod
    def marginalize(store: DataStore, vars_remove: list):
        new_data = store.data
        for v in vars_remove:
            new_data = TreeStoreOperations._marginalize_dict(new_data, v, len(store.domain[v]), lambda x, y: x + y)
        new_dom = OrderedDict([(v, d) for v, d in store.domain.items() if v not in vars_remove])
        return store.builder(domain=new_dom, data=new_data)

    @staticmethod
    def maxmarginalize(store: DataStore, vars_remove: list):
        new_data = store.data
        for v in vars_remove:
            new_data = TreeStoreOperations._marginalize_dict(new_data, v, 1, lambda x, y: max(x,y))
        new_dom = OrderedDict([(v, d) for v, d in store.domain.items() if v not in vars_remove])
        return store.builder(domain=new_dom, data=new_data)
    @staticmethod
    def _combine_dict(d1, d2, operation):
        if not isinstance(d1, dict):
            if not isinstance(d2, dict):
                out = operation(d1, d2)
            else:
                new_var = d2["var"]
                new_ch = {state: TreeStoreOperations._combine_dict(d1, ch, operation) for state, ch in d2["children"].items()}
                out = treeNode(new_var, new_ch)
        else:
            new_var = d1["var"]
            new_ch = {state: TreeStoreOperations._combine_dict(ch, TreeStore.restrict_dict(d2, {new_var: state}), operation) for
                      state, ch in d1["children"].items()}
            out = treeNode(new_var, new_ch)

        return out

    @staticmethod
    def _marginalize_dict(d, var_to_remove, k, operation:callable):

        if not isinstance(d, dict):
            out = d * k
        else:
            if d["var"] == var_to_remove:
                out = reduce(lambda ch1, ch2: TreeStoreOperations._combine_dict(ch1, ch2, operation),
                             [ch for ch in d["children"].values()])
            else:
                new_var = d["var"]
                new_ch = {state: TreeStoreOperations._marginalize_dict(ch, var_to_remove, k, operation) for state, ch in d["children"].items()}
                out = treeNode(new_var, new_ch)

        return out

    @staticmethod
    def _generic_combine(op1:NumpyStore, op2: NumpyStore, operation:callable) -> NumpyStore:

        if op1.__class__.__name__ != op2.__class__.__name__:
            raise ValueError("Combination with non-compatible data structure")

        new_domain = OrderedDict({**op1.domain, **op2.domain})
        new_data = TreeStoreOperations._combine_dict(op1.data, op2.data, operation)
        return op1.builder(domain=new_domain, data=new_data)


    @staticmethod
    def multiply(store: DataStore, other: DataStore) -> DataStore:
        return TreeStoreOperations._generic_combine(store, other, lambda x, y: x * y)

    @staticmethod
    def addition(store: DataStore, other: DataStore) -> DataStore:
        return TreeStoreOperations._generic_combine(store, other, lambda x, y: x + y)

    @staticmethod
    def subtract(store: DataStore, other: DataStore) -> DataStore:
        return TreeStoreOperations._generic_combine(store, other, lambda x, y: x - y)

    @staticmethod
    def divide(store: DataStore, other: DataStore) -> DataStore:
        return TreeStoreOperations._generic_combine(store, other, lambda x, y: x / y)


if __name__=="__main__":
    left_domain = dict(A=["a1", "a2"])
    right_domain = dict(B=[0, 1, 3])
    domain = {**left_domain, **right_domain}

    new_var_order = ["B", "A"]
    #complete vars

    new_dom = OrderedDict([(v,domain[v]) for v in new_var_order])

    data = [[0.5, .4, 0.1], [0.3, 0.6, 0.1]]

    vars_remove = ["B"]
    f1 = NumpyStore(domain, data)
    f2 = ListStore(domain, data)

    for f in [f1,f2]:
        print(f.multiply(f.restrict(B=1)).marginalize("A").data)




