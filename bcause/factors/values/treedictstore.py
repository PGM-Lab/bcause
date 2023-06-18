from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Iterable, Union

import numpy as np

import bcause.util.domainutils as dutil
from bcause.factors.values.operations import OperationSet
from bcause.factors.values.store import DiscreteStore
from bcause.factors.values.treedictops import TreeDictStoreOperations
from bcause.util.treesutil import build_default_tree, treeNode




class TreeDictStore(DiscreteStore):

    def __init__(self, domain: Dict, data: Union[Iterable, int, float, dict]=None):
        #defualt data
        if data is None:
            data = np.zeros(np.prod([len(d) for d in domain.values()]))
        if not isinstance(data, dict):
            if len(domain)>0 and type(data) not in [dict, OrderedDict]:
                data = build_default_tree(domain, data)
        def builder(**kwargs):
            return TreeDictStore(**kwargs)

        self.builder = builder
        self.set_operationSet(TreeDictStoreOperations)   # set implement
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

