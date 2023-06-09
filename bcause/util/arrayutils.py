from typing import Iterable
from itertools import chain, combinations

import numpy as np


def normalize_array(data:Iterable, axis:Iterable):
    data = np.array(data)
    sums = data
    for i in axis:
        sums = sums.sum(axis=i, keepdims=1)

    return data/sums

def powerset(iterable:Iterable):
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

def len_iterable(it:Iterable):
    return sum(1 for _ in it)

def set_value(value, data, idx):
    d = data
    for i in range(np.ndim(data) - 1):
        d = d[idx[i]]
    d[idx[-1]] = value

def as_lists(*args):
    def as_list(elem):
        if isinstance(elem, str): return [elem]
        return list(elem)

    if len(args)>1:
        return [as_list(elem) for elem in args]
    return as_list(args[0])

def change_shape_order(values:list, original_shape:tuple):
    return np.array(values).reshape(original_shape,order="F").ravel()

