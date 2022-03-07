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