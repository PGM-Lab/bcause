from typing import Iterable

import numpy as np


def normalize_array(data:Iterable, axis:Iterable):
    data = np.array(data)
    sums = data
    for i in axis:
        sums = sums.sum(axis=i, keepdims=1)

    return data/sums
