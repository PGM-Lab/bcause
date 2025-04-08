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
        if elem is None: return None
        if isinstance(elem, str): return [elem]
        return list(elem)

    if len(args)>1:
        return [as_list(elem) for elem in args]
    return as_list(args[0])

def as_sets(*args):
    return [set(l) for l in as_lists(*args)]

def change_shape_order(values:list, original_shape:tuple):
    return np.array(values).reshape(original_shape,order="F").ravel()


def delete_outliers_iqr(data):
    data = np.array(data)
    if np.ndim(data)==1:
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)

        iqr = q3 - q1
        threshold = 1.5 * iqr

        mask = (data < q1 - threshold) | (data > q3 + threshold)
        return data[~mask]
    elif np.ndim(data)==2:
        return [delete_outliers_iqr(data[i,:]) for i in range(data.shape[0])]
    else:
        raise ValueError("Wrong dimensions")


def min_max_iqr(data):
    non_outliers = delete_outliers_iqr(data)
    # check for dimension 1
    if np.ndim(data)>1:
        return [np.min(v) for v in non_outliers], [np.max(v) for v in non_outliers]
    return np.min(non_outliers),np.max(non_outliers)


def concatenate_with(arr, fill_value, axis=0):
    # Create a new array filled with the fill_value, matching the shape of arr along other axes
    fill_shape = [s for s in arr.shape]
    fill_shape[axis] = 1

    fill_array = np.full(fill_shape, fill_value)

    # Concatenate along the specified axis
    return np.concatenate((arr, fill_array), axis=axis)