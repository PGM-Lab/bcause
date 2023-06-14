
from bcause.factors.values.numpystore import NumpyStore, Numpy1DStore
from bcause.factors.values.liststore import ListStore

store_dict = {"numpy": NumpyStore,"numpy1d": Numpy1DStore, "list":ListStore}#, "tree":TreeStore}
__ALL__ = list(store_dict.keys())
