from bcause.factors.values.btreestore import BTreeStore
from bcause.factors.values.numpystore import NumpyStore, Numpy1DStore
from bcause.factors.values.liststore import ListStore
from bcause.factors.values.treedictstore import TreeDictStore

store_dict = {"numpy": NumpyStore,"numpy1d": Numpy1DStore, "list":ListStore, "treedict":TreeDictStore} #"btree":BTreeStore}
__ALL__ = list(store_dict.keys())
