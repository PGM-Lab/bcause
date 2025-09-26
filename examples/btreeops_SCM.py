import logging
import sys
import copy
import math
from abc import ABC, abstractmethod
from typing import Hashable, List, OrderedDict, Dict, Union, Iterable

import numpy as np
from pandas.core.computation.expr import intersection
from torch.distributions.constraints import multinomial

from bcause.factors.values import BTreeStore
from bcause.factors.values.btreeops import BTreeStoreOperations
from bcause.factors.values.btreestore import BTreeNode
# from bcause.factors.multinomial import btree_equation
from bcause.factors.values.store import DiscreteStore
from bcause.models.cmodel import StructuralCausalModel
import pandas as pd
from bcause.util.datautils import to_counts
import time

from dev_ig.WUPES.Test_Structural import U_value

m = StructuralCausalModel.read("./models/model_wupes.bif")
data = pd.read_csv("./models/data_wupes.csv")

# Data to Multifactor
empirical_prob = to_counts(domains=dict(H=[0, 1], T=[0, 1], S=[0, 1]), data=data,normalize=True)

log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format, datefmt='%Y%m%d_%H%M%S')


exovar = "U"

def reshape_value(domain, unshaped_values):
    dom = domain
    shape = [len(d) for d in dom.values()]
    val = np.reshape(unshaped_values, shape)
    return val

# SCM into Trees
dom_T = m.factors["T"].domain
bt_T = BTreeStore(domain=dom_T, data=reshape_value(dom_T,m.factors["T"].values), exovar=exovar, is_equation=True)

dom_S = m.factors["S"].domain
bt_S = BTreeStore(domain=dom_S, data=reshape_value(dom_S,m.factors["S"].values), exovar=exovar, is_equation=True)

# Empirical into Trees
endo_domain = dict(H=["0", "1"], T=["0", "1"], S=["0", "1"])
empirical_tree = BTreeStore(domain=endo_domain, data=reshape_value(endo_domain,empirical_prob.values), is_equation=False)

# Exogenous into Tree
dom_U = m.factors["U"].domain
u_values = m.factors["U"].values
u_values = [0.1,0.1,0.6,0.2,0,0,0,0,0]
U_tree = BTreeStore(domain=dom_U, data=reshape_value(dom_U,u_values), is_equation=False)


print(U_tree.data.summary())

U_tree.data.right_child

U_tree.set_data(BTreeStore.var_to_nonconsecutive(U_tree.data, "U"))

# Equivalente a T1 en el pdf
phi_2 = BTreeStoreOperations.SE_operation(bt_T,bt_S)
# Equivalente a T2 en el pdf
phi_1 = BTreeStoreOperations.multiply(empirical_tree,phi_2)

start = time.time()
t3_time, t4_time,t5_time, t6_time, final_step_time = 0,0,0,0,0
for i in range(10000):

    t0 = time.time()
    # Original
    # T3 = BTreeStoreOperations.multiply(phi_2, U_tree)
    # Alternative
    T3 = BTreeStore(data=BTreeStoreOperations.multiply_exogenous(phi_2.data, U_tree.data), domain=phi_2.domain)
    t3_time += time.time() - t0

    # Step T4
    t0 = time.time()
    T4 = BTreeStoreOperations.marginalize(T3, ["U"])
    t4_time += time.time() - t0

    # Step T5
    t0 = time.time()
    T5 = BTreeStoreOperations.divide(phi_1, T4)
    t5_time += time.time() - t0

    #print(T5.data.summary())
    T6 = BTreeStore(data=BTreeStoreOperations.marginalize_endogenous(T5.data,exovar=exovar),domain=U_value)
    # T6_ = BTreeStoreOperations.marginalize(BTreeStoreOperations.multiply(T5, U_tree), ["H", "T", "S"])
    # Step T6
    t0 = time.time()

    subtrees = [
        BTreeStoreOperations.restrict(T5,dict(H="0", T="0", S="0")),
        BTreeStoreOperations.restrict(T5,dict(H="0", T="0", S="1")),
        BTreeStoreOperations.restrict(T5,dict(H="0", T="1", S="1")),
        BTreeStoreOperations.restrict(T5,dict(H="0", T="1", S="0")),
        BTreeStoreOperations.restrict(T5,dict(H="1", T="0", S="0")),
        BTreeStoreOperations.restrict(T5,dict(H="1", T="0", S="1")),
#        BTreeStoreOperations.restrict(T5,dict(H="1", T="1", S="1")),
#        BTreeStoreOperations.restrict(T5,dict(H="1", T="1", S="0"))
]

    from functools import reduce

    T6 = reduce(lambda a,b : BTreeStoreOperations.addition(a,b), subtrees)


    #print(T6.data.summary())
    #print(T6_.data.summary())


    #T6 = BTreeStoreOperations.marginalize(T5, ["H", "T", "S"])
    t6_time += time.time() - t0

    # Final update
    t0 = time.time()
    U_tree = BTreeStoreOperations.multiply(T6, U_tree)
    final_step_time += time.time() - t0

end = time.time()
print("Time taken: ", end - start)
print("T3 time taken: ", t3_time)
print("T4 time taken: ", t4_time)
print("T5 time taken: ", t5_time)
print("T6 time taken: ", t6_time)
print("Final step time taken: ", final_step_time)


