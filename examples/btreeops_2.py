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

def reshape_value(domain, unshaped_values):
    dom = domain
    shape = [len(d) for d in dom.values()]
    val = np.reshape(unshaped_values, shape)
    return val
import numpy as np

def random_array_with_sum(n=256, k=16, total=1.0):
    """
    Create an array of length n where:
      - k positions have random values that sum to `total`
      - the rest are zeros
    """
    if k > n:
        raise ValueError("k cannot be larger than n")
    values = np.random.rand(k)
    values = values / values.sum() * total
    arr = np.concatenate([values, np.zeros(n - k)])
    np.random.shuffle(arr)

    return arr


m2 = StructuralCausalModel.read("/Users/antoniogonzalezalves/Documents/BenchMarkWUPES/g5_model_22.bif")
data2 = pd.read_csv("/Users/antoniogonzalezalves/Documents/BenchMarkWUPES/g5_data_22.csv")

exovar = "U1U2U3U4U5"
# Variable call endo_dict with values 0 and 1 for variables X,Y1,Y2,Y3,Y4
domains = dict(X=["0", "1"], Y1=["0", "1"], Y2=["0", "1"], Y3=["0", "1"], Y4=["0", "1"])

empirical_prob2 = to_counts(dict(X=[0, 1], Y1=[0, 1], Y2=[0, 1], Y3=[0, 1], Y4=[0, 1]), data=data2,normalize=True)
empirical_tree2 = BTreeStore(domain=domains, data=reshape_value(domains,empirical_prob2.values), is_equation=False)

U_value2 = m2.factors["U1U2U3U4U5"]
val_U = U_value2.values
val_U = random_array_with_sum(n=1024,k=8)

U_tree2 = BTreeStore(domain=U_value2.domain, data=reshape_value(U_value2.domain,val_U), exovar=exovar, is_equation=False)
U_tree2.set_data(BTreeStore.var_to_nonconsecutive(U_tree2.data, "U1U2U3U4U5"))
dom_X = m2.factors["X"].domain
bt_X = BTreeStore(domain=dom_X, data=reshape_value(dom_X,m2.factors["X"].values), exovar="V", is_equation=False)
dom_Y1 = m2.factors["Y1"].domain
bt_Y1 = BTreeStore(domain=dom_Y1, data=reshape_value(dom_Y1,m2.factors["Y1"].values), exovar=exovar, is_equation=True)
dom_Y2 = m2.factors["Y2"].domain
bt_Y2 = BTreeStore(domain=dom_Y2, data=reshape_value(dom_Y2,m2.factors["Y2"].values), exovar=exovar, is_equation=True)
dom_Y3 = m2.factors["Y3"].domain
bt_Y3 = BTreeStore(domain=dom_Y3, data=reshape_value(dom_Y3,m2.factors["Y3"].values), exovar=exovar, is_equation=True)
dom_Y4 = m2.factors["Y4"].domain
bt_Y4 = BTreeStore(domain=dom_Y4, data=reshape_value(dom_Y4,m2.factors["Y4"].values), exovar=exovar, is_equation=True)
dom_Y5 = m2.factors["Y5"].domain
bt_Y5 = BTreeStore(domain=dom_Y5, data=reshape_value(dom_Y5,m2.factors["Y5"].values), exovar=exovar, is_equation=True)

phi_2 = BTreeStoreOperations.SE_operation(bt_Y1,bt_Y2)
phi_2 = BTreeStoreOperations.SE_operation(phi_2,bt_Y3)
phi_2 = BTreeStoreOperations.SE_operation(phi_2,bt_Y4)
phi_2 = BTreeStoreOperations.SE_operation(phi_2,bt_Y5)
phi_1 = BTreeStoreOperations.multiply(empirical_tree2,phi_2)
U_values2 = [U_value2.values]

start = time.time()
t3_time, t4_time,t5_time, t6_time, final_step_time = 0,0,0,0,0
for i in range(1000):

    t0 = time.time()
    # T3 = BTreeStoreOperations.multiply(phi_2, U_tree2)
    T3 = BTreeStore(data=BTreeStoreOperations.multiply_exogenous(phi_2.data, U_tree2.data), domain=phi_2.domain)
    t3_time += time.time() - t0

    # Step T4
    t0 = time.time()
    T4 = BTreeStoreOperations.marginalize(T3, ["U1U2U3U4U5"])
    t4_time += time.time() - t0

    # Step T5
    t0 = time.time()
    T5 = BTreeStoreOperations.divide(phi_1, T4)
    t5_time += time.time() - t0

    # Step T6
    t0 = time.time()
    T6 = BTreeStore(data=BTreeStoreOperations.marginalize_endogenous(T5.data, exovar=exovar), domain=U_tree2.domain)
    # T6 = BTreeStoreOperations.marginalize(T5, ["X", "Y1", "Y2", "Y3", "Y4"])
    t6_time += time.time() - t0

    # Final update
    t0 = time.time()
    U_tree2 = BTreeStoreOperations.multiply(T6, U_tree2)
    final_step_time += time.time() - t0

end = time.time()
print("Time taken: ", end - start)
print("T3 time taken: ", t3_time)
print("T4 time taken: ", t4_time)
print("T5 time taken: ", t5_time)
print("T6 time taken: ", t6_time)
print("Final step time taken: ", final_step_time)