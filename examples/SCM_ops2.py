import logging
import sys
import copy
import math
from abc import ABC, abstractmethod
from typing import Hashable, List, OrderedDict, Dict, Union, Iterable

import numpy as np
from pandas.core.computation.expr import intersection
from torch.distributions.constraints import multinomial

# from bcause.factors.multinomial import btree_equation
from bcause.factors.values.store import DiscreteStore
from bcause.learning.parameter import expectation_maximization as em
from bcause.models.cmodel import StructuralCausalModel
import pandas as pd
from bcause.util.datautils import to_counts
import time

m2 = StructuralCausalModel.read("/Users/antoniogonzalezalves/Documents/BenchMarkWUPES/g5_model_22.bif")
data2 = pd.read_csv("/Users/antoniogonzalezalves/Documents/BenchMarkWUPES/g5_data_22.csv")
data2 = data2[["X","Y1","Y2","Y3","Y4","Y5"]]
exovar = "U1U2U3U4U5"

endo_domain2 = dict(X=[0, 1], Y1=[0, 1], Y2=[0, 1], Y3=[0, 1], Y4=[0, 1],Y5=[0,1])
empirical_prob2 = to_counts(domains=endo_domain2, data=data2,normalize=True)

U_value2 = m2.factors[exovar]
# Precomputed
phi_2_2 = m2.factors["Y1"]*m2.factors["Y2"]*m2.factors["Y3"]*m2.factors["Y4"]*m2.factors["Y5"]
phi_1_2 = empirical_prob2*phi_2_2
U_values2 = [U_value2.values]
#check time
start_time = time.time()
for i in range(100):
    T3_2 = phi_2_2*U_value2
    T4_2 = T3_2.marginalize(exovar)
    T5_2 = phi_1_2/T4_2
    T6_2= T5_2.marginalize("X","Y1","Y2","Y3","Y4","Y5")
    U_value2 = T6_2 * U_value2
    U_values2.append(U_value2.values)
end_time = time.time()
print("Time taken 2nd Model:", end_time - start_time)