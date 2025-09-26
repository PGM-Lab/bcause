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

m = StructuralCausalModel.read("./models/model_wupes.bif")
data = pd.read_csv("./models/data_wupes.csv")

log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format, datefmt='%Y%m%d_%H%M%S')

endo_domain = dict(H=[0, 1], T=[0, 1], S=[0, 1])
empirical_prob = to_counts(domains=endo_domain, data=data,normalize=True)

U_value = m.factors["U"]

# Precomputed
phi_2 = m.factors["T"]*m.factors["S"]
phi_1 = empirical_prob*phi_2
U_values = [U_value.values]

#check time
start_time = time.time()
for i in range(10000):
    T3 = phi_2*U_value
    T4 = T3.marginalize("U")
    T5 = phi_1/T4
    T6= T5.marginalize("H","T","S")
    U_value = T6 * U_value
    U_values.append(U_value.values)
end_time = time.time()
print("Time taken 1st model:", end_time - start_time)

