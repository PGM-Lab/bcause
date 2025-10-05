import logging
import sys
import pandas as pd
from functools import reduce
from bcause.models.cmodel import StructuralCausalModel
from bcause.util.datautils import to_counts
from bcause.util.watch import Watch



def SCM_operations(data,m,max_iter=100):

    log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format, datefmt='%Y%m%d_%H%M%S')

    endo_domain = {k: [int(v) for v in vals] for k, vals in m.domains.items() if k not in m.exogenous}
    empirical_prob = to_counts(domains=endo_domain, data=data,normalize=True)

    exovar = [x for x in m.exogenous if x != "V"][0]
    U_value = m.factors[exovar]

    # Precomputed
    phi_2 = reduce(lambda x, y: x * y, [m.factors[v] for v in m.endogenous if v != "X"])
    phi_1 = empirical_prob*phi_2
    U_values = [U_value.values]

    #check time
    Watch.start()
    for i in range(max_iter):
        T3 = phi_2*U_value
        T4 = T3.marginalize(exovar)
        T5 = phi_1/T4
        T6= T5.marginalize(*m.endogenous)
        U_value = T6 * U_value
        U_values.append(U_value.values)
    Watch.stop_print()
    return U_values


if __name__ == "__main__":
    m = StructuralCausalModel.read("./models/WUPES/model_wupes.bif")
    data = pd.read_csv("./models/WUPES/data_wupes.csv")
    max_iter = 100
    U_values = SCM_operations(data,m,max_iter=max_iter)

