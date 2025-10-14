import logging
import sys
import pandas as pd
from functools import reduce
from bcause.models.cmodel import StructuralCausalModel
from bcause.util.datautils import to_counts
from bcause.util.watch import Watch
from bcause.factors import MultinomialFactor



def SCM_operations(data,m,exovar,max_iter=100, as_list=False):

    log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format, datefmt='%Y%m%d_%H%M%S')

    endo_domain = {k: m.domains[k] for k in m.endogenous}
    vtype = "list" if as_list else None
    empirical_prob = to_counts(domains=endo_domain, data=data,normalize=True, vtype=vtype)

    U_value = m.factors[exovar]
    component = next((s for s in m.ccomponents if exovar in s), None)
    if not as_list:
        phi_2 = reduce(lambda x, y: x * y, [m.factors[x] for x in m.endogenous if x in component])
        phi_1 = empirical_prob*phi_2

    else:
        def change_type(factor, new_type="list"):
            if new_type == "list":
                return MultinomialFactor(domain=factor.domain, values=factor.values, left_vars=factor.left_vars,
                                         right_vars=factor.right_vars, vtype="list")
        list_factors = [change_type(m.factors[v], new_type="list") for v in m.endogenous if v != "X"]
        phi_2 = reduce(lambda x, y: x * y, list_factors)
        phi_1 = phi_2*empirical_prob
        U_value = change_type(U_value, new_type="list")

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
    time_taken = Watch.get_time()
    return U_values, time_taken


if __name__ == "__main__":
    # Read model and data
    g2_model_path = "./models/WUPES/g2_model_22.bif"
    g2_data_path = "./models/WUPES/g2_data_22.csv"
    g5_model_path = "./models/WUPES/g5_model_22.bif"
    g5_data_path = "./models/WUPES/g5_data_22.csv"
    g6_data_path = "./models/WUPES/g6_data_0.csv"
    g6_model_path = "./models/WUPES/g6_model_0.bif"
    m = StructuralCausalModel.read(g2_model_path)
    data = pd.read_csv(g2_data_path).astype(str)
    exovar = [x for x in m.exogenous if x != "V"][0]

    max_iter = 1000
    U_values = SCM_operations(data,m,exovar,max_iter=max_iter, as_list=False)

