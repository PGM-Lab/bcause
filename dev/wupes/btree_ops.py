import pandas as pd
import time
from bcause.util.watch import Watch


from bcause.factors.values import BTreeStore
from bcause.factors.values.btreeops import BTreeStoreOperations
from functools import reduce
from bcause.models.cmodel import StructuralCausalModel
from bcause.util.datautils import to_counts


def compute_tree_operation(data,m,exovar,max_iter=100, non_zero_values = False, combine_steps=False):

    # reshape values for the btrees
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


    # Define domains for endogenous variables
    endo_domain =  {k: m.domains[k] for k in m.endogenous}

    # Convert empirical data to probability counts
    empirical_prob = to_counts(endo_domain, data=data, normalize=True)
    empirical_tree = BTreeStore(domain=endo_domain, data=reshape_value(endo_domain, empirical_prob.values), is_equation=False)

    # Create BTreeStore for the exogenous variable
    U_value = m.factors[exovar]
    val_U = U_value.values

    val_U = val_U if not non_zero_values else random_array_with_sum(n=len(val_U), k=non_zero_values)
    U_tree = BTreeStore(domain=U_value.domain, data=reshape_value(U_value.domain, val_U), exovar=exovar, is_equation=False)
    U_tree.set_data(BTreeStore.var_to_nonconsecutive(U_tree.data, exovar))
    U_tree = BTreeStore(data=BTreeStoreOperations.correct_tree(U_tree.data), domain=U_tree.domain, is_equation=False)

    # Create BTreeStore for each endogenous variable except for X
    bt_factors = {}
    component = next((s for s in m.ccomponents if exovar in s), None)
    for var in [x for x in m.endogenous if x in component]:
        dom = m.factors[var].domain
        bt_factors[var] = BTreeStore(domain=dom, data=reshape_value(dom, m.factors[var].values), exovar=exovar, is_equation=True)

    # Perform SE operations
    phi_2 = reduce(lambda x,y: BTreeStoreOperations.multiply_SE(x, y, method = "SE_only"), bt_factors.values())
    phi_1 = BTreeStoreOperations.multiply(phi_2, empirical_tree)

    Watch().start()
    for i in range(max_iter):

        # Step T3
        # T3 = BTreeStoreOperations.multiply(phi_2, U_tree)
        T3 = BTreeStore(data=BTreeStoreOperations._mult_exogenous(phi_2.data, U_tree.data), domain=phi_2.domain)

        # Step T4
        T4 = BTreeStoreOperations.marginalize(T3, [exovar])

        # Step T5
        T5 = BTreeStoreOperations.divide(phi_1, T4)

        if combine_steps:
            # Step T6 and T7
            subtree = BTreeStoreOperations.marginalize_endogenous(T5.data,exovar=exovar)
            U_tree = BTreeStore(data = BTreeStoreOperations.addition_exo(subtree,U_tree.data,exo_mult=True),domain=U_tree.domain)
        else:
            # Step T6
            subtrees = BTreeStoreOperations.marginalize_endogenous(T5.data, exovar=exovar)
            T6 = BTreeStore(data=BTreeStoreOperations.addition_exo(subtrees,U_tree,combine_steps = False), domain=U_tree.domain)
            # T6 = BTreeStoreOperations.marginalize(T5, [k for k in endo_domain if k != exovar])

            # Step T7
            U_tree = BTreeStoreOperations.multiply(T6, U_tree)
            # U_tree = BTreeStore(data=BTreeStoreOperations.multiply_exogenous(T6.data, U_tree.data), domain=U_tree.domain)

    time_taken = Watch().get_time()

    return U_tree, time_taken

def initialize_tree(data,m,exovar, non_zero_values = False):
    # reshape values for the btrees
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

    # Define domains for endogenous variables
    endo_domain =  {k: m.domains[k] for k in m.endogenous}

    # Convert empirical data to probability counts
    empirical_prob = to_counts(endo_domain, data=data, normalize=True)
    empirical_tree = BTreeStore(domain=endo_domain, data=reshape_value(endo_domain, empirical_prob.values), is_equation=False)

    # Create BTreeStore for the exogenous variable
    U_value = m.factors[exovar]
    val_U = U_value.values

    val_U = val_U if not non_zero_values else random_array_with_sum(n=len(val_U), k=non_zero_values)
    U_tree = BTreeStore(domain=U_value.domain, data=reshape_value(U_value.domain, val_U), exovar=exovar, is_equation=False)
    U_tree.set_data(BTreeStore.var_to_nonconsecutive(U_tree.data, exovar))
    U_tree = BTreeStore(data=BTreeStoreOperations.correct_tree(U_tree.data), domain=U_tree.domain, is_equation=False)

    # Create BTreeStore for each endogenous variable except for X
    bt_factors = {}
    component = next((s for s in m.ccomponents if exovar in s), None)
    for var in [x for x in m.endogenous if x in component]:
        dom = m.factors[var].domain
        bt_factors[var] = BTreeStore(domain=dom, data=reshape_value(dom, m.factors[var].values), exovar=exovar, is_equation=True)
    return empirical_tree, U_tree, bt_factors

def execute_tree(empirical_tree, U_tree, bt_factors, exovar, max_iter=100, combine_steps=False):
    # Perform SE operations
    phi_2 = reduce(lambda x,y: BTreeStoreOperations.multiply_SE(x, y, method = "SE_only"), bt_factors.values())
    phi_1 = BTreeStoreOperations.multiply(phi_2, empirical_tree)

    Watch().start()
    for i in range(max_iter):

        # Step T3
        # T3 = BTreeStoreOperations.multiply(phi_2, U_tree)
        T3 = BTreeStore(data=BTreeStoreOperations._mult_exogenous(phi_2.data, U_tree.data), domain=phi_2.domain)

        # Step T4
        T4 = BTreeStoreOperations.marginalize(T3, [exovar])

        # Step T5
        T5 = BTreeStoreOperations.divide(phi_1, T4)

        if combine_steps:
            # Step T6 and T7
            subtree = BTreeStoreOperations.marginalize_endogenous(T5.data, exovar=exovar)
            U_tree = BTreeStore(data=BTreeStoreOperations.addition_exo(subtree, U_tree.data, exo_combine=True),
                                domain=U_tree.domain)
        else:
            # Step T6
            subtrees = BTreeStoreOperations.marginalize_endogenous(T5.data, exovar=exovar)
            subtrees_complemented = BTreeStoreOperations.sum_complementary_states(subtrees)
            T6 = BTreeStore(
                data=reduce(lambda a, b: BTreeStoreOperations.combine_btreenode(a, b, lambda x, y: x + y),
                            subtrees_complemented), domain=U_tree.domain)
            # T6 = BTreeStoreOperations.marginalize(T5, [k for k in endo_domain if k != exovar])

            # Step T7
            U_tree = BTreeStoreOperations.multiply(T6, U_tree)
            # U_tree = BTreeStore(data=BTreeStoreOperations.multiply_exogenous(T6.data, U_tree.data), domain=U_tree.domain)

    time_taken = Watch().get_time()
    return U_tree, time_taken


if __name__ == "__main__":

    # Read model and data
    g2_model_path = "./models/WUPES/g2_model_22.bif"
    g2_data_path = "./models/WUPES/g2_data_22.csv"
    g4_model_path = "./models/WUPES/g4_model_64.bif"
    g4_data_path = "./models/WUPES/g4_data_64.csv"
    g5_model_path = "./models/WUPES/g5_model_22.bif"
    g5_data_path = "./models/WUPES/g5_data_22.csv"
    g6_data_path = "./models/WUPES/g6_data_0.csv"
    g6_model_path = "./models/WUPES/g6_model_0.bif"
    m = StructuralCausalModel.read(g2_model_path)
    data = pd.read_csv(g2_data_path).astype(str)

    max_iter = 100
    exovar = [x for x in m.exogenous if x != "V"][0]
    # U_tree,time = compute_tree_operation(data, m, exovar,max_iter=max_iter, non_zero_values=False, combine_steps=False)
    Watch.start()
    empirical_tree, U_tree, bt_factors = initialize_tree(data, m, exovar, non_zero_values=False)
    Watch.stop_print()
    Watch.start()
    U_tree, time = execute_tree(empirical_tree, U_tree, bt_factors, exovar, max_iter=max_iter, combine_steps=False)
    Watch.stop_print()
    print(time)