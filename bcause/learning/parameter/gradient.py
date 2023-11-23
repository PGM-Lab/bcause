"""
Compute the PMF of exogenous variables given data for endogenous variable using
gradient descent method for negative likelihood objective function

TODO:
1) to check correctness, implement Theorem 1 from Causal EM paper
2) render canonical SEs - Rafa will do it
With RAFA: (it about effectivity - if some approach is usual, let's use it for the following)
3) which example models to use
4) which metrics to use for a comparision with EM
5) how to present the results (visuals)

TESTS DONE:
I tested 12.10.23 the likehood values obtained by fitPMF and they were very close to model.max_log_likelihood(data, variables=...)
"""

from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import itertools
import random
from typing import Dict, List, Tuple, Any


import bcause as bc
from bcause.factors import MultinomialFactor, DeterministicFactor
from bcause.factors.mulitnomial import random_multinomial   # TODO: mulitnomial -> multinomial
from bcause.learning.parameter import IterativeParameterLearning
from bcause.models.cmodel import StructuralCausalModel
from bcause.util.domainutils import assingment_space, state_space # TODO: assingment_space -> assignment_space


class GradientLikelihood(IterativeParameterLearning):
    '''
    This class implements a method for running a single optimization of the exogenous variables in an SCM.
    '''

    def __init__(self, prior_model: StructuralCausalModel, trainable_vars: list = None, tol : float = 1e-8):
        self._prior_model = prior_model
        self._trainable_vars = trainable_vars
        self._tol = tol


    def initialize(self, data: pd.DataFrame, **kwargs):
        self._model = self.prior_model.copy()
        self._process_data(data)

    def _stop_learning(self) -> bool:
        # todo: implement a function that check if the process is stopped
        pass

    def _calculate_updated_factors(self, **kwargs) -> dict[MultinomialFactor]:
        return {U:self._updated_factor(U, **kwargs) for U in self.trainable_vars}

    def _updated_factor(self, U: str, **kwargs) -> MultinomialFactor:
        # Updates the PMF of variable U given data and an SCM        
        #
        # Parameters:
        #   U (string): The name of an exogenous variable.
        # Returns:
        #   MultinomialFactor(U, values): Talues (thetas) describe the estimated distribution of U (P(u_i) = \theta_i)
        m = self._prior_model
        data = self._data

        if 1:
            initial_params = m.factors[U].values # we start the optimization from the current PMF of U
        else: # use this case only when m.factors[U].values are not generated randomly but you what it
            dirich_distr = [1.0] * len(m.factors[U].values)
            initial_params = np.random.dirichlet(dirich_distr, 1)

        # get the quantities named as in our paper
        bmV = m.get_edogenous_children(U)
        bmY = self._get_bmY(bmV, m)

        # compute N[\bmv,\bmy] and \prod_{V \in \bmV} P(v | pa_V) from the paper
        N_bmVbmY, P_bmVbmYu = self._compute_N_and_P(bmV, bmY, data, m, U)

        results = self.fitPMF(initial_params, N_bmVbmY, P_bmVbmYu, **kwargs)
        updated_factor = MultinomialFactor(m.get_domains(U), values=results['params'])
        updated_factor.trajectory = results['trajectory'] # the trajectory of the iterations in the optimization process

        return updated_factor

    def _get_bmY(self, bmV: List[str], m: Any) -> List[str]:
        bmY = list(itertools.chain(*[m.get_edogenous_parents(V) for V in bmV]))
        bmY = [V for V in bmY if V not in bmV]
        return bmY

    def _compute_N_and_P(self, bmV: List[str], bmY: List[str], data: Any, m: Any, U: str) -> Tuple[Dict, Dict]:
        N_bmVbmY = defaultdict(lambda: {})
        P_bmVbmYu = defaultdict(lambda: {})

        for bmv in assingment_space(m.get_domains(bmV)):
            id_v = tuple(bmv.items())
            for bmy in assingment_space(m.get_domains(bmY)):
                id_y = tuple(bmy.items())
                filtered_data = self._filter_data(data, bmv, bmy)
                N = len(filtered_data)
                N_bmVbmY[id_v][id_y] = N

                P_bmVbmYu[id_v][id_y] = self._compute_P(bmv, bmy, m, U, bmV, bmY)

        return N_bmVbmY, P_bmVbmYu

    def _filter_data(self, data: Any, bmv: Dict, bmy: Dict) -> Any:
        # Perform the comparison operations
        filter_bmv = (data[list(bmv.keys())] == list(bmv.values()))
        filter_bmy = (data[list(bmy.keys())] == list(bmy.values()))

        # Check if the results are empty. For the latter, this check is only for the case when bmy is non-empty (i.e., V has some endogenous parent)
        if filter_bmv.empty or (bmy and filter_bmy.empty):
            # Handle the empty case (e.g., return an empty DataFrame)
            return pd.DataFrame()

        # Apply the 'all' operation and filter the data
        return data[
            filter_bmv.all(axis=1) &
            filter_bmy.all(axis=1)
        ]


    def _compute_P(self, bmv: Dict, bmy: Dict, m: Any, U: str, bmV: List[str], bmY: List[str]) -> Dict:
        # compute \prod_{V \in \bmV} P(v|Pa(V))
        P = {}
        for u in m.get_domains(U)[U]:
            prod_P_v_pav = 1 # As the product of P(v|Pa(V)) \in {0, 1}, so we initiate the value to 1.
                             # If we find any P(v|Pa(V)) = 0, we stop the computation as it must be that \prod_{V \in \bmV} P(v|Pa(V)) = 0   
            for V in bmV:
                P_v_pav_key = self._get_P_v_pav_key(bmv, bmy, m, u, V, bmV, bmY)
                if m.factors[V].values_dict[tuple(P_v_pav_key)] == 0: # TODO: Rafa, please initiate values_dict in MultinomialFactor in order to contain Integer (binary) only values!
                    prod_P_v_pav = 0
                    break # once we got 0, the product must be zero, so we finish immediately
            P[u] = prod_P_v_pav
        return P

    def _get_P_v_pav_key(self, bmv: Dict, bmy: Dict, m: Any, u: str, V: str, bmV: List[str], bmY: List[str]) -> List[Tuple[str, Any]]:
        P_v_pav_key = []
        for var in m.factors[V].variables:
            if var in m.exogenous:
                key = (var, u)
            elif var in bmV:
                key = (var, bmv[var])
            elif var in bmY:
                key = (var, bmy[var])
            else:
                raise Exception(f'Unclassifiable variable! ({var})')
            P_v_pav_key.append(key)
        return P_v_pav_key

    def _process_data(self, data: pd.DataFrame):
        # add missing variables
        missing_vars = [v for v in self.prior_model.variables if v not in data.columns]
        for v in missing_vars: data[v] = float("nan")

        # Set as trainable variables those with missing
        self._trainable_vars = self.trainable_vars or list(data.columns[data.isna().any()]) 
                                                                                            

        for v in self._trainable_vars:
            # check exogenous and completely missing
            if not self.prior_model.is_exogenous(v):
                raise ValueError(f"Trainable variable {v} is not exogenous")

            if (~data[v].isna()).any():
                raise ValueError(f"Trainable variable {v} is not completely missing")

        # save the dataset
        self._data = data


    def negative_log_likelihood(self, theta, N_bmVbmY, P_bmVbmYu):
        log_likelihood = .0
        m = self._prior_model
        for id_v in P_bmVbmYu:
            for id_y in P_bmVbmYu[id_v]:
                if N_bmVbmY[id_v][id_y]:
                    sum_lkh_bmvbmy = .0
                    for i_u, u in enumerate(P_bmVbmYu[id_v][id_y]):
                        sum_lkh_bmvbmy += P_bmVbmYu[id_v][id_y][u] * theta[i_u] # TODO: this "for" is just a scalar dot product - thus could be made faster
                    if sum_lkh_bmvbmy:
                        log_likelihood += N_bmVbmY[id_v][id_y] * np.log(sum_lkh_bmvbmy)
                    else:
                        return np.inf # sum_lkh_bmvbmy == 0 implies that the parameters induce the worst possible likelihood for the given data
                                      # so, we can immediately end with the worth possible value

        #log_likelihood = np.sum(counts * np.log(np.sum(theta * pa_probs, axis=1)))
        return -log_likelihood  # We negate the value to convert maximization to minimization


    def constraint(self,theta):
        return np.sum(theta) - 1  # The probabilities should sum up to 1


    def fitPMF(self, initial_params, N_bmVbmY, P_bmVbmYu, **kwargs):
        """
        Fits the PMF of an exogenous variable MultinomialFactor(U, values), where 
        values (thetas) describe the estimated distribution of U (P(u_i) = \theta_i)
        given data distribution in N_bmVbmY and SE in P_bmVbmYu

        Parameters:
            initial_params (array-like): The initial parameter values of thetas for the optimization algorithm.
            N_bmVbmY: The data distribution.
            P_bmVbmYu: The SE.

        Returns:
            dict: A dictionary containing the estimated parameters and additional information from the optimization algorithm.
        """

        trajectory = []  # saves each iteration of the optimization process
        def callback(xk):
            trajectory.append(xk) # save the parameters

        # Constraint dictionary
        con = {'type': 'eq', 'fun': self.constraint}

        options = dict()
        if self._max_iter < float("inf"):
            options["maxiter"] = self._max_iter

        # Perform the optimization using scipy's minimize function
        result = minimize(self.negative_log_likelihood, initial_params, 
                          args = (N_bmVbmY, P_bmVbmYu), constraints=con, 
                          bounds=[(0, 1)]*len(initial_params),
                          tol = self._tol, callback = callback,
                          options=options) # default: method='SLSQP'

        # Transform the estimated raw parameters to the constrained parameters
        estimated_params = result.x
        #print(f'{estimated_params}=')
        #print(f'sum_params = {estimated_params.sum()}')

        # Create a dictionary of results
        estimation_results = {
            'params': estimated_params,
            'success': result.success,
            'message': result.message,
            'fun': result.fun,
            'nit': result.nit,
            'trajectory': trajectory,
        }
        #print(f'{result.nit=}') # number of iterations
        #print(f'{result.fun=}') # neg-log-likelihood
        return estimation_results


    def run(self, data: pd.DataFrame, max_iter: int = float("inf")):
        """
        This method performs a given number of optimization steps.
        Args:
            data: training data.
            max_iter: number of iterations. Default is None and runs util converge.

        Self is updated with the optimized parameters.
        """
        self._max_iter = max_iter
        self.initialize(data)
        self.step()



###################################
### further auxiliary functions ###
###################################

def tick(n, var_label = None):
    use_var_label = True # switch to False to avoid using var_label
    if use_var_label and var_label:
        ticks = []
        for i in range(n):
            ticks.append(f'{var_label}{i+1}')
    else:
        ticks = list(range(n))
    return ticks


def show_dag(DAG):
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(DAG)  # Position nodes using a spring layout algorithm
    nx.draw(DAG, pos, with_labels=True, node_size=1000, node_color="lightblue", font_size=10, font_color="black", arrowsize=20)
    plt.show()



def define_model0():
    dag = nx.DiGraph([("X", "Y"), ("U", "Y"), ("V", "X")])
    if 0:
        show_dag(DAG)
    domains = dict(X=["x1", "x2"], Y=["y1","y2"], U=[0, 1, 2, 3], V=[0, 1])
    domx = dutils.var_parents_domain(domains,dag,"X")
    fx = DeterministicFactor(domx, right_vars=["V"], values=["x1", "x2"]).to_multinomial()
    domy = dutils.var_parents_domain(domains,dag,"Y")
    values = ["y1", "y1", "y2", "y1", "y2", "y2", "y1", "y1"] 
    fy = DeterministicFactor(domy, left_vars=["Y"], values=values,vtype="list").to_multinomial()
    domv = dutils.subdomain(domains, "V")
    pv = MultinomialFactor(domv, values=[.5, .5])
    domu = dutils.subdomain(domains, "U")
    pu = MultinomialFactor(domu, values=[.2, .2, .6, .0])
    model = StructuralCausalModel(dag, [fx, fy, pu, pv], cast_multinomial=True)
    return model


def define_model1():
    # basic definition & paper2code mapping
    DAG = nx.DiGraph([("X1", "X2"), ("U1", "X1"), ("U2", "X2")]) # definition of \mathcal{G}
    if 0:
        show_dag(DAG)
    domains = dict(X1=tick(2, 'x1'), X2=tick(2, 'x2'), U1=tick(2, 'u1'), U2=tick(4, 'u2')) # definition of \dom{X1}, \dom{X2}, \dom{U1}, \dom{U2}

    parents_X1 = gutils.relevat_vars(DAG, "X1") # \mathrm{Pa}_{X_1} \cup {X1}
    dom_X1 = dutils.subdomain(domains, *parents_X1) # get domains of parents_X1
    f_X1 = DeterministicFactor(dom_X1, right_vars=["U1"], values=["x12", "x11"]) # SE f_X1 

    dom_X2 = dutils.subdomain(domains, *gutils.relevat_vars(DAG, "X2"))
    values = ["x21", "x21", "x22", "x21", "x21", "x21", "x22", "x21"]
    f_X2 = DeterministicFactor(dom_X2, left_vars=["X2"], values=values) # SE f_X2

    dom_U1 = dutils.subdomain(domains, "U1")
    p_U1 = MultinomialFactor(dom_U1, values=[.5, .5]) # P(U1)

    dom_U2 = dutils.subdomain(domains, "U2")
    p_U2 = MultinomialFactor(dom_U2, values=[.2, .2, .1, .5]) # P(U2)

    m = StructuralCausalModel(DAG, [f_X2, f_X1, p_U2, p_U1], cast_multinomial=True) # an instance of the SCM based the graph DAG with SE f_X2, f_X1 and given distributions of U2 and U1
    return m


def define_model2():
    dag = nx.DiGraph([("V1", "V2"), ("V2", "V3"),("V3", "V4"),("U1", "V1"),("U2", "V2"),("U2", "V4"),("U3", "V3")])
    #show_dag(dag)
    model = StructuralCausalModel(dag)
    domains = dict(V1=[0,1],V2=[0,1],V3=[0,1],V4=[0,1], U1=[0,1,2,3],U2=[0,1,2,3],U3=[0,1,2,3]) 
    bc.randomUtil.seed(1)
    model.fill_random_factors(domains)
    return model


if __name__ == "__main__":
    import logging, sys
    import networkx as nx
    import bcause.util.domainutils as dutils
    import bcause.util.graphutils as gutils

    log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'
    # logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format, datefmt='%Y%m%d_%H%M%S')

    m = define_model0()
    import numpy as np
    np.random.seed(407)
    data = m.sample(1000, as_pandas=True)[m.endogenous] # \mathcal{D}
    #data = data.append(dict(Y=0, X="x1", U="u1"), ignore_index=True)

    for i in range(5):
        gl = GradientLikelihood(m.randomize_factors(m.exogenous, allow_zero=False), tol=0.000001)
        gl.run(data)

        #print the resulting model
        print(gl.model.factors["U"])


