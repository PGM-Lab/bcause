"""
Compute the PMF of exogenous variables given data for endogenous variable using
gradient descent method for negative likelihood objective function

TODO:
1) to check corectness, implement Theorem 1 from Causal EM paper
2) render canonical SEs (according to Rafa's mail from 3.10)
With RAFA: (it about effectivity - if some approach is usual, let's use it for the following)
3) which example models to use
4) which metrics to use for a comparision with EM
5) how to present the results (visuals)

TODO2:
- what can be accessed from outside (e.g. tol) should be in the constructor
- to test that gradient likelihood is correct, check the likelihood of the optimal model with 
  the likelihood of the true model
"""

from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import itertools
import random 

import bcause as bc
from bcause.factors import MultinomialFactor, DeterministicFactor
from bcause.factors.mulitnomial import random_multinomial   # TODO: "mulitnomial"
from bcause.learning.parameter import IterativeParameterLearning
from bcause.models.cmodel import StructuralCausalModel
from bcause.util.domainutils import assingment_space, state_space


class GradientLikelihood(IterativeParameterLearning):
    '''
    This class implements a method for running a single optimization of the exogenous variables in an SCM.
    '''

    def __init__(self, prior_model: StructuralCausalModel, trainable_vars: list = None):
        self._prior_model = prior_model
        self._trainable_vars = trainable_vars


    def initialize(self, data: pd.DataFrame, **kwargs):
        self._model = self.prior_model.copy()
        self._process_data(data)

    def _stop_learning(self) -> bool:
        # todo: implement a function that check if the process is stopped
        pass

    def _calculate_updated_factors(self, **kwargs) -> dict[MultinomialFactor]:
        return {U:self._updated_factor(U) for U in self.trainable_vars}

    def _updated_factor(self, U) -> MultinomialFactor:

        # todo: replace this code, now it returns a random distribution while it should return the result of a step in the gradient ascent
        f = random_multinomial({U:self._model.domains[U]})
        return f

    def _process_data(self, data: pd.DataFrame):
        # add missing variables
        missing_vars = [v for v in self.prior_model.variables if v not in data.columns]
        for v in missing_vars: data[v] = float("nan")

        # Set as trainable variables those with missing
        self._trainable_vars = self.trainable_vars or list(data.columns[data.isna().any()]) 
                                                                                            

        print(f"trainable: {self.trainable_vars}")

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
                sum_lkh_bmvbmy = .0
                for u in P_bmVbmYu[id_v][id_y]:
                    sum_lkh_bmvbmy += P_bmVbmYu[id_v][id_y][u] * theta[u]
                log_likelihood += N_bmVbmY[id_v][id_y] * np.log(sum_lkh_bmvbmy)

        #log_likelihood = np.sum(counts * np.log(np.sum(theta * pa_probs, axis=1)))
        return -log_likelihood  # We negate the value to convert maximization to minimization


    def constraint(self,theta):
        return np.sum(theta) - 1  # The probabilities should sum up to 1


    def maximum_likelihood_estimation(self, initial_params, N_bmVbmY, P_bmVbmYu):
        """
        TODO: DO NOT NAME IT "maximum likelihood estimation (MLE)"
        Perform maximum likelihood estimation (MLE) for a given model and data starting at initial_params.

        Parameters:
            initial_params (array-like): The initial parameter values for the optimization algorithm.

        Returns:
            dict: A dictionary containing the estimated parameters and additional information from the optimization algorithm.
        """

        trajectory = []  # saves each iteration of the optimization process
        def callback(xk):
            trajectory.append(xk) # save the parameters

        # Constraint dictionary
        con = {'type': 'eq', 'fun': self.constraint}

        # Perform optimization using scipy's minimize function
        result = minimize(self.negative_log_likelihood, initial_params, 
                          args = (N_bmVbmY, P_bmVbmYu), constraints=con, 
                          bounds=[(0, 1)]*initial_params.size,
                          tol = 1e-3, callback = callback) # default: method='SLSQP'

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
        print(f'{result.nit=}') # number of iterations
        return estimation_results


    #def multi_mle(py_x, fy, num_runs):
    #    solutions = []
    #    trajectories = []
    #    dirich_distr = [1.0] * len(domains['U'])
    #    for _ in range(num_runs):
    #        initial_params = np.random.dirichlet(dirich_distr, 1) # 1 (vector) sample u_0 such that u_0i > 0 and sum(u_0i) = 1
    #        result = maximum_likelihood_estimation(initial_params, py_x, fy)
    #        solutions.append(result['params'])
    #        trajectories.append(result['trajectory'])
    #    return solutions, trajectories

    def prepare_MLE(self):
        # compute counts and pa_probs
        pass

# TODO: do not overwrite method step
# rather, move this to self._updated_factor()
    def step(self):
        # one gradient descent (MLE) process
        m = self._prior_model
        domains = m.get_domains(self.trainable_vars)
        for U in m.exo_ccomponents: # we do MLE separately for each c-component 
            assert len(U) == 1, f'Quasi-Markovianity violated! ({len(U)=})' # remove this, we assume the model is correct
            U = U.pop() # get this only element of U
            if U != 'U':
                continue # DEBUG: skip this trivial case for now
            #dirich_distr = [1.0] * len(m.domains[U])
            #initial_params = np.random.dirichlet(dirich_distr, 1) # 1 (vector) sample u_0 such that u_0i > 0 and sum(u_0i) = 1
            initial_params = m.factors[U].values
            
            ### get the quantities named as in our paper ###
            # endogenous children
            bmV = m.get_edogenous_children(U) 

            # get the endogenous parents
            bmY = list(itertools.chain(*[m.get_edogenous_parents(V) for V in bmV]))

            # remove the variables in V
            bmY = [V for V in bmY if V not in bmV]

            # get the joint domains of Y union V
            dom_bmYbmV = m.get_domains(bmY+bmV)

            # get all possible assignments in a domain
            #assingment_space(dom_bmYbmV)

            # get all possible states in a domain
            #state_space(dom_bmYbmV)
            
            data = self._data

            N_bmVbmY = defaultdict(lambda : {}) # data counts
            P_bmVbmYu = defaultdict(lambda : {}) # SEs probabilities
            for bmv in assingment_space(m.get_domains(bmV)):
                id_v = tuple(bmv.items())
                for bmy in assingment_space(m.get_domains(bmY)):
                    id_y = tuple(bmy.items())
                    filtered_data = data[
                        (data[list(bmv.keys())] == list(bmv.values())).all(axis=1) &
                        (data[list(bmy.keys())] == list(bmy.values())).all(axis=1)
                    ]
                    N = len(filtered_data)
                    N_bmVbmY[id_v][id_y] = N

                    P_bmVbmYu[id_v][id_y] = {}
                    for u in m.get_domains(U)[U]:
                        # compute \prod_{V \in \bmV} P(v|Pa(V))
                        P_v_pav = 1
                        for V in bmV:
                            P_v_pav_key = []
                            for var in m.factors[V].variables:
                                if var in m.exogenous:
                                    key = (var, u)
                                elif var in bmV:
                                    key = (var, bmv[var])
                                elif var in bmY:
                                    key = (var, bmy[var])
                                else:
                                    raise Exception(f'Unclassfiable variable! ({var})')
                                P_v_pav_key.append(key)
                            if m.factors[V].values_dict[tuple(P_v_pav_key)] == 0: # TODO: assure that values_dict contain Integer values!
                                P_v_pav = 0
                                break # once we got 0, the product must be zero, so we finish immediately
                        P_bmVbmYu[id_v][id_y][u] = P_v_pav

            self.prepare_MLE() # TODO: strcit to vyse sem
            results = self.maximum_likelihood_estimation(initial_params, N_bmVbmY, P_bmVbmYu) # TODO: rename to avoid MLE in the name
            new_probs = {U : MultinomialFactor(m.get_domains(U), values=results['params'])}
            self._update_model(new_probs) 
            #trajectories.append(result['trajectory']) # TODO: store also the trajectory


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
    domains = dict(X=["x1", "x2"], Y=["y1","y2"], U=[0, 1, 2, 3], V=["v1", "v2"])
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

    gl = GradientLikelihood(m)
    gl.run(data,max_iter=1)

    # # print the model evolution
    # for model_i in gl.model_evolution:
    #     print(model_i.get_factors(*model_i.exogenous))
    #

    #print the resulting model
    print(gl.model)


