import pandas as pd
import numpy as np
from scipy.optimize import minimize

from bcause.factors import MultinomialFactor, DeterministicFactor
from bcause.factors.mulitnomial import random_multinomial
from bcause.learning.parameter import IterativeParameterLearning
from bcause.models.cmodel import StructuralCausalModel


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
        self._trainable_vars = self.trainable_vars or list(data.columns[data.isna().any()]) # TODO: a bit confusing what't the difference between self._trainable_vars and self.trainable_vars
                                                                                            #       Moreover, it looks they are equal. (if one is changed, the second one also changes)

        print(f"trainable: {self.trainable_vars}")

        for v in self._trainable_vars:
            # check exogenous and completely missing
            if not self.prior_model.is_exogenous(v):
                raise ValueError(f"Trainable variable {v} is not exogenous")

            if (~data[v].isna()).any():
                raise ValueError(f"Trainable variable {v} is not completely missing")

        # save the dataset
        self._data = data


    def negative_log_likelihood(self, raw_params, py_x, fy):
        """
        Calculate the negative log-likelihood for the given raw (unconstrainted) parameters.

        Parameters:
            raw_params:     unconstrainted parameters that are being optimized
            py_x:           P(Y|X)
            fy:             f_y
        Returns:
            float: evaluation of the objective function, i.e, negative ML
        """
        # TODO: can be subtantially optimized
        constrained_params = self.transform_params(raw_params) # constrained_params are all positive and sum up to 1

        # TODO: add here this
        neg_log_lkh = -self._prior_model.log_likelihood(data, variables=fy.left_vars) # TODO: but the parameters are NOT an argument here!!!

        #neg_log_lkh = .0
        #for x in domains['X']:
        #    for y in domains['Y']:
        #        N_xy = py_x.values_dict[(('X', x), ('Y', y))]
        #        sum_lkh_xy = .0
        #        for i_u, u in enumerate(domains['U']):
        #            sum_lkh_xy += fy.values_dict[(('X', x), ('U', u), ('Y', y))] * constrained_params[i_u]
        #        neg_log_lkh -= N_xy * np.log(sum_lkh_xy)
        ##print(f'{neg_log_lkh=} {constrained_params=}')
        return neg_log_lkh


    def transform_params(self, raw_params):
        """
        Transform the raw parameters to satisfy the constraints (all positive and sum up to 1).
        """
        # Apply the exponential function to ensure positivity
        positive_params = np.exp(raw_params)
    
        # Normalize the parameters to sum up to 1
        normalized_params = positive_params / np.sum(positive_params)
        
        return normalized_params


    def inverse_transform_params(self, constrained_params):
        """
        Inverse transform the constrained parameters to obtain the raw parameters (a vector of floats).
        """
        # Apply the log function to reverse the exponential transformation
        raw_params = np.log(constrained_params)
        
        return raw_params


    def maximum_likelihood_estimation(self, initial_params, py_x, fy):
        """
        Perform maximum likelihood estimation (MLE) for a given model and data starting at initial_params.

        Parameters:
            initial_params (array-like): The initial parameter values for the optimization algorithm.

        Returns:
            dict: A dictionary containing the estimated parameters and additional information from the optimization algorithm.
        """

        trajectory = []  # saves each iteration of the optimization process
        def callback(xk):
            trajectory.append(self.transform_params(xk)) # save the constrainted parameters

        # Perform optimization using scipy's minimize function
        result = minimize(self.negative_log_likelihood, self.inverse_transform_params(initial_params), 
                          args = (py_x, fy), tol = 1e-3, callback = callback) # default: method='BFGS'

        # Transform the estimated raw parameters to the constrained parameters
        estimated_params = self.transform_params(result.x)
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


    def step(self):
        # one gradient descent (MLE) process
        m = self._prior_model
        domains = m.get_domains(self.trainable_vars)
        for U in m.exo_ccomponents: # we do MLE separately for each c-component (due to the theory (d-separation?))
            assert len(U) == 1, f'Quasi-Markovianity violated! ({len(U)=})'
            U = U.pop() # get this only element of U
            dirich_distr = [1.0] * len(m.domains[U])
            initial_params = np.random.dirichlet(dirich_distr, 1) # 1 (vector) sample u_0 such that u_0i > 0 and sum(u_0i) = 1
            children_of_U = list(m.graph.succ[U].keys())[0] # TODO: rafa, do we have a tool that returns the children of an exogeneous node?
            fy = m.factors[children_of_U]
            if len(fy.right_vars) == 1:
                py_x = None # 
            elif len(fy.right_vars) > 1:
                py_x = fy
            else: # len(fy.right_vars) == 0:
                raise Exception('Unsupported')
            results = self.maximum_likelihood_estimation(initial_params, py_x, fy) # TODO: rename to avoid MLE in the name
            new_probs = results['params'] 
            self._update_model(new_probs) # TODO: the data structure of new_probs is unclear
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


if __name__ == "__main__":
    import logging, sys
    import networkx as nx
    import bcause.util.domainutils as dutils
    import bcause.util.graphutils as gutils

    log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'
    # logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format, datefmt='%Y%m%d_%H%M%S')

    # basic definition & paper2code mapping
    DAG = nx.DiGraph([("X1", "X2"), ("U1", "X1"), ("U2", "X2")]) # definition of \mathcal{G}
    if 0:
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(DAG)  # Position nodes using a spring layout algorithm
        nx.draw(DAG, pos, with_labels=True, node_size=1000, node_color="lightblue", font_size=10, font_color="black", arrowsize=20)
        plt.show()
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

    data = m.sample(10, as_pandas=True)[m.endogenous] # \mathcal{D}
    #data = data.append(dict(Y=0, X="x1", U="u1"), ignore_index=True)

    gl = GradientLikelihood(m)
    gl.run(data,max_iter=3)

    # print the model evolution
    for model_i in gl.model_evolution:
        print(model_i.get_factors(*model_i.exogenous))


    #print the resulting model
    print(gl.model)


