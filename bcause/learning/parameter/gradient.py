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


    def negative_log_likelihood(raw_params, py_x, fy):
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
        constrained_params = transform_params(raw_params) # constrained_params are all positive and sum up to 1
        neg_log_lkh = .0
        for x in domains['X']:
            for y in domains['Y']:
                N_xy = py_x.values_dict[(('X', x), ('Y', y))]
                sum_lkh_xy = .0
                for i_u, u in enumerate(domains['U']):
                    sum_lkh_xy += fy.values_dict[(('X', x), ('U', u), ('Y', y))] * constrained_params[i_u]
                neg_log_lkh -= N_xy * np.log(sum_lkh_xy)
        #print(f'{neg_log_lkh=} {constrained_params=}')
        return neg_log_lkh


    def transform_params(raw_params):
        """
        Transform the raw parameters to satisfy the constraints (all positive and sum up to 1).
        """
        # Apply the exponential function to ensure positivity
        positive_params = np.exp(raw_params)
    
        # Normalize the parameters to sum up to 1
        normalized_params = positive_params / np.sum(positive_params)
        
        return normalized_params


    def inverse_transform_params(constrained_params):
        """
        Inverse transform the constrained parameters to obtain the raw parameters (a vector of floats).
        """
        # Apply the log function to reverse the exponential transformation
        raw_params = np.log(constrained_params)
        
        return raw_params


    def maximum_likelihood_estimation(initial_params, py_x, fy):
        """
        Perform maximum likelihood estimation (MLE) for a given model and data starting at initial_params.

        Parameters:
            initial_params (array-like): The initial parameter values for the optimization algorithm.

        Returns:
            dict: A dictionary containing the estimated parameters and additional information from the optimization algorithm.
        """

        trajectory = []  # saves each iteration of the optimization process
        def callback(xk):
            trajectory.append(transform_params(xk)) # save the constrainted parameters

        # Perform optimization using scipy's minimize function
        result = minimize(negative_log_likelihood, inverse_transform_params(initial_params), 
                          args = (py_x, fy), tol = 1e-3, callback = callback) # default: method='BFGS'

        # Transform the estimated raw parameters to the constrained parameters
        estimated_params = transform_params(result.x)
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
        domains = self._prior_model.get_domains(self.trainable_vars)

        dirich_distr = [1.0] * len(domains['U'])
        initial_params = np.random.dirichlet(dirich_distr, 1) # 1 (vector) sample u_0 such that u_0i > 0 and sum(u_0i) = 1
        result = maximum_likelihood_estimation(initial_params, py_x, fy) # TODO: rename to avoid MLE in the name
        new_probs = result['params']
        self._update_model(new_probs)
        #trajectories.append(result['trajectory']) # TODO: store also the trajectory



if __name__ == "__main__":
    import logging, sys

    log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'

    # logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format, datefmt='%Y%m%d_%H%M%S')

    import networkx as nx

    dag = nx.DiGraph([("Y", "X"), ("V", "Y"), ("U", "X")])
    domains = dict(X=["x1", "x2"], Y=[0, 1], U=["u1", "u2", "u3", "u4"], V=["v1", "v2"]) # TODO: why Y=[0, 1]? Why not Y=["y1", "y2"]?

    import bcause.util.domainutils as dutils
    import bcause.util.graphutils as gutils

    antecedents = gutils.relevat_vars(dag, "Y") # get antecedents of "Y" + "Y"
    domy = dutils.subdomain(domains, *antecedents) # get domains of antecedents
    fy = DeterministicFactor(domy, right_vars=["V"], values=[1, 0])

    domx = dutils.subdomain(domains, *gutils.relevat_vars(dag, "X"))

    values = ["x1", "x1", "x2", "x1", "x1", "x1", "x2", "x1"]
    fx = DeterministicFactor(domx, left_vars=["X"], values=values)

    domv = dutils.subdomain(domains, "V")
    pv = MultinomialFactor(domv, values=[.1, .9])

    domu = dutils.subdomain(domains, "U")
    pu = MultinomialFactor(domu, values=[.2, .2, .1, .5])

    m = StructuralCausalModel(dag, [fx, fy, pu, pv], cast_multinomial=True)

    data = m.sample(1000, as_pandas=True)[m.endogenous]
    #data = data.append(dict(Y=0, X="x1", U="u1"), ignore_index=True)

    gl = GradientLikelihood(m)
    gl.run(data,max_iter=3)

    # print the model evolution
    for model_i in gl.model_evolution:
        print(model_i.get_factors(*model_i.exogenous))


    #print the resulting model
    print(gl.model)
