'''
Maximum likelihood estimation () of credal networks

Runs multiple MLE (function multi_mle()) of parameters in a predefined credal network in order to 
approximate the simplex representing all acceptable parameters of the network.
'''

import numpy as np
import pickle
from scipy.optimize import minimize

from bcause.util.plotutils import plot_3d
from dev.optimization import py_x, fy, domains
from dev.mle_aux import *

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


def multi_mle(py_x, fy, num_runs):
    solutions = []
    trajectories = []
    dirich_distr = [1.0] * len(domains['U'])
    for _ in range(num_runs):
        initial_params = np.random.dirichlet(dirich_distr, 1) # 1 (vector) sample u_0 such that u_0i > 0 and sum(u_0i) = 1
        result = maximum_likelihood_estimation(initial_params, py_x, fy)
        solutions.append(result['params'])
        trajectories.append(result['trajectory'])
    return solutions, trajectories


if __name__ == "__main__":
    LOAD_SOL = 0 # 0 - compute a solution, 1 - load the previously computed and saved solution
    sol_filename = 'results/solutions.dat' # CAREFUL: check that the folder 'results' exists!
    if LOAD_SOL:
        # just load previously stored solution
        with open(sol_filename, 'rb') as file:
            solutions, trajectories = pickle.load(file)
    else:
        # compute a new solution and save it 
        solutions, trajectories = multi_mle(py_x, fy, 5)
        with open(sol_filename, 'wb') as file:
            pickle.dump((solutions, trajectories), file)

    # plots
    if 1: # show solutions and MLE trajectories (and eventually save the figure)
        plot_3d(solutions, 
                save_path = None, #'results/traj_' + convert_to_filename(data),
                trajectories = trajectories)
    if 1: # plot evolution of the ML objectiove function 
        plot_lkh_evol(trajectories, save_path = None) #'results/evolution.png')

