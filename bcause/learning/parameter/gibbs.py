"""
Compute the PMF of exogenous variables given data for endogenous variable using
Gibbs sampling
"""

from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import dirichlet
import itertools
import random
from typing import Dict, List, Tuple, Any


import bcause as bc
from bcause.factors import MultinomialFactor, DeterministicFactor
from bcause.factors.mulitnomial import random_multinomial   # TODO: mulitnomial -> multinomial
from bcause.inference.causal.multi import CausalMultiInference
from bcause.inference.probabilistic.elimination import VariableElimination
from bcause.learning.parameter import IterativeParameterLearning
from bcause.models.cmodel import StructuralCausalModel
from bcause.util.domainutils import assingment_space, state_space # TODO: assingment_space -> assignment_space
from bcause.util.graphutils import dcon_nodes

class GibbsSampling(IterativeParameterLearning):
    '''
     This class implements a method for running a single optimization of the exogenous variables in a SCM.
     '''

    def __init__(self, prior_model: StructuralCausalModel, trainable_vars: list = None, alpha: dict = None):
        self._prior_model = prior_model.fix_numeric_domains()
        self._trainable_vars = trainable_vars
        self._alpha = alpha
        self._is_prior = True

    def initialize(self, data: pd.DataFrame, **kwargs):
        self._model = self.prior_model.copy()
        self._process_data(data)
        # Create count table from data
        self._create_frequency_tables(data)

    def _stop_learning(self) -> bool:
        pass

    def _calculate_updated_factors(self, **kwargs) -> dict[MultinomialFactor]:
        return {U: self._updated_factor(U) for U in self.trainable_vars}

    def _updated_factor(self, U) -> MultinomialFactor:
        m = self._model

        # Isolate target variable
        ve = VariableElimination(m)
        model_adjusted = ve.query(U, conditioning=list(dcon_nodes(m.graph,U,m.endogenous)))

        if self._is_prior == True:
            if self._alpha is None:
                self._set_non_informative_alpha()
            theta = dirichlet.rvs(self._alpha[U])[0]
            self._is_prior = False
        else:
            # Get cpt, merge with data events and sample U
            cpt = self._get_conditional_probability_table(model_adjusted)
            cpt_count = cpt.merge(self._frequency_tables[U], on=model_adjusted.right_vars, how='left').fillna(0)
            samples_u = np.concatenate(cpt_count.apply(lambda row: np.random.choice(model_adjusted.left_domain[U],
                                                                                  size=row['count'], p=row['Probabilities']),axis=1).to_list())
            # Get posterior
            counts_u = np.bincount(samples_u)
            beta = [int(a + c) for a, c in zip(self._alpha[U], counts_u)]

            # sample the theta and set it to the model
            theta = dirichlet.rvs(beta)[0]

        f =  MultinomialFactor({U: m.domains[U]}, theta)
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

    def _create_frequency_tables(self, data:pd.DataFrame):
        # Count the number of times an event appear in the data
        frequency_tables = { U: data.groupby(list(dcon_nodes(self._model.graph,U,self._model.endogenous))).size().reset_index(name='count') for U in self.trainable_vars }
        self._frequency_tables = frequency_tables

    def _get_conditional_probability_table(self,model) -> pd.DataFrame:
        # Get the conditional probability table of the exo given endo
        cpt = pd.DataFrame(list(itertools.product(*model.right_domain.values())),
                             columns=model.right_domain.keys())
        cpt['Probabilities'] = list(map(lambda obs: model.R(**obs).values, cpt.to_dict(orient='records')))
        cpt[cpt.columns.difference(['Probabilities'])] = cpt[cpt.columns.difference(['Probabilities'])]
        return cpt

    def _set_non_informative_alpha(self):
        self._alpha = {U: np.ones(len(self._model.domains[U])) for U in self._trainable_vars}

if __name__ == "__main__":
    import logging, sys

    log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'

    # logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format, datefmt='%Y%m%d_%H%M%S')


    # Nota: probar también con modelos semi-markovianos

    #m = StructuralCausalModel.read("./models/literature/pearl_small.bif")
    m = StructuralCausalModel.read("./models/modelTest_SM.bif")
    #data = pd.read_csv("./models/literature/pearl_small.csv")
    data = pd.read_csv(
        "/Users/antoniogonzalezalves/Library/CloudStorage/GoogleDrive-aga081@ual.es/Mi unidad/random_boolean_data.csv")

    dic = my_dict = {
    'U': [3, 2, 1, 4],
    'V': [0.3, 0.7] }

    gs = GibbsSampling(m, alpha=dic)
    gs.run(data, max_iter=100)

    import matplotlib.pyplot as plt

    U_store = np.empty((0,4))
    V_store = np.empty([0,2])

    Q = []

    # print the model evolution
    for model_i in gs.model_evolution:
        inf = CausalMultiInference([model_i])
        q = inf.prob_sufficiency("T","S", true_false_cause=(1,0), true_false_effect=(1,0))[0]
        Q.append(q)


    plt.hist(Q[10:], density=True)
    plt.xlim(0, 1)
    plt.show()

        #print(q)




    #
    #
    #     U_store = np.vstack([U_store,model_i.get_factors(*model_i.exogenous)[0].values])
    #     V_store = np.vstack([V_store,model_i.get_factors(*model_i.exogenous)[1].values])
    #     #print(model_i.get_factors(*model_i.exogenous))
    #
    # fig = plt.figure(figsize=(10, 7))
    # # Creating plot
    # bp = plt.boxplot(U_store)
    # plt.ylim(-0.05, 1.05)
    # plt.yticks(np.arange(-0.1, 1.1, 0.1))
    # plt.grid(axis='y')
    # # show plot
    # plt.show()
