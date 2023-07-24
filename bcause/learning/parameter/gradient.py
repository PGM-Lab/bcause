import pandas as pd

from bcause.factors import MultinomialFactor, DeterministicFactor
from bcause.factors.mulitnomial import random_multinomial
from bcause.learning.parameter import IterativeParameterLearning
from bcause.models.cmodel import StructuralCausalModel


class GradientLikelihood(IterativeParameterLearning):
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

if __name__ == "__main__":
    import logging, sys

    log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'

    # logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format, datefmt='%Y%m%d_%H%M%S')

    import networkx as nx

    dag = nx.DiGraph([("Y", "X"), ("V", "Y"), ("U", "X")])
    domains = dict(X=["x1", "x2"], Y=[0, 1], U=["u1", "u2", "u3", "u4"], V=["v1", "v2"])

    import bcause.util.domainutils as dutils
    import bcause.util.graphutils as gutils

    domy = dutils.subdomain(domains, *gutils.relevat_vars(dag, "Y"))
    fy = DeterministicFactor(domy, right_vars=["V"], values=[1, 0])

    domx = dutils.subdomain(domains, *gutils.relevat_vars(dag, "X"))

    values = ["x1", "x1", "x2", "x1", "x1", "x1", "x2", "x1"]
    fx = DeterministicFactor(domx, left_vars=["X"], values=values)

    domv = dutils.subdomain(domains, "V")
    pv = MultinomialFactor(domv, values=[.1, .9])

    domu = dutils.subdomain(domains, "U")
    pu = MultinomialFactor(domu, values=[.2, .2, .1, .5])

    m = StructuralCausalModel(dag, [fx, fy, pu, pv], cast_multinomial=True)

    data = m.sample(10000, as_pandas=True)[m.endogenous]
    #data = data.append(dict(Y=0, X="x1", U="u1"), ignore_index=True)

    gl = GradientLikelihood(m)
    gl.run(data,max_iter=3)

    # print the model evolution
    for model_i in gl.model_evolution:
        print(model_i.get_factors(*model_i.exogenous))


    #print the resulting model
    print(gl.model)
