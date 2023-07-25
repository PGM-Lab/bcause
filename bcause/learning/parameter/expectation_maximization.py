from abc import abstractmethod

import pandas as pd

from bcause.factors import MultinomialFactor, DeterministicFactor
from bcause.inference.elimination.variableelimination import VariableElimination
from bcause.learning.parameter import IterativeParameterLearning
from bcause.models.cmodel import StructuralCausalModel
from bcause.models.pgmodel import DiscreteDAGModel
from bcause.util.datadeps import DataDepAnalysis

from bcause.util import domainutils as dutils


class AbastractExpectationMaximization(IterativeParameterLearning):
    @abstractmethod
    def _expectation(self, **kwargs):
        pass

    @abstractmethod
    def _maximization(self, **kwargs):
        pass

    def initialize(self, data: pd.DataFrame, **kwargs):
        self._model = self.prior_model.copy()
        self._process_data(data)

    def _calculate_updated_factors(self):
        counts = self._expectation()
        return self._maximization(counts)

    def _get_pseudocounts_dict(self) -> dict:
        def get_pcounts(v):
            pa = self.prior_model.get_parents(v)
            dom = dutils.subdomain(self.prior_model.domains, *([v] + pa))
            return MultinomialFactor(dom, values=0)

        return {v: get_pcounts(v) for v in self.trainable_vars}


class ExpectationMaximization(AbastractExpectationMaximization):
    def __init__(self, prior_model: DiscreteDAGModel, trainable_vars: list = None,
                 inference_method=VariableElimination):
        self._prior_model = prior_model
        self._trainable_vars = trainable_vars
        self._inference_method = inference_method

    def _get_obs_counts(self, target):
        obs_blanket = self._datadeps.get_minimal_obs_blanket(target)
        return [(obs, len(self._data.loc[self._data[obs.keys()].isin(obs.values()).all(axis=1), :])) for obs in obs_blanket]

    def _expectation(self, **kwargs):
        self._inf = self._inference_method(self._model, preprocess_flag=False)
        pcounts = self._get_pseudocounts_dict()
        for v in self.trainable_vars:

            for obs, c in self._get_obs_counts(v):
                relevant = [v] + self._prior_model.get_parents(v)
                hidden = [x for x in relevant if x not in obs]
                exp_counts = self._inf.query(target=hidden, evidence=obs) * c
                pcounts[v] = pcounts[v] + exp_counts

        return pcounts

    def _maximization(self, pcounts, **kwargs):
        new_probs = dict()
        for v in self.trainable_vars:
            joint_counts = pcounts[v]
            new_probs[v] = joint_counts / (joint_counts.marginalize(v))

        return new_probs

    def _process_data(self, data: pd.DataFrame):
        # add missing variables
        missing_vars = [v for v in self.prior_model.variables if v not in data.columns]
        for v in missing_vars: data[v] = float("nan")

        # Set as trainable variables those with missing
        self._trainable_vars = self.trainable_vars or list(data.columns[data.isna().any()])

        self._datadeps = DataDepAnalysis(self.model.graph, data)
        self._variables = list(data.columns)
        self._data = data

    def _stop_learning(self) -> bool:
        # todo: add stopping criteria
        return False


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

    print(m)
    em = ExpectationMaximization(m.randomize_factors(m.exogenous, allow_zero=False))
    em.run(data, max_iter=4)

    print(em.prior_model)

    print(len(em.model_evolution))