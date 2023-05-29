import math
from abc import ABC, abstractmethod

import pandas as pd

from bcause.factors import DeterministicFactor, MultinomialFactor
from bcause.models.cmodel import StructuralCausalModel
from bcause.models.pgmodel import DiscreteDAGModel
from bcause.util.datautils import to_counts

class ParameterLearning(ABC):
    @property
    def prior_model(self):
        return self._prior_model

    @property
    def model(self):
        return self._model

    @property
    def trainable_vars(self):
        return self._trainable_vars

class IterativeParameterLearning(ParameterLearning):

    def step(self, data:pd.DataFrame=None):
        if data is not None: self._process_data(data)
        return self._step()

    @property
    def step_args(self):
        return self._step_args

    @abstractmethod
    def _step(self, **kwargs):
        pass

    @abstractmethod
    def _process_data(self, data:pd.DataFrame=None):
        pass

    @abstractmethod
    def run(self, data:pd.DataFrame):
        pass

    @abstractmethod
    def initialize(self, data : pd.DataFrame, **kwargs):
        pass

    @abstractmethod
    def stop_inference(self) -> bool:
        pass

    def run(self, data: pd.DataFrame, max_iter: int = float("inf")):
        """
        This method performs a given number of optimization steps.
        Args:
            data: training data.
            max_iter: number of iterations. Default is None and runs util converge.

        Returns:

        """


        self.initialize(data)
        i = 0
        while i < max_iter:
            self.step()
            if self.stop_inference(): break
            i = i+1


class AbastractExpectationMaximization(IterativeParameterLearning):
    @abstractmethod
    def _expectation(self, **kwargs):
        pass

    @abstractmethod
    def _maximization(self, **kwargs):
        pass

    def initialize(self, data:pd.DataFrame, **kwargs):
        self._model = self.prior_model.copy()
        self. _process_data(data)


    def _step(self):
        counts = self._expectation()
        self._maximization(counts)

    def _get_pseudocounts_dict(self) -> dict:
        def get_pcounts(v):
            pa = self.prior_model.get_parents(v)
            dom = dutils.subdomain(self.prior_model.domains, *([v] + pa))
            return MultinomialFactor(dom, values=0)

        return {v:get_pcounts(v) for v in self.trainable_vars}


class ExpectationMaximization(AbastractExpectationMaximization):
    def __init__(self, prior_model : DiscreteDAGModel, trainable_vars:list = None):
        self._prior_model = prior_model
        self._step_args = None
        self._counts_as_factor = False
        self._trainable_vars = trainable_vars


#    d,c = list(self.step_args.items())[0]

    def _get_obs_counts(self):
        for d, c in self.step_args.items():
            yield dict(zip(self._variables, d)), c
            # todo: implement for factor-like counts


    def _expectation(self, **kwargs):         # todo: implement
        pcounts = self._get_pseudocounts_dict()
        import math
        for v in self.trainable_vars:
            for d,c in self._get_obs_counts():
                d = {v: s for v, s in d.items() if (type(s) == str and s != "nan") or (not math.isnan(s))}
                relevant = [v] + self._prior_model.get_parents(v)
                hidden = [x for x in relevant if x not in d]

                # P(hidden | d)
                # todo: implement posterior inference, with d-separation model cache,




    def _maximization(self, **kwargs):         # todo: implement
        pass

    def _process_data(self, data:pd.DataFrame):
        # add missing variables
        missing_vars = [v for v in self.prior_model.variables if v not in data.columns]
        for v in missing_vars: data[v] = float("nan")

        # Set as trainable variables those with missing
        self._trainable_vars = self.trainable_vars or list(data.columns[data.isna().any()])

        # set input data counts
        if self._counts_as_factor == True:
            self._step_args = to_counts(self.prior_model.domains, data)
        else:
            self._variables = list(data.columns)
            self._step_args = data.value_counts(dropna=False).to_dict()

    def stop_inference(self) -> bool:
        # todo: add stopping criteria
        return False






if __name__ == "__main__":

    import networkx as nx

    dag = nx.DiGraph([("Y" ,"X"), ("V" ,"Y"), ("U" ,"X")])
    domains = dict(X=["x1", "x2"], Y=[0, 1], U=["u1", "u2", "u3", "u4"], V=["v1", "v2"])

    import bcause.util.domainutils as dutils
    import bcause.util.graphutils as gutils



    domy = dutils.subdomain(domains, *gutils.relevat_vars(dag, "Y"))
    fy = DeterministicFactor(domy, right_vars = ["V"], values=[1, 0])

    domx = dutils.subdomain(domains, *gutils.relevat_vars(dag, "X"))

    # todo: check data shape for this
    values = ["x1", "x1", "x2", "x1","x1", "x1", "x2", "x1"]
    fx = DeterministicFactor(domx, left_vars = ["X"], values=values)

    domv = dutils.subdomain(domains, "V")
    pv = MultinomialFactor(domv, values= [.1, .9])

    domu = dutils.subdomain(domains, "U")
    pu = MultinomialFactor(domu, values= [.2, .2, .1, .5])

    m = StructuralCausalModel(dag, [fx, fy, pu, pv], cast_multinomial=True)

    m.builder(dag=dag, factors=m.factors)

    data = m.sample(10, as_pandas=True)[m.endogenous]

    em = ExpectationMaximization(m)

    em.initialize(data)
    self = em
    em.run(data, max_iter=4)


'''
class foo(object):
    pass
self = foo()

self.prior_model = m
data = m.sample(10, as_pandas=True)[m.endogenous]
data



data.loc[0, "V"] = None
data.fillna("nan")

sum(data.value_counts(dropna=False).to_dict().values())

missing_vars = [v for v in self.prior_model.variables if v not in data.columns]
for v in missing_vars:
    data[v] = float("nan")


counts = to_counts(domains, data)


domv = dutils.subdomain(domains, "V")
domv = dict(V=[True,False])
pv = MultinomialFactor(domv, data=[.1, .9])




list(data.columns[data.isna().any()])

import collections
isinstance(float("nan"), collections.Hashable)
isinstance(None, collections.Hashable)
'''