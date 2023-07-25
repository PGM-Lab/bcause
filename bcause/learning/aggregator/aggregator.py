from abc import abstractmethod, ABC
import multiprocessing
from multiprocessing import Process

from bcause.factors import DeterministicFactor, MultinomialFactor
from bcause.learning.parameter import ParameterLearning
from bcause.learning.parameter.expectation_maximization import ExpectationMaximization
from bcause.models.cmodel import StructuralCausalModel
from bcause.models.pgmodel import PGModel


class ModelAggregator(ABC):

    def __init__(self, parallel = False):
        self._parallel = parallel
        self.reset()

    def reset(self):
        print("reset models")
        self._final_models = []
        self._generated_models = []
        self._learn_objects = []

    @property
    def learn_objects(self):
        return self._learn_objects
    @property
    def models(self):
        return self._final_models

    def generate(self, num_models : int):
        print(self._parallel)
        if num_models<1: raise ValueError(f"Wrong number of models: {num_models}")
        # if not self._parallel:
        self._generated_models += [self._single_generate(i) for i in range(num_models)]
        # else:
        #
        #     def thread(self, i):
        #         return self._single_generate(i)
        #
        #     processes = [Process(target=thread, args=(self,i,)) for i in range(num_models)]
        #     self._generated_models += [p.start() for p in processes]
        #     for p in processes: p.join()

    @abstractmethod
    def _single_generate(self, i):
        pass

    @abstractmethod
    def merge(self):
        pass

    def run(self, num_models):
        self.generate(num_models)
        self.merge()


class SimpleModelAggregator(ModelAggregator):
    def merge(self):
        self._final_models = self._generated_models

class ModelAggregatorEM(ModelAggregator):
    def _single_generate(self, i):
        em = ExpectationMaximization(self._model.randomize_factors(self._trainlable_vars, allow_zero=False), trainable_vars=self._trainlable_vars)
        em.run(self._data, max_iter=self._max_iter)
        self._learn_objects.append(em)
        return em.model

class SimpleModelAggregatorEM(SimpleModelAggregator, ModelAggregatorEM):

    def __init__(self, model, data, trainable_vars=None, max_iter=200, parallel=False):
        self._model = model
        self._data = data
        self._trainlable_vars = trainable_vars or model.exogenous
        self._max_iter = max_iter
        super().__init__(parallel=parallel)





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

    agg = SimpleModelAggregatorEM(m, data, max_iter=10, parallel=True)

    agg.run(2)

    for l in agg.learn_objects:
        print(len(l.model_evolution))

    for m in agg.models:
        print(m.factors)