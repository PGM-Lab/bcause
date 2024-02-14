from abc import ABC
from typing import Callable

import numpy as np

from bcause.factors import DeterministicFactor, MultinomialFactor
from bcause.factors.imprecise import IntervalProbFactor
from bcause.inference.causal.causal import CausalInference
from bcause.inference.inference import Inference
from bcause.learning.aggregator.aggregator import SimpleModelAggregatorEM, SimpleModelAggregatorGD
from bcause.learning.parameter.expectation_maximization import ExpectationMaximization
from bcause.models.cmodel import StructuralCausalModel


class CausalMultiInference(CausalInference):
    def __init__(self, models: list[StructuralCausalModel] = None, causal_inf_fn: Callable = None, interval_result=True):
        self.set_models(models or [])
        self._interval_result = interval_result

        if causal_inf_fn is None:
            from bcause.inference.causal.elimination import CausalVariableElimination
            causal_inf_fn = CausalVariableElimination
        self._causal_inf_fn = causal_inf_fn


    @property
    def models(self) -> list[StructuralCausalModel]:
        return self._models

    def set_models(self, models: list[StructuralCausalModel]):
        self._models = models or []
        self._compiled = False

    def add_models(self, models: list[StructuralCausalModel]):
        models = models or []
        self._models += models
        self._compiled = False

    def compile(self, *args, **kwargs) -> Inference:
        if len(self._models)<1: raise ValueError("Required at least 1 precise model")
        self._model = self._models[0]
        self._causal_inf = [self._causal_inf_fn(m) for m in self._models]
        self._compiled = True
        return self

    def query(self, target, do, evidence=None, counterfactual=False, targets_subgraphs = None):
        if not self._compiled: self.compile()
        return self._process_output([inf.query(target,do,evidence,counterfactual,targets_subgraphs) for inf in self._causal_inf])

    def _process_output(self, result, obs=None):

        if obs is not None:
            if isinstance(result, list):
                result = [r.get_value(**obs) for r in result]
            elif isinstance(result, IntervalProbFactor):
                result = result.restrict(**obs).values
            else:
                result = result.get_value(**obs)

        if self._interval_result:
            if all(np.isscalar(r) for r in result):
                result = [min(result), max(result)]
            else:
                result = IntervalProbFactor.from_precise(result)

        return result

    def set_interval_result(self, value:bool):
        self._interval_result = value




class CausalObservationalInference(ABC):
    @property
    def data(self):
        return self._data

class EMCC(CausalMultiInference, CausalObservationalInference):
    def __init__(self, model:StructuralCausalModel, data, causal_inf_fn: Callable = None, interval_result=True, max_iter=100, num_runs=10, parallel = False):
        self._data = data
        self._prior_model = model
        self._num_runs = num_runs
        self._max_iter = max_iter
        self._agg = None
        self._parallel = parallel
        super().__init__([], causal_inf_fn=causal_inf_fn, interval_result=interval_result)

    def compile(self, *args, **kwargs) -> Inference:
        self._agg = SimpleModelAggregatorEM(self._prior_model, self._data, max_iter=self._max_iter, parallel=self._parallel)
        self._agg.run(num_models=self._num_runs)
        self.set_models(self._agg.models)
        return super().compile()


    def compile_incremental(self, step_runs=1, *args, **kwargs) -> Inference:
        #for i in range(self._num_runs):
        while len(self.models)<self._num_runs:
            self._agg = SimpleModelAggregatorEM(self._prior_model, self._data, max_iter=self._max_iter, parallel=self._parallel)
            self._agg.run(num_models=step_runs)
            self.add_models(self._agg.models)
            yield super().compile()

class GDCC(CausalMultiInference, CausalObservationalInference):
    def __init__(self, model: StructuralCausalModel, data, causal_inf_fn: Callable = None, interval_result=True,
                 num_runs=10, tol=1e-3, max_iter = float("inf"), parallel=False):
        self._data = data
        self._prior_model = model
        self._num_runs = num_runs
        self._agg = None
        self._parallel = parallel
        self._tol = tol
        self._max_iter = max_iter
        super().__init__([], causal_inf_fn=causal_inf_fn, interval_result=interval_result)

    def compile(self, *args, **kwargs) -> Inference:
        self._agg = SimpleModelAggregatorGD(self._prior_model, self._data, tol=self._tol, max_iter=self._max_iter, parallel=self._parallel)
        self._agg.run(num_models=self._num_runs)
        self.set_models(self._agg.models)
        return super().compile()

    def get_model_evolution(self, index):
        if self._agg is not None:
            return self._agg.learn_objects[index].model_evolution

if __name__=="__main__":
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

    # inf = EMCC(m, data, num_runs=10, max_iter=3)
    # print(inf.causal_query("X", do=dict(Y=0)))
    # print(inf.counterfactual_query("X", do=dict(Y=0)))
    # print(inf.prob_necessity("Y","X"))



    inf = GDCC(m, data, num_runs=10)
    print(inf.causal_query("X", do=dict(Y=0)))
    print(inf.counterfactual_query("X", do=dict(Y=0)))
    print(inf.prob_necessity("Y","X"))


