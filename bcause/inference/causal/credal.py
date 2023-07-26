from typing import Callable

import numpy as np

from bcause.factors import DeterministicFactor, MultinomialFactor
from bcause.factors.imprecise import IntervalProbFactor
from bcause.inference.causal.causal import CausalInference
from bcause.inference.inference import Inference
from bcause.learning.aggregator.aggregator import SimpleModelAggregatorEM
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


    def compile(self, *args, **kwargs) -> Inference:
        if len(self._models)<1: raise ValueError("Required at least 1 precise model")
        self._model = self._models[0]
        self._causal_inf = [self._causal_inf_fn(m) for m in self._models]
        self._compiled = False
        return self

    def query(self, target, do, evidence=None, counterfactual=False, targets_subgraphs = None):
        if not self._compiled: self.compile()
        return self._process_output([inf.query(target,do,evidence,counterfactual,targets_subgraphs) for inf in self._causal_inf])

    def _process_output(self, result, obs=None):

        if obs is not None:
            if isinstance(result, list):
                result = [r.get_value(**obs) for r in result]
            else:
                result = result.get_value(**obs)

        if self._interval_result:
            if all(np.isscalar(r) for r in result):
                result = [min(result), max(result)]
            else:
                result = IntervalProbFactor.from_precise(result)

        return result


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

    print(m)
    em = ExpectationMaximization(m.randomize_factors(m.exogenous, allow_zero=False))

    agg = SimpleModelAggregatorEM(m, data, max_iter=10)
    agg.run(num_models=5)

    inf = CausalMultiInference(agg.models, interval_result=True).compile()

    print(inf.causal_query("X", do=dict(Y=0)))
    print(inf.counterfactual_query("X", do=dict(Y=0)))
    print(inf.prob_necessity("Y","X"))
