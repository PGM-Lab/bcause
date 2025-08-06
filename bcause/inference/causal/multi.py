from abc import ABC
from typing import Callable

import numpy as np

from bcause.factors import DeterministicFactor, MultinomialFactor
from bcause.factors.imprecise import IntervalProbFactor
from bcause.inference.causal.causal import CausalInference, CausalObservationalInference
from bcause.inference.inference import Inference
from bcause.inference.probabilistic.elimination import VariableElimination
from bcause.learning.aggregator.aggregator import SimpleModelAggregatorEM, SimpleModelAggregatorGD, \
    SimpleModelAggregatorGibbs
from bcause.learning.parameter.expectation_maximization import ExpectationMaximization
from bcause.models.cmodel import StructuralCausalModel
from bcause.models.transform.combination import counterfactual_model
from bcause.util.arrayutils import min_max_iqr
import bcause.util.domainutils as dutils
from bcause.util.watch import Watch


class CausalMultiInference(CausalInference):
    def __init__(self, models: list[StructuralCausalModel] = None, causal_inf_fn: Callable = None, interval_result=True,
                 min_rating=0.0, calculate_rating = False, outliers_removal = True):
        self.set_models(models or [])
        self._interval_result = interval_result
        self._min_rating = min_rating
        self._outliers_removal = outliers_removal
        self._calculate_rating = calculate_rating

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

        if self._calculate_rating:
            for m in self._models: m.rating = m.ratio(self._data)
            self._causal_inf = [self._causal_inf_fn(m) for m in self._models if m.rating >= self._min_rating]
        else:
            self._causal_inf = [self._causal_inf_fn(m) for m in self._models]


        self._compiled = True
        return self

    def query(self, target, do=None, evidence=None, counterfactual=False, targets_subgraphs = None):
        do = do or dict()

        if not self._compiled: self.compile()
        results = [inf.query(target, do, evidence, counterfactual, targets_subgraphs)  for inf in self._causal_inf]
        if len(results)==0:
            return None
        return self._process_output(results)

    def _process_output(self, result, obs=None):

        if result is None:
            return None

        if obs is not None:
            if isinstance(result, list):
                result = [r.restrict(**obs).marginalize(*list(obs.keys())).values[0] for r in result]
            elif isinstance(result, IntervalProbFactor):
                result = result.restrict(**obs).marginalize(*list(obs.keys())).values[0]
            else:
                result = result.restrict(**obs).marginalize(*list(obs.keys())).values[0]




        if self._interval_result:
            if all(np.isscalar(r) for r in result):
                if not self._outliers_removal:
                    result = [min(result), max(result)]
                else:
                    result = list(min_max_iqr(result))
            else:
                result = IntervalProbFactor.from_precise(result, self._outliers_removal)

        return result

    def set_interval_result(self, value:bool):
        self._interval_result = value

    def prob_necessity(self, cause, effect, true_false_cause:tuple=None, true_false_effect:tuple=None):
        # PN: P(X_{Y=f} = f |X=t, Y=t)   Y->X

        #if not self._compiled: self.compile()
        # Determine the true and false states
        Tcause, Fcause = true_false_cause or dutils.identify_true_false(cause, self.model.domains[cause])
        Teffect, Feffect = true_false_effect or dutils.identify_true_false(effect, self.model.domains[effect])

        # Run the query
        interval_result = self._interval_result
        self._interval_result = False
        result = self.counterfactual_query(
            effect,
            do={cause: Fcause},
            evidence={cause: Tcause, effect: Teffect},
        )

        self._interval_result = interval_result

        return self._process_output(result, {effect+"_1": Feffect})

    def prob_sufficiency(self, cause, effect, true_false_cause:tuple=None, true_false_effect:tuple=None):
        # PS: P(X_{Y=t} = t |X=f, Y=f)   Y->X

        #if not self._compiled: self.compile()
        # Determine the true and false states
        Tcause, Fcause = true_false_cause or dutils.identify_true_false(cause, self.model.domains[cause])
        Teffect, Feffect = true_false_effect or dutils.identify_true_false(effect, self.model.domains[effect])

        # Run the query
        interval_result = self._interval_result
        self._interval_result = False
        result = self.counterfactual_query(
            effect,
            do={cause: Tcause},
            evidence={cause: Fcause, effect: Feffect}
        )
        self._interval_result = interval_result

        return self._process_output(result, {effect+"_1": Teffect})


    def prob_necessity_sufficiency(self, cause, effect, true_false_cause:tuple=None, true_false_effect:tuple=None):

        #if not self._compiled: self.compile()

        # Determine the true and false states
        Tcause, Fcause = true_false_cause or dutils.identify_true_false(cause, self.model.domains[cause])
        Teffect, Feffect = true_false_effect or dutils.identify_true_false(effect, self.model.domains[effect])

        # Run the query
        interval_result = self._interval_result
        self._interval_result = False
        result = self.counterfactual_query(
            [effect]*2,
            do=[{cause: Fcause}, {cause: Tcause}],
            targets_subgraphs=[1,2]
        )
        self._interval_result = interval_result

        return self._process_output(result, {effect + "_1": Feffect, effect + "_2": Teffect})


    def get_counterfactual_model(self, do, exclude_from_common=None, as_bnet=False):
        out = [counterfactual_model(m, do, exclude_from_common) for m in self.models]
        if as_bnet:
            return [m.to_bnet() for m in out]
        return out

    def query_on_twin_model(self, target, do, evidence=None, exclude_from_common=None):
        # Extract the twin models
        twin_models = self.get_counterfactual_model(do, exclude_from_common)

        # Create a new inference engine
        inf2 = CausalMultiInference(twin_models)
        return inf2.query(target, evidence)

class EMCC(CausalMultiInference, CausalObservationalInference):
    def __init__(self, model:StructuralCausalModel, data, causal_inf_fn: Callable = None, em_inf_fn: Callable = VariableElimination, interval_result=True, max_iter=100, num_runs=10, parallel = False, min_rating=0.0, calculate_rating=False, outliers_removal=True, random_init=True):
        self._data = data
        self._prior_model = model
        self._model = model
        self._num_runs = num_runs
        self._max_iter = max_iter
        self._agg = None
        self._parallel = parallel
        self._em_inf_fn = em_inf_fn
        self._random_init = random_init
        super().__init__([], causal_inf_fn=causal_inf_fn, interval_result=interval_result, min_rating=min_rating, calculate_rating=calculate_rating, outliers_removal=outliers_removal)

    def compile(self, *args, **kwargs) -> Inference:
        self._agg = SimpleModelAggregatorEM(self._prior_model, self._data, max_iter=self._max_iter, parallel=self._parallel, inference_method=self._em_inf_fn, random_init=self._random_init)
        self._agg.run(num_models=self._num_runs)
        self.set_models(self._agg.models)
        #self._model = self._models[0]
        return super().compile()


    def compile_incremental(self, step_runs=1, *args, **kwargs) -> Inference:
        #for i in range(self._num_runs):
        while len(self.models)<self._num_runs:
            self._agg = SimpleModelAggregatorEM(self._prior_model, self._data, max_iter=self._max_iter, parallel=self._parallel, inference_method=self._em_inf_fn, random_init=self._random_init)
            self._agg.run(num_models=step_runs)
            self.add_models(self._agg.models)
            yield super().compile()

class GDCC(CausalMultiInference, CausalObservationalInference):
    def __init__(self, model: StructuralCausalModel, data, causal_inf_fn: Callable = None, interval_result=True,
                 num_runs=10, tol=1e-7, max_iter = float("inf"), parallel=False, min_rating=0.0, calculate_rating=False, outliers_removal=True):
        self._data = data
        self._prior_model = model
        self._num_runs = num_runs
        self._agg = None
        self._parallel = parallel
        self._tol = tol
        self._max_iter = max_iter
        super().__init__([], causal_inf_fn=causal_inf_fn, interval_result=interval_result, min_rating=min_rating, calculate_rating=calculate_rating, outliers_removal=outliers_removal)

    def compile(self, *args, **kwargs) -> Inference:
        self._agg = SimpleModelAggregatorGD(self._prior_model, self._data, tol=self._tol, max_iter=self._max_iter, parallel=self._parallel)
        self._agg.run(num_models=self._num_runs)
        self.set_models(self._agg.models)
        self._model = self._models[0]
        return super().compile()
    
    def compile_incremental(self, step_runs=1, *args, **kwargs) -> Inference:
        #for i in range(self._num_runs):
        while len(self.models)<self._num_runs:
            self._agg = SimpleModelAggregatorGD(self._prior_model, self._data, tol=self._tol, max_iter=self._max_iter,
                                                parallel=self._parallel)
            self._agg.run(num_models=step_runs)
            self.add_models(self._agg.models)
            yield super().compile()

    def get_model_evolution(self, index):
        if self._agg is not None:
            return self._agg.learn_objects[index].model_evolution



class GibbsCausal(CausalMultiInference, CausalObservationalInference):
    def __init__(self, model:StructuralCausalModel, data, causal_inf_fn: Callable = None,  interval_result=True, num_runs=10, parallel = False, min_rating=0.0, calculate_rating=False, outliers_removal=True):
        self._data = data
        self._prior_model = model
        self._model = model
        self._num_runs = num_runs
        self._agg = None
        self._parallel = parallel
        super().__init__([], causal_inf_fn=causal_inf_fn, interval_result=interval_result, min_rating=min_rating, calculate_rating=calculate_rating, outliers_removal=outliers_removal)

    def compile(self, *args, **kwargs) -> Inference:
        self._agg = SimpleModelAggregatorGibbs(self._prior_model, self._data, parallel=self._parallel)
        self._agg.run(num_models=self._num_runs)
        self.set_models(self._agg.models)
        return super().compile()

    def compile_incremental(self, step_runs=1, *args, **kwargs) -> Inference:
        #for i in range(self._num_runs):
        while len(self.models)<self._num_runs:
            self._agg = SimpleModelAggregatorGibbs(self._prior_model, self._data, parallel=self._parallel)
            self._agg.run(num_models=step_runs)
            self.add_models(self._agg.models)
            yield super().compile()





if __name__=="__main__":
    import logging, sys

    log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'

    # logging.getLogger( __name__ ).basicConfig(level=logging.getLogger( __name__ ).DEBUG, stream=sys.stdout, format=log_format, datefmt='%Y%m%d_%H%M%S')

    import networkx as nx

    dag = nx.DiGraph([("Y", "X"), ("V", "Y"), ("U", "X")])
    domains = dict(X=[0,1], Y=[0, 1], U=[0,1,2,3], V=[0,1])

    import bcause.util.domainutils as dutils
    import bcause.util.graphutils as gutils

    domy = dutils.subdomain(domains, *gutils.relevat_vars(dag, "Y"))
    fy = DeterministicFactor(domy, right_vars=["V"], values=[1, 0])

    domx = dutils.subdomain(domains, *gutils.relevat_vars(dag, "X"))

    values = [0, 0, 1, 0, 0, 0, 1, 0]
    fx = DeterministicFactor(domx, left_vars=["X"], values=values)

    domv = dutils.subdomain(domains, "V")
    pv = MultinomialFactor(domv, values=[.1, .9])

    domu = dutils.subdomain(domains, "U")
    pu = MultinomialFactor(domu, values=[.2, .2, .1, .5])

    m = StructuralCausalModel(dag, [fx, fy, pu, pv], cast_multinomial=True)

    data = m.sample(1000, as_pandas=True)[m.endogenous]


    Watch.start()
    inf = GibbsCausal(m,data, num_runs=1000)
    print(inf.causal_query("X", do=dict(Y=0)))
    print(inf.counterfactual_query("X", do=dict(Y=0)))
    print(inf.prob_necessity("Y","X"))
    Watch.stop_print()

    Watch.start()

    inf = EMCC(m, data, num_runs=100, max_iter=50)
    print(inf.causal_query("X", do=dict(Y=0)))
    print(inf.counterfactual_query("X", do=dict(Y=0)))
    print(inf.prob_necessity("Y","X"))
    Watch.stop_print()

    #
    #
    # inf = GDCC(m, data, num_runs=10)
    # print(inf.causal_query("X", do=dict(Y=0)))
    # print(inf.counterfactual_query("X", do=dict(Y=0)))
    # print(inf.prob_necessity("Y","X"))
    #

