from abc import abstractmethod
from functools import reduce

import pandas as pd
import numpy as np
import time

from torch.ao.nn.quantized.functional import threshold

from bcause.factors import MultinomialFactor, DeterministicFactor
from bcause.factors.multinomial import uniform_multinomial
from bcause.inference.probabilistic.elimination import VariableElimination
from bcause.learning.parameter import IterativeParameterLearning
from bcause.models.cmodel import StructuralCausalModel
from bcause.models.pgmodel import DiscreteDAGModel
from bcause.util.datadeps import DataDepAnalysis
from bcause.util.datautils import to_counts
from bcause.factors.values import BTreeStore
from bcause.factors.values.btreeops import BTreeStoreOperations


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
            from bcause.util import domainutils as dutils
            pa = self.prior_model.get_parents(v)
            dom = dutils.subdomain(self.prior_model.domains, *([v] + pa))
            return MultinomialFactor(dom, values=0)

        return {v: get_pcounts(v) for v in self.trainable_vars}


class ExpectationMaximization(AbastractExpectationMaximization):
    def __init__(self, prior_model: DiscreteDAGModel, trainable_vars: list = None,
                 ignore_convergence:bool = False, vtype = "numpy", inference_method=VariableElimination):
        self._prior_model = prior_model
        self._trainable_vars = trainable_vars
        self._inference_method = inference_method
        self._converged_vars = set()
        self._ignore_convergence = ignore_convergence
        self._vtype = vtype

    def _get_pseudocounts_dict(self) -> dict:
        def get_pcounts(v):
            from bcause.util import domainutils as dutils
            pa = self.prior_model.get_parents(v)
            dom = dutils.subdomain(self.prior_model.domains, *([v] + pa))
            return MultinomialFactor(dom, values=0, vtype=self._vtype)

        return {v: get_pcounts(v) for v in self.trainable_vars}

    def _get_obs_counts(self, target):
        obs_blanket = self._datadeps.get_minimal_obs_blanket(target)
        return [(obs, sum(reduce( lambda x,y : x&y, [self._data[v] == s for v,s in obs.items()]))) for obs in obs_blanket]


    def _expectation(self, **kwargs):
        #print("===")
        self._inf = self._inference_method(self._model)
        pcounts = self._get_pseudocounts_dict()
        for v in set(self.trainable_vars).difference(self._converged_vars):

            for obs, c in self._get_obs_counts(v):
                relevant = [v] + self._prior_model.get_parents(v)
                hidden = [x for x in relevant if x not in obs]

                #print(f"{hidden} | {obs}")

                post = self._inf.query(target=hidden, evidence=obs)
                if all(v==0 for v in post.values):
                    post = uniform_multinomial(post.domain)

                exp_counts = post * c
                pcounts[v] = pcounts[v] + exp_counts

        return pcounts

    def _maximization(self, pcounts, **kwargs):
        new_probs = dict()
        for v in set(self.trainable_vars).difference(self._converged_vars):
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
        from scipy.special import rel_entr
        if self._ignore_convergence:
            return False

        for v in self._trainable_vars:
            if v not in self._converged_vars:
                P = self.model_evolution[-2].factors[v]
                Q = self.model_evolution[-1].factors[v]
                kl_div = sum(rel_entr(P.values, Q.values))
                if kl_div == 0:
                    self._converged_vars = self._converged_vars | {v}
        return set(self._trainable_vars) == self._converged_vars


class ExpectationMaximizationPrecomputed(ExpectationMaximization):

    def __init__(self, prior_model: DiscreteDAGModel, trainable_vars: list = None,
                 ignore_convergence:bool = False , as_list = False, inference_method=VariableElimination):
        super().__init__(prior_model, trainable_vars, ignore_convergence, inference_method)
        self._phi_1 = dict()
        self._phi_2 = dict()
        self._as_list = as_list
        self._init_time = 0
        self._iteration_time = 0

    @property
    def phi_1(self):
        return self._phi_1

    @phi_1.setter
    def phi_1(self, value):
        self._phi_1 = value

    @property
    def phi_2(self):
        return self._phi_2

    @phi_2.setter
    def phi_2(self, value):
        self._phi_2 = value

    @property
    def init_time(self):
        return self._init_time

    @init_time.setter
    def init_time(self, value):
        self._init_time = value

    @property
    def iteration_time(self):
        return self._iteration_time

    @iteration_time.setter
    def iteration_time(self, value):
        self._iteration_time = value


    def initialize(self, data: pd.DataFrame, **kwargs):
        init_time = time.time()
        super().initialize(data, **kwargs)
        data = self._data.copy()[self.model.endogenous]

        # get the data as factor
        factor_data = to_counts(domains= {k:v for k,v in self._model.domains.items() if k in self._model.endogenous}, data= data, normalize=True, vtype="list" if self._as_list else None)

        # multiply the factors of the model for each ccomponent. E.g. P(V1|Y,U) * P(V2|V1,U)
        endo_component = {v: self._model.get_endo_ccomponent(v) for v in self.trainable_vars}
        phi_1 =  {v: reduce(lambda x, y: x * y, [self._model.factors[f] for f in endo_component[v]])
                         for v in self.trainable_vars}

        #Product for each factor_table with factor_data marginalize by the corresponding variables
        phi_2 = {v: factor_data.marginalize(*set(factor_data.variables).difference(phi_1[v].variables)) * phi_1[v] for v in self.trainable_vars}
        self.phi_1 = phi_1
        self.phi_2 = phi_2
        self.init_time = time.time() - init_time


    def _calculate_updated_factors(self):
        init_time = time.time()
        self._inf = self._inference_method(self._model)
        new_probs = dict()

        # loop over trainable variables
        for v in set(self.trainable_vars).difference(self._converged_vars):
            # Multiply each v of self.phi_1 by the probability of "U"
            # numerator = self.phi_2[v] * self._model.factors[v]
            # denominator = (self.phi_1[v] * self._model.factors[v]).marginalize(v)
            # result = (numerator / denominator).marginalize(*set(numerator.variables).difference({v}))
            # new_probs[v] = result/ (result.marginalize(v))
            denominator = (self.phi_1[v] * self._model.factors[v]).marginalize(v)
            fraction = ((self.phi_2[v]/denominator).marginalize(*set(self.phi_2[v].variables).difference({v})))
            new_probs[v] = fraction * self._model.factors[v]
        self.iteration_time += time.time() - init_time

        return new_probs    # return the updated factors


class ExpectationMaximizationTrees(ExpectationMaximization):
    def __init__(self, prior_model: DiscreteDAGModel, trainable_vars: list = None,
                 ignore_convergence:bool = False, combine_steps = False,threshold = 0, inference_method=VariableElimination):
        super().__init__(prior_model, trainable_vars, ignore_convergence, inference_method)
        self._phi_1 = dict()
        self._phi_2 = dict()
        self._init_time = 0
        self._iteration_time = 0
        self._combine_steps = combine_steps
        self._threshold = threshold

    @property
    def phi_1(self):
        return self._phi_1

    @phi_1.setter
    def phi_1(self, value):
        self._phi_1 = value

    @property
    def phi_2(self):
        return self._phi_2

    @phi_2.setter
    def phi_2(self, value):
        self._phi_2 = value

    @property
    def init_time(self):
        return self._init_time

    @init_time.setter
    def init_time(self, value):
        self._init_time = value

    @property
    def iteration_time(self):
        return self._iteration_time

    @iteration_time.setter
    def iteration_time(self, value):
        self._iteration_time = value

    @property
    def threshold(self):
        return self._threshold
    @threshold.setter
    def threshold(self, value):
        self._threshold = value

    def _reshape_value(self,domain, unshaped_values):
        dom = domain
        shape = [len(d) for d in dom.values()]
        val = np.reshape(unshaped_values, shape)
        return val

    def initialize(self, data: pd.DataFrame, **kwargs):
        init_time = time.time()
        super().initialize(data, **kwargs)
        data = self._data.copy()[self.model.endogenous]

        # get the data as factor
        factor_data = to_counts(domains= {k:v for k,v in self._model.domains.items() if k in self._model.endogenous}, data= data, normalize=True)
        factor_data = BTreeStore(domain=factor_data.domain, data=self._reshape_value(factor_data.domain, factor_data.values), is_equation=False)

        # change endo factors to BTreeStore
        for v in self._model.endogenous:
            f = self._model.factors[v]
            exovar =  list(set(f.variables) & set(self._model.exogenous))[0]
            self._model.factors[v] = BTreeStore(domain=f.domain, data=self._reshape_value(f.domain, f.values),exovar=exovar, is_equation=True)

        # self._model.factors["U"] = MultinomialFactor(domain=self._model.factors["U"].domain, values= [0.0648, 0.2646, 0.0417, 0.0502, 0.0158, 0.2251, 0.0842, 0.0877, 0.1605])
        # change exo factors to BTreeStore
        for v in self._model.exogenous:
            # f = self._model.factors[v]
            # self._model.factors[v] = BTreeStore(domain=f.domain, data=self._reshape_value(f.domain, f.values), exovar=v, is_equation=False)
            # self._model.factors[v].set_data(BTreeStore.var_to_nonconsecutive(self._model.factors[v].data, v))
            f = self._model.factors[v]
            self._model.factors[v] = BTreeStore(data = BTreeStoreOperations.build_random_marginal_tree(v, f.domain[v], threshold=self._threshold), domain=f.domain, exovar=v, is_equation=False)
            self._model.factors[v].set_data(BTreeStore.var_to_nonconsecutive(self._model.factors[v].data, v))

        # multiply the factors of the model for each ccomponent. E.g. P(V1|Y,U) * P(V2|V1,U)
        endo_component_unsorted = {v: self._model.get_endo_ccomponent(v) for v in self.trainable_vars}
        # get right order for multiplication
        endo_component = {v: list(nx.topological_sort(self._model.graph.subgraph(endo_component_unsorted[v])))  for v in self.trainable_vars}
        phi_1 =  {v: reduce(lambda x, y: BTreeStoreOperations.multiply_SE(x, y, method = "SE_only"), [self._model.factors[f] for f in endo_component[v]])
                         for v in self.trainable_vars}

        #Product for each factor_table with factor_data marginalize by the corresponding variables
        phi_2 = {v: BTreeStoreOperations.multiply(phi_1[v],factor_data.marginalize(*set(factor_data.variables).difference(phi_1[v].variables))) for v in self.trainable_vars}


         # Store phi_1 and phi_2
        self.phi_1 = phi_1
        self.phi_2 = phi_2
        self.init_time = time.time() - init_time

    def _calculate_updated_factors(self):
        loop_time = time.time()
        self._inf = self._inference_method(self._model)
        new_probs = dict()

        # loop over trainable variables
        for v in set(self.trainable_vars).difference(self._converged_vars):
            # Multiply each v of self.phi_1 by the probability of "U"
            denominator_product = BTreeStoreOperations.multiply_SE(self.phi_1[v], self._model.factors[v], method = "exogenous")
            denominator = BTreeStoreOperations.marginalize(denominator_product, [v])
            fraction = BTreeStoreOperations.divide(self.phi_2[v], denominator)
            if self._combine_steps:
                subtrees = BTreeStoreOperations.marginalize_endogenous(fraction.data)
                new_probs[v] = BTreeStore(data=BTreeStoreOperations.addition_exo(subtrees, self._model.factors[v].data,combine_steps=True, exo_mult=True), domain=self._model.factors[v].domain)
            else:
                subtrees = BTreeStoreOperations.marginalize_endogenous(fraction.data) #exovar= v
                marginalize = BTreeStore(data=BTreeStoreOperations.addition_exo(subtrees,self._model.factors[v].data,combine_steps=False), domain=self._model.factors[v].domain)
                new_probs[v] = BTreeStoreOperations.multiply(marginalize, self._model.factors[v])

        self.iteration_time += time.time() - loop_time
        return new_probs



if __name__ == "__main__":
    log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'

    # logging.getLogger( __name__ ).basicConfig(level=logging.getLogger( __name__ ).DEBUG, stream=sys.stdout, format=log_format, datefmt='%Y%m%d_%H%M%S')

    import networkx as nx

    def factor_as_list(factor: MultinomialFactor):
        return MultinomialFactor(domain=factor.domain, values=factor.values, left_vars=factor.left_vars,
                                         right_vars=factor.right_vars, vtype="list")

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
    pu = MultinomialFactor(domu, values=[0.95, 0.02, 0.01, 0.02])

    m = StructuralCausalModel(dag, [fx, fy, pu, pv], cast_multinomial=True)

    # Set seed
    from bcause.util import randomUtil
    data = m.sample(10000, as_pandas=True)[m.endogenous]

    # randomUtil.seed(1234)
    # em1 = ExpectationMaximization(m.randomize_factors(m.exogenous, allow_zero=False), ignore_convergence=True)
    # em1.run(data, max_iter=100)


    g5_model_path = "./models/WUPES/g5_model_22.bif"
    g5_data_path = "./models/WUPES/g5_data_22.csv"
    g4_model_path = "./models/WUPES/g4_model_64.bif"
    g4_data_path = "./models/WUPES/g4_data_64.csv"
    wupes_model = "/Users/antoniogonzalezalves/Desktop/model_wupes.bif"
    wupes_data = "/Users/antoniogonzalezalves/Desktop/data_wupes.csv"
    m2 = StructuralCausalModel.read(wupes_model)
    data2 = pd.read_csv(wupes_data).astype(str)[m2.endogenous]

    # print(f"Running EM for {g5_model_path} for 100 iterations")
    # print("---")
    # print(f"Running as numpy")
    # numpy_time = time.time()
    # randomUtil.seed(1234)
    # em2 = ExpectationMaximizationPrecomputed(m2.randomize_factors(m2.exogenous, allow_zero=False), ignore_convergence=True, as_list=False)
    # em2.run(data2, max_iter=100)
    # print(f"Numpy time: {time.time() - numpy_time}")

    #
    print("---")
    print("Running as list")
    list_time = time.time()
    randomUtil.seed(1234)


    as_list = False
    random_fact = m2.randomize_factors(m2.exogenous, allow_zero=False)
    if as_list:
        for v in random_fact.variables:
            random_fact[v] = factor_as_list(random_fact[v])

    em2 = ExpectationMaximizationPrecomputed(m2,
                                             ignore_convergence=True,as_list=as_list)
    em2.run(data2, max_iter=100)
    print(f"List time: {time.time() - list_time}")

    print("---")
    print("Running as BTree")
    randomUtil.seed(1234)
    Btree_time = time.time()
    em3 = ExpectationMaximizationTrees(m2.randomize_factors(m2.exogenous, allow_zero=False), ignore_convergence=True, combine_steps=False)
    em3.run(data2, max_iter=100)
    print(f"BTree time: {time.time() - Btree_time}")


