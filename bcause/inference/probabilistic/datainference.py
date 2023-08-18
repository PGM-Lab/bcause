import logging
import time

import pandas as pd

from bcause.factors import MultinomialFactor
from bcause.inference.inference import Inference
from bcause.inference.probabilistic.probabilistic import ProbabilisticInference
from bcause.util.arrayutils import as_lists
from bcause.util.datautils import filter_data, to_counts
import bcause.util.domainutils as dutils


class LaplaceInference(ProbabilisticInference):
    def __init__(self, data : pd.DataFrame, domains:dict, preprocess_flag:bool = True):
        self._preprocess_flag = preprocess_flag
        self._domains = domains
        super(self.__class__, self).__init__(data)
    def compile(self, target, evidence=None) -> Inference:

        target = as_lists(target)
        if len(set(target)) != len(target): raise ValueError("Repeated variables in target")
        if not set(target).isdisjoint(evidence.keys()):
            raise ValueError(f"Target {target} and evidence are not disjoint {evidence.keys()}")

        self._target = target
        self._evidence = evidence or dict()
        logging.info(f"Starting inference: target={str(target)} evidence={str(evidence)}")

        self._inference_model = self._preprocess()
        self._compiled = True;
        return self


    def _preprocess(self) -> pd.DataFrame:
        if not self._preprocess_flag:
            return self._model
        relevant_vars = set(self._target).union(self._evidence.keys())
        return self.model[list(relevant_vars)]

    def run(self) -> MultinomialFactor:

        tstart = time.time()

        # Filter according to the evidence
        data = filter_data(self._inference_model, self._evidence)

        # Get the counts
        dom = {k:v for k,v in self._domains.items() if k in self._target}
        #dom = dutils.subdomain(self._domains, *self._target)

        result = to_counts(dom, data, normalize=True)

        self.time = (time.time()-tstart)*1000
        logging.info(f"Finished Laplace Inference in {self.time} ms.")
        return result


if __name__ == "__main__":

    import networkx as nx
    from bcause.models.cmodel import StructuralCausalModel

    # Define a DAG and the domains
    dag = nx.DiGraph([("V1", "V2"), ("V2", "V3"), ("V3", "V4"), ("U1", "V1"), ("U2", "V2"), ("U2", "V4"), ("U3", "V3")])
    model = StructuralCausalModel(dag)

    domains = dict(V1=[0, 1], V2=[0, 1], V3=[0, 1], V4=[0, 1], U1=[0, 1], U2=[0, 1, 2, 3], U3=[0, 1, 2, 3])

    model.fill_random_factors(domains)

    data = model.sample(100, as_pandas=True)

    inf = LaplaceInference(data, domains)

    p = inf.query(["V1"], conditioning="V2", evidence=dict(V3=1))
    D = data.loc[data.V3==1]

    assert p.R(V1=1, V2=0) == len(D.loc[(D.V1==1) & (D.V2==0)])/len(D.loc[(D.V2==0)])
