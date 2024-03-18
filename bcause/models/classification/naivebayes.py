import networkx as nx

from bcause import BayesianNetwork, MultinomialFactor
from bcause.factors.mulitnomial import uniform_multinomial
from bcause.inference.probabilistic.datainference import LaplaceInference
from bcause.inference.probabilistic.elimination import VariableElimination
from bcause.learning.parameter import MaximumLikelihoodEstimation
from bcause.models.classification.classifier import  BNetClassifier
from bcause.util.domainutils import subdomain


class NaiveBayes(BNetClassifier):
    def __init__(self, domains, classVar):
        super().__init__(domains,classVar)

    def _build_model(self) -> BayesianNetwork:
        dag = nx.DiGraph()
        for v in self.attributes:
            dag.add_edge(self.classvar, v)
        return BayesianNetwork.buid_uniform(dag, self._domains)


if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv("./data/igd_test.csv", index_col=0)
    classVar = "y"
    domains = {c:list(data[c].unique()) for c in data.columns}


    clf = NaiveBayes(domains, classVar)
    clf.fit(data)

    y_pred = clf.predict(data)

    clf.model.factors

    clf.predict(data.loc[0:10])
    clf.predict_proba(data.loc[0:10])

    i = 1
    for i in range(0,10):
        print(f"{data.loc[[i]]} {clf.predict_proba(data.loc[[i]])}")

    y = list(data["y"])

    clf.model.domains

    print(sum(y == y_pred))


