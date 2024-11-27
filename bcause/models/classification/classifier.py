from abc import ABC, abstractmethod

import numpy as np

from bcause import BayesianNetwork
from bcause.inference.probabilistic.elimination import VariableElimination
from bcause.learning.parameter import MaximumLikelihoodEstimation


class Classifier(ABC):
    @property
    def classvar(self):
        return self._classvar

    @property
    def attributes(self):
        return self._attributes


    @abstractmethod
    def _build_model(self) -> BayesianNetwork:
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    @abstractmethod
    def fit(self, data):
        pass


class BNetClassifier(Classifier):

    def __init__(self, domains, classvar):
        self._domains = domains
        self._classvar = classvar
        self._attributes = [c for c in domains.keys() if c != classvar]
        self._model = self._build_model()


    @property
    def model(self):
        return self._model


    def predict(self, data, inference_engine = None):
        probs = self.predict_proba(data, inference_engine)
        dom_y = self._domains[self.classvar]
        return np.array([dom_y[c] for c in np.argmax(probs, axis=1)])

    def predict_proba(self, data, inference_engine = None):
        inference_engine = inference_engine or VariableElimination
        inf = inference_engine(self.model)
        probs = [inf.query(self.classvar, evidence=dict(d)).values for _, d in data[self.attributes].iterrows()]
        return np.array(probs)

    def fit(self, data):
        mle = MaximumLikelihoodEstimation(self.model)
        mle.run(data)
        self._model = mle.model


