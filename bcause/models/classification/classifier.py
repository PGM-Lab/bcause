from abc import ABC, abstractmethod

from bcause import BayesianNetwork

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
    def fit(self, X, y):
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


    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

    def fit(self, X, y):
        pass

