import logging
from abc import ABC
from typing import Callable

from bcause.factors.factor import Factor
from bcause.inference.inference import Inference
from bcause.inference.probabilistic.datainference import LaplaceInference
from bcause.models.cmodel import StructuralCausalModel
from bcause.models.pgmodel import PGModel
from bcause.models.transform.combination import fusion_roots, counterfactual_model
from bcause.util import domainutils as dutils
from bcause.util.arrayutils import as_lists
from bcause.util.assertions import assert_dag_with_nodes


class CausalInference(Inference):
    """
    Performs causal inference using a structural causal model (SCM) with interventions, counterfactuals,
    and probability calculations for necessity and sufficiency.
    """
    def __init__(self, model: StructuralCausalModel, prob_inf_fn: Callable):
        """
        Initializes the CausalInference object.

        Args:
            model (StructuralCausalModel): The structural causal model.
            prob_inf_fn (Callable): A function to create the probabilistic inference engine.
        """
        self._model = model
        self._inf = None
        self._prob_inf_fn = prob_inf_fn
        self._compiled = False

    def compile(self, target, do, evidence=None, counterfactual=False) -> Inference:
        """
        Prepares the causal inference model with the specified target, intervention, and evidence.

        Args:
            target: The target variables to query.
            do: The interventions to apply to the model.
            evidence (dict, optional): Observational evidence for the query.
            counterfactual (bool, optional): Whether to perform counterfactual inference. Defaults to False.

        Returns:
            Inference: The compiled inference object.
        """

        target = as_lists(target)
        evidence = evidence or dict()

        self._do = do
        self._counterfactual = counterfactual

        logging.getLogger( __name__ ).info(f"Starting causal inference: target={str(target)}  intervention={str(do)} evidence={str(evidence)}")


        do_vars = list(do.keys()) if isinstance(do, dict) else sum([list(d.keys()) for d in do], [])
        assert_dag_with_nodes(self.model.graph, do_vars)

        if counterfactual:
            do = do if isinstance(self._do, list) else [self._do]
            new_do = dict()
            for i in range(len(do)):
                for k, v in do[i].items():
                    new_do[f"{k}_{i+1}"] = v
            do = new_do


        evidence = {**evidence, **do}

        self._inference_model = self._preprocess()
        self._inf = self._prob_inf_fn(self._inference_model)
        #self._inf.compile(target, evidence)
        self._query_args = dict(target=target, evidence=evidence)
        self._compiled = True

        return self

    def _preprocess(self, *args, **kwargs) -> PGModel:
        """
        Preprocess the model by applying interventions or constructing a counterfactual model.

        Returns:
            PGModel: The preprocessed model ready for inference.
        """
        if not self._counterfactual:
            new_model = self.model.intervention(**self._do)
            logging.getLogger( __name__ ).debug(f"Intervened DAG: {new_model.graph.edges}")
        else:
            new_model = counterfactual_model(self.model, self._do)
            logging.getLogger( __name__ ).debug(f"Counterfactual DAG: {new_model.graph.edges}")

        return new_model


    @property
    def counterfactual(self):
        """
        Indicates whether the current inference involves counterfactual analysis.

        Returns:
            bool: True if counterfactual analysis is involved, False otherwise.
        """
        return self._counterfactual

    def run(self) -> Factor:
        """
        Executes the inference and returns the resulting factor.

        Returns:
            Factor: The result of the inference query.
        """
        #return self._inf.run()
        return self._inf.query(**self._query_args)

    def query(self, target, do, evidence=None, counterfactual=False, targets_subgraphs = None):
        """
        Executes a general causal or counterfactual query.

        Args:
            target: The target variables to query.
            do: The interventions to apply.
            evidence (dict, optional): Observational evidence for the query.
            counterfactual (bool, optional): Whether to perform counterfactual inference. Defaults to False.
            targets_subgraphs (list, optional): Subgraph indices for counterfactual targets. Defaults to None.

        Returns:
            The result of the query.
        """
        if counterfactual:
            #if not isinstance(do, dict): raise ValueError("Intervention must be specified in a single dictionary")
            target = as_lists(target)
            targets_subgraphs = targets_subgraphs or [1]*len(target)
            target = [f"{t[0]}_{t[1]}" for t in zip(target, targets_subgraphs)]
        return self.compile(target, do, evidence,counterfactual).run()

    def causal_query(self, target, do, evidence=None):
        """
        Handles causal queries.

        Args:
            target: The target variables to query.
            do: The interventions to apply.
            evidence (dict, optional): Observational evidence for the query.

        Returns:
            The result of the causal query.
        """
        return self.query(target, do, evidence=evidence, counterfactual=False)

    def counterfactual_query(self, target, do, evidence=None, targets_subgraphs = None):
        """
        Handles counterfactual queries.

        Args:
            target: The target variables to query.
            do: The interventions to apply.
            evidence (dict, optional): Observational evidence for the query.
            targets_subgraphs (list, optional): Subgraph indices for counterfactual targets. Defaults to None.

        Returns:
            The result of the counterfactual query.
        """
        return self.query(target, do, evidence=evidence, counterfactual=True, targets_subgraphs=targets_subgraphs)

    def prob_necessity_sufficiency(self, cause, effect, true_false_cause:tuple=None, true_false_effect:tuple=None):
        """
        Computes the probability of necessity and sufficiency for a given cause-effect pair.

        Args:
            cause: The causal variable.
            effect: The effect variable.
            true_false_cause (tuple, optional): Tuple specifying the true and false states of the cause. Defaults to None.
            true_false_effect (tuple, optional): Tuple specifying the true and false states of the effect. Defaults to None.

        Returns:
            float: The computed probability.
        """
        #if not self._compiled: self.compile()

        # Determine the true and false states
        Tcause, Fcause = true_false_cause or dutils.identify_true_false(cause, self.model.domains[cause])
        Teffect, Feffect = true_false_effect or dutils.identify_true_false(effect, self.model.domains[effect])

        # Run the query
        return self._process_output(self.counterfactual_query(
            [effect]*2,
            do=[{cause: Fcause}, {cause: Tcause}],
            targets_subgraphs=[1,2]
        ), {effect + "_1": Feffect, effect + "_2": Teffect})

    def _process_output(self, result, obs):
        """
        Processes the output of a counterfactual query to compute probabilities.

        Args:
            result: The raw query result.
            obs: Observational constraints for processing the result.

        Returns:
            float: The processed result.
        """
        return result.R(**obs).marginalize(*list(obs.keys())).values[0]

    def prob_necessity(self, cause, effect, true_false_cause:tuple=None, true_false_effect:tuple=None):
        """
        Computes the probability of necessity for a given cause-effect pair.

        Args:
            cause: The causal variable.
            effect: The effect variable.
            true_false_cause (tuple, optional): Tuple specifying the true and false states of the cause. Defaults to None.
            true_false_effect (tuple, optional): Tuple specifying the true and false states of the effect. Defaults to None.

        Returns:
            float: The computed probability of necessity.
        """

        #if not self._compiled: self.compile()
        # Determine the true and false states
        Tcause, Fcause = true_false_cause or dutils.identify_true_false(cause, self.model.domains[cause])
        Teffect, Feffect = true_false_effect or dutils.identify_true_false(effect, self.model.domains[effect])

        # Run the query
        result = self.counterfactual_query(
            effect,
            do={cause: Fcause},
            evidence={cause: Tcause, effect: Teffect},
        )
        return self._process_output(result, {effect+"_1": Feffect})

    def prob_sufficiency(self, cause, effect, true_false_cause:tuple=None, true_false_effect:tuple=None):
        """
        Computes the probability of sufficiency for a given cause-effect pair.

        Args:
            cause: The causal variable.
            effect: The effect variable.
            true_false_cause (tuple, optional): Tuple specifying the true and false states of the cause. Defaults to None.
            true_false_effect (tuple, optional): Tuple specifying the true and false states of the effect. Defaults to None.

        Returns:
            float: The computed probability of sufficiency.
        """
        #if not self._compiled: self.compile()
        # Determine the true and false states
        Tcause, Fcause = true_false_cause or dutils.identify_true_false(cause, self.model.domains[cause])
        Teffect, Feffect = true_false_effect or dutils.identify_true_false(effect, self.model.domains[effect])

        # Run the query
        return self._process_output(self.counterfactual_query(
            effect,
            do={cause: Tcause},
            evidence={cause: Fcause, effect: Feffect}
        ), {effect+"_1": Teffect})


class CausalObservationalInference(ABC):
    @property
    def data(self):
        return self._data


class PearlBounds(CausalInference, CausalObservationalInference):
    def __init__(self, model:StructuralCausalModel, data, causal_inf_fn: Callable = None, interval_result=True, max_iter=100, num_runs=10, parallel = False, min_rating=0.9, outliers_removal=True):
        self._data = data
        super().__init__(model, LaplaceInference)

    def compile(self) -> Inference:
        self._inf = self._prob_inf_fn(self._data, domains=self._model.domains)
        self._compiled = True

        return self

    def _preprocess(self, *args, **kwargs) -> PGModel:
        pass

    @property
    def counterfactual(self):
        return self._counterfactual

    def run(self) -> Factor:
        raise ValueError("Invalid method")
    def query(self, target, do, evidence=None, counterfactual=False, targets_subgraphs = None):
        raise ValueError("Invalid method")

    def causal_query(self, target, do, evidence=None):
        raise ValueError("Invalid method")

    def counterfactual_query(self, target, do, evidence=None, targets_subgraphs = None):
        raise ValueError("Invalid method")

    def _compute_bounds(self, query, cause, effect, true_false_cause:tuple=None, true_false_effect:tuple=None):

#        if not self.model.exogeneity(cause,effect):
#            raise ValueError("Exogeneity condition not satisfied")

        if not self._compiled: self.compile()

        # Determine the true and false states
        Tcause, Fcause = true_false_cause or dutils.identify_true_false(cause, self.model.domains[cause])
        Teffect, Feffect = true_false_effect or dutils.identify_true_false(effect, self.model.domains[effect])

        # Compute the P(Y|X)
        p = self._inf.query(effect, cause)
        pyx = p.R(**{effect: Teffect, cause: Tcause}).values[0]
        pyx_ = p.R(**{effect: Teffect, cause: Fcause}).values[0]
        py_x_ = 1 - pyx_

        pns_l = max(0, pyx - pyx_)
        pns_u = min(pyx, py_x_)

        if query == "PNS":
            out = [pns_l, pns_u]
        elif query == "PN":
            out = [pns_l / pyx, pns_u / pyx]
        elif query == "PS":
            out = [pns_l / py_x_, pns_u / py_x_]
        else:
            raise ValueError("Invalid query key")

        return out

    def prob_necessity_sufficiency(self, cause, effect, true_false_cause:tuple=None, true_false_effect:tuple=None):
        return self._compute_bounds("PNS", cause, effect, true_false_cause, true_false_effect)


    def prob_necessity(self, cause, effect, true_false_cause:tuple=None, true_false_effect:tuple=None):
        return self._compute_bounds("PN", cause, effect, true_false_cause, true_false_effect)

    def prob_sufficiency(self, cause, effect, true_false_cause:tuple=None, true_false_effect:tuple=None):
        return self._compute_bounds("PS", cause, effect, true_false_cause, true_false_effect)


