from __future__ import annotations
import copy
import logging
import math
from typing import Dict, List, Iterable, Union, Hashable
import numpy as np
import pandas as pd
import bcause.util.domainutils as dutils
from bcause.factors.values.store import DataStore
from bcause.factors.values import store_dict
import bcause.factors.factor as bf
from bcause.util.domainutils import (
    assingment_space,
    state_space,
    steps,
    random_assignment,
    to_numeric_domains
)
from bcause.util.arrayutils import (
    normalize_array,
    set_value,
    concatenate_with
)


class MultinomialFactor(bf.DiscreteFactor, bf.ConditionalFactor):
    """
    Represents a multinomial factor, which can be a joint or conditional distribution over discrete variables.
    """

    def __init__(self, domain: Dict, values, left_vars: list = None, right_vars: list = None, vtype=None):
        """
        Initializes a MultinomialFactor instance.

        Args:
            domain (Dict): Dictionary defining variable names and their possible states.
            values (Iterable): Values representing the factor's probabilities or data.
            left_vars (list, optional): Variables on the left side of the conditional. Defaults to None.
            right_vars (list, optional): Variables on the right side of the conditional. Defaults to None.
            vtype (optional): Type of data storage. Defaults to DataStore.DEFAULT_STORE.
        """
        vtype = vtype or DataStore.DEFAULT_STORE
        self._check_domain(domain)

        if (isinstance(values, Iterable) and not isinstance(values, dict)) or np.isscalar(values):
            shape = [len(d) for d in domain.values()]
            if np.ndim(values) == 0:
                values = [values] * int(np.prod(shape))
            if np.ndim(values) == 1:
                values = np.reshape(values, shape)

        self._store = store_dict[vtype](data=values, domain=domain)
        self.set_variables(list(domain.keys()), left_vars, right_vars)
        self.vtype = vtype

        def builder(**kwargs):
            if "left_vars" not in kwargs and "right_vars" not in kwargs:
                kwargs["left_vars"] = self.left_vars
            return MultinomialFactor(**kwargs, vtype=vtype)

        self.builder = builder

    def to_deterministic(self):
        """
        Converts the multinomial factor to a deterministic one.

        Returns:
            DeterministicFactor: A deterministic factor based on the maximum probability value.

        Raises:
            ValueError: If more than one variable is on the left side.
        """
        if len(self.left_vars) != 1:
            raise ValueError("Wrong number of variables on the left")
        v = self.left_vars[0]
        values = self.values_array().argmax(axis=self.variables.index(v))
        from bcause.factors import DeterministicFactor
        return DeterministicFactor(self.domain, left_vars=[v], values=values)

    def constant(self, left_value):
        """
        Creates a new factor where the left variable(s) take a constant value with probability 1.

        Args:
            left_value (Any): The value(s) to which the left variable(s) are set.

        Returns:
            MultinomialFactor: A new factor with the specified constant value.

        Raises:
            ValueError: If the left value is not part of the left domain.
        """
        new_dom = self.left_domain
        if len(new_dom) != 1:
            raise ValueError("Only one variable on the left is allowed")
        states = new_dom[self.left_vars[0]]
        new_data = [0.0] * len(states)

        if not isinstance(left_value, (tuple, list)):
            if left_value not in states:
                raise ValueError("Value not in domain")
            new_data[states.index(left_value)] = 1.0
        else:
            left_values = list(self.left_domain.values())[0]
            if not set(left_value).issubset(set(left_values)):
                raise ValueError("Values are not contained")
            k = 1 / len(left_value)
            for v in left_value:
                new_data[states.index(v)] = k

        return self.builder(domain=new_dom, values=new_data)

    def domain_change(self, new_dom):
        """
        Returns a new factor with a modified domain.

        Args:
            new_dom (Dict): The new domain to apply.

        Returns:
            MultinomialFactor: A new factor with the updated domain.
        """
        return self.builder(domain=new_dom, values=self.values)

    def domain_to_str(self):
        """
        Converts the domain values to strings.

        Returns:
            MultinomialFactor: A new factor with the domain values converted to strings.
        """
        new_dom = {v: [str(s) for s in d] for v, d in self.domain.items()}
        return self.domain_change(new_dom)

    def restrict(self, **observation) -> MultinomialFactor:
        """
        Restricts the factor to specific observed values.

        Args:
            **observation: Key-value pairs of variables and their observed values.

        Returns:
            MultinomialFactor: A new factor restricted to the observed values.
        """
        if len(set(observation.keys()).intersection(self._variables)) == 0:
            return self
        new_store = self.store.restrict(**observation)
        new_right_vars = [v for v in new_store.variables if v in self.right_vars]
        return self.builder(domain=new_store.domain, values=new_store.data, right_vars=new_right_vars)

    def _prepare_operand(self, f):
        """
        Prepares an operand by converting scalars to constant factors.

        Args:
            f: The operand to prepare.

        Returns:
            MultinomialFactor: The prepared operand as a factor.
        """
        if isinstance(f, (int, float)):
            f = self.builder(domain=dict(), values=[f])
        return f

    def multiply(self, other: Union[MultinomialFactor, int, float]):
        """
        Multiplies this factor by another factor or scalar.

        Args:
            other (MultinomialFactor | int | float): The factor or scalar to multiply by.

        Returns:
            MultinomialFactor: A new factor representing the product.
        """
        other = self._prepare_operand(other)
        new_store = self.store.multiply(other.store)
        new_right_vars = [
            v for v in new_store.variables if v not in self.left_vars and v not in other.left_vars
        ]
        return self.builder(domain=new_store.domain, values=new_store.data, right_vars=new_right_vars)

    def addition(self, other):
        """
        Adds this factor to another factor or scalar.

        Args:
            other (MultinomialFactor): The factor or scalar to add.

        Returns:
            MultinomialFactor: A new factor representing the sum.
        """
        other = self._prepare_operand(other)
        new_store = self.store.addition(other.store)
        new_right_vars = [
            v for v in new_store.variables if v not in self.left_vars and v not in other.left_vars
        ]
        return self.builder(domain=new_store.domain, values=new_store.data, right_vars=new_right_vars)

    def subtract(self, other):
        """
        Subtracts another factor or scalar from this factor.

        Args:
            other (MultinomialFactor): The factor or scalar to subtract.

        Returns:
            MultinomialFactor: A new factor representing the difference.
        """
        other = self._prepare_operand(other)
        new_store = self.store.subtract(other.store)
        new_right_vars = [
            v for v in new_store.variables if v not in self.left_vars and v not in other.left_vars
        ]
        return self.builder(domain=new_store.domain, values=new_store.data, right_vars=new_right_vars)

    def divide(self, other):
        """
        Divides this factor by another factor or scalar.

        Args:
            other (MultinomialFactor): The factor or scalar to divide by.

        Returns:
            MultinomialFactor: A new factor representing the quotient.

        Warnings:
            Logs a warning if division by zero occurs.
        """
        import warnings
        with warnings.catch_warnings(record=True) as W:
            other = self._prepare_operand(other)
            new_store = self.store.divide(other.store)
            new_right_vars = [
                v for v in new_store.variables if v in self.right_vars or v in other.variables
            ]
            out = self.builder(domain=new_store.domain, values=new_store.data, right_vars=new_right_vars)
            for w in W:
                logging.getLogger(__name__).warning(f"{w.message}: {self.name}/{other.name}")
        return out

    def marginalize(self, *vars_remove) -> MultinomialFactor:
        """
        Marginalizes (sums out) the specified variables from the factor.

        Args:
            *vars_remove: Variables to marginalize over.

        Returns:
            MultinomialFactor: A new factor with the specified variables marginalized out.
        """
        if len(set(vars_remove).intersection(self._variables)) == 0:
            return self
        new_store = self.store.marginalize(*vars_remove)
        new_right_vars = [v for v in new_store.variables if v in self.right_vars]
        return self.builder(domain=new_store.domain, values=new_store.data, right_vars=new_right_vars)

    def maxmarginalize(self, *vars_remove) -> MultinomialFactor:
        """
        Performs max-marginalization, retaining the maximum value over the specified variables.

        Args:
            *vars_remove: Variables to marginalize over.

        Returns:
            MultinomialFactor: A new factor with max-marginalization applied.
        """
        if len(set(vars_remove).intersection(self._variables)) == 0:
            return self
        new_store = self.store.maxmarginalize(*vars_remove)
        new_right_vars = [v for v in new_store.variables if v in self.right_vars]
        return self.builder(domain=new_store.domain, values=new_store.data, right_vars=new_right_vars)

        def prob(self, observations: List[Dict]) -> List:
            """
            Computes the probability for a list of observations.

            Args:
                observations (List[Dict]): A list of dictionaries representing the variable assignments.

            Returns:
                List: A list of probabilities corresponding to each observation.
            """
            return [self.get_value(**x) for x in observations]


    def log_prob(self, observations: List[Dict]) -> List:
        """
        Computes the log-probability for a list of observations.

        Args:
            observations (List[Dict]): A list of dictionaries representing the variable assignments.

        Returns:
            List: A list of log-probabilities corresponding to each observation.
        """
        return [math.log(self.get_value(**x)) for x in observations]


    def sample(self, size: int = 1, varnames: bool = True) -> List:
        """
        Generates random samples from the factor's distribution.

        Args:
            size (int): Number of samples to generate (default is 1).
            varnames (bool): Whether to return samples with variable names (default is True).

        Returns:
            List: A list of sampled variable assignments.

        Raises:
            ValueError: If the sample size is less than 1.
        """
        if size < 1:
            raise ValueError("Sample size cannot be lower than 1.")
        return [self._sample(varnames=varnames) for _ in range(size)]


    def sample_conditional(self, observations: List[Dict], varnames: bool = True) -> List:
        """
        Generates samples conditional on the given observations.

        Args:
            observations (List[Dict]): A list of observed values.
            varnames (bool): Whether to return samples with variable names (default is True).

        Returns:
            List: A list of conditional samples.
        """
        if len(observations) < 1:
            raise ValueError("Observations must not be empty.")
        df_obs = pd.DataFrame(observations).drop_duplicates()
        factors = {
            tuple(val_obs): self.R(**dict(zip(df_obs.columns, val_obs))) for val_obs in df_obs.values
        }
        return [factors[tuple(obs.values())]._sample(varnames=varnames) for obs in observations]


    def _sample(self, varnames: bool = True) -> tuple:
        """
        Internal method to generate a single sample from the factor's distribution.

        Args:
            varnames (bool): Whether to return samples with variable names (default is True).

        Returns:
            tuple: A single sampled variable assignment.

        Raises:
            NotImplementedError: If sampling for conditional distributions is not supported.
        """
        if len(self.right_vars) == 0:
            possible_states = state_space(self.left_domain)
            observations = assingment_space(self.left_domain)
            probs = np.array([float(self.store.get_value(**obs)) for obs in observations])
            idx = np.random.choice(len(possible_states), p=probs / probs.sum())
            sample = observations[idx] if varnames else possible_states[idx]
            return sample
        else:
            raise NotImplementedError("Sampling not available for conditional distributions.")


    def copy_with_dummy_state(self, target_var, state_name):
        """
        Creates a copy of the factor, adding a dummy state to the specified variable.

        Args:
            target_var: The variable to add a dummy state to.
            state_name: The name of the new dummy state.

        Returns:
            MultinomialFactor: A new factor with the dummy state added.
        """
        axis = self.variables.index(target_var)
        new_values = concatenate_with(self.values_array(self.variables).copy(), 0.0, axis)
        new_domain = copy.deepcopy(self.domain)
        new_domain[target_var].append(state_name)
        return self.builder(domain=new_domain, values=new_values, left_vars=self.left_vars)


    def __mul__(self, other):
        """Overrides the multiplication operator."""
        return self.multiply(other)


    def __rmul__(self, other):
        """Overrides the reverse multiplication operator."""
        return other.multiply(self)


    def __add__(self, other):
        """Overrides the addition operator."""
        return self.addition(other)


    def __radd__(self, other):
        """Overrides the reverse addition operator."""
        return other.addition(self)


    def __sub__(self, other):
        """Overrides the subtraction operator."""
        return self.subtract(other)


    def __rsub__(self, other):
        """Overrides the reverse subtraction operator."""
        return other.subtract(self)


    def __truediv__(self, other):
        """Overrides the division operator."""
        return self.divide(other)


    def __rtruediv__(self, other):
        """Overrides the reverse division operator."""
        return other.divide(self)


    def __xor__(self, vars_remove):
        """
        Overrides the XOR operator for max-marginalization.

        Args:
            vars_remove: Variables to marginalize over.

        Returns:
            MultinomialFactor: A new factor with max-marginalization applied.
        """
        if isinstance(vars_remove, str):
            return self.maxmarginalize(vars_remove)
        return self.maxmarginalize(*vars_remove)


    def __pow__(self, vars_remove):
        """
        Overrides the power operator for marginalization.

        Args:
            vars_remove: Variables to marginalize over.

        Returns:
            MultinomialFactor: A new factor with marginalization applied.
        """
        if isinstance(vars_remove, str):
            return self.marginalize(vars_remove)
        return self.marginalize(*vars_remove)


    @property
    def name(self):
        """
        Returns the name of the factor, formatted as P(left_vars|right_vars).

        Returns:
            str: The name of the factor.
        """
        vars_str = ",".join(self.left_vars)
        if self.right_vars:
            vars_str += "|" + ",".join(self.right_vars)
        return f"P({vars_str})"


    def __repr__(self):
        """
        Returns a string representation of the factor.

        Returns:
            str: A string representing the factor.
        """
        cardinality_dict = self.store.cardinality_dict
        card_str = ",".join(f"{v}:{cardinality_dict[v]}" for v in self._variables)
        return f"<{self.__class__.__name__} {self.name}, cardinality=({card_str}), " \
               f"values=[{self.store.values_str()}]>"


def random_multinomial(domain: Dict, right_vars: list = None, vtype=None, allow_zero=True):
    """
    Creates a random multinomial factor with probabilities sampled from a uniform distribution.

    Args:
        domain (Dict): A dictionary defining the domain of each variable.
        right_vars (list): Variables conditioned on (default is None).
        vtype: Type of value storage (default is DataStore.DEFAULT_STORE).
        allow_zero (bool): Whether to allow zero probabilities (default is True).

    Returns:
        MultinomialFactor: A randomly generated multinomial factor.
    """
    vtype = vtype or DataStore.DEFAULT_STORE
    right_vars = right_vars or []
    left_dims = [i for i, v in enumerate(domain.keys()) if v not in right_vars]

    if allow_zero:
        data = normalize_array(np.random.uniform(0, 1, size=[len(d) for d in domain.values()]), axis=left_dims)
    else:
        data = normalize_array(-1 * np.random.uniform(-1, 0, size=[len(d) for d in domain.values()]), axis=left_dims)
    return MultinomialFactor(domain=domain, values=data, right_vars=right_vars, vtype=vtype)


def uniform_multinomial(domain: Dict, right_vars: list = None, vtype=None):
    """
    Creates a uniform multinomial factor where all assignments have equal probability.

    Args:
        domain (Dict): A dictionary defining the domain of each variable.
        right_vars (list): Variables conditioned on (default is None).
        vtype: Type of value storage (default is DataStore.DEFAULT_STORE).

    Returns:
        MultinomialFactor: A uniformly distributed multinomial factor.
    """
    vtype = vtype or DataStore.DEFAULT_STORE
    right_vars = right_vars or []
    left_dims = [i for i, v in enumerate(domain.keys()) if v not in right_vars]
    data = normalize_array(np.ones([len(d) for d in domain.values()]), axis=left_dims)
    return MultinomialFactor(domain=domain, values=data, right_vars=right_vars, vtype=vtype)

def random_deterministic(dom: Dict, right_vars: list = None, vtype=None):
    """
    Creates a deterministic factor with randomly assigned deterministic states.

    Args:
        dom (Dict): A dictionary defining the domain of each variable.
        right_vars (list): Variables conditioned on (default is None).
        vtype: Type of value storage (default is DataStore.DEFAULT_STORE).

    Returns:
        MultinomialFactor: A deterministic multinomial factor.
    """

    vtype = vtype or DataStore.DEFAULT_STORE
    data = np.zeros([len(d) for d in dom.values()])
    for idx in [list(s.values()) for s in random_assignment(to_numeric_domains(dom), right_vars)]:
        set_value(1.0, data, idx)
    return MultinomialFactor(domain=dom, right_vars=right_vars, values=data, vtype=vtype)

def canonical_multinomial(domain: Dict, exo_var: Hashable, right_endo_vars: list = None,
                          vtype=None) -> MultinomialFactor:
    """
    Creates a canonical multinomial factor from a deterministic one.

    Args:
        domain (Dict): A dictionary defining the domain of each variable.
        exo_var (Hashable): The exogenous variable.
        right_endo_vars (list): Endogenous variables to condition on (default is None).
        vtype: Type of value storage (default is DataStore.DEFAULT_STORE).

    Returns:
        MultinomialFactor: A canonical multinomial factor.
    """

    from bcause.factors.deterministic import canonical_deterministic
    return canonical_deterministic(domain, exo_var, right_endo_vars, vtype).to_multinomial()

def canonical_for_model(model, domains, x):
    """
    Constructs a canonical multinomial factor for a variable in a given model.

    Args:
        model: The probabilistic model defining relationships.
        domains: The domains of all variables in the model.
        x: The target variable for which to create the canonical factor.

    Returns:
        MultinomialFactor: The canonical multinomial factor for the target variable.

    Raises:
        ValueError: If the model is not Markovian.
    """

    exoPa = model.get_exogenous_parents(x)
    exovar = exoPa[0]

    if len(exoPa) > 1 or len(model.get_edogenous_children(exovar)) != 1:
        raise ValueError("Method only valid for Markovian models")

    right_endoVars = model.get_edogenous_parents(x)
    endoVars = right_endoVars + [x]
    dom = dutils.subdomain(domains, *endoVars)
    return canonical_multinomial(dom, exovar, right_endoVars).reorder(*endoVars)


if __name__ == "__main__":

    # P(B|A)
    left_domain = dict(A=["a1","a2"])
    right_domain = dict(B=[0,1, 3])
    domain = {**left_domain, **right_domain}

    data=np.array([[0.5, .4, 1.0], [0.3, 0.6, 0.1]])
    f = MultinomialFactor(domain, data, right_vars=["A"])

    f2 = MultinomialFactor(left_domain, values= [0.1, 0.9])
    f - f2

