from bcause.factors.mulitnomial import MultinomialFactor
from bcause.util.domainutils import assingment_space

import numpy as np
import pytest

from numpy.testing import assert_array_almost_equal

# domains
domain = dict(A=["a1", "a2"], B=[0, 1, 3])

# testing factors
cond = MultinomialFactor(domain, data=[[0.2, 0.1, 0.7], [0.3, 0.6, 0.1]], right_vars=["A"])
marg = MultinomialFactor(dict(A=["a1", "a2"]), data=[0.2, 0.8])
join = MultinomialFactor(domain, data = [[0.1, 0.15, 0.15], [0.2, 0.4, 0.0]])



def test_sum_values():
    join.store.sum_all() == 1.0
    marg.store.sum_all() == 1.0
    cond.store.sum_all() == 2.0


def test_multiply():
    f = cond * marg
    actual = f.values
    expected = [0.04, 0.02, 0.14, 0.24, 0.48, 0.08]
    assert_array_almost_equal(actual, expected)


def test_marginalize():
    f = join**"A"
    actual = f.values
    expected = [0.3, 0.55, 0.15]
    assert_array_almost_equal(actual, expected)

    f = join**("A","B")
    actual = f.values
    expected = [1.0]
    assert_array_almost_equal(actual, expected)

def test_maxmarginalize():
    # max-marginalize
    f = join^"A"
    actual = f.values
    expected = [0.2, 0.4, 0.15]
    assert_array_almost_equal(actual, expected)

    f = join^("A","B")
    actual = f.values
    expected = [0.4]
    assert_array_almost_equal(actual, expected)

def test_addition():
    # sum
    f = cond + marg
    actual = f.values
    expected = [0.4, 0.3, 0.9, 1.1, 1.4, 0.9]
    assert_array_almost_equal(actual, expected)

def test_subtract():
    # subtract
    f = cond - marg
    actual = f.values
    expected = [0.0, -0.1, 0.5, -0.5, -0.2,-0.7]
    assert_array_almost_equal(actual, expected)

def test_sampling():
    np.random.seed(0)
    dataset = join.sample(100)
    assert sum(join.log_prob(dataset)) == -151.26474464995295


def test_values():
    actual = marg.to_values_array()
    expected = np.array([0.2, 0.8])
    assert_array_almost_equal(actual, expected)


    actual = cond.to_values_array()
    expected = np.array([[0.2, 0.1, 0.7],[0.3, 0.6, 0.1]])


