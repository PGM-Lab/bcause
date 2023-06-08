from bcause.factors.mulitnomial import MultinomialFactor
from bcause.util.domainutils import assingment_space
from bcause.factors.values.store import __ALL__ as store_types, DataStore, store_dict

import numpy as np
import pytest

from numpy.testing import assert_array_almost_equal




def build_factors(vtype=None):

    # domains
    domain = dict(A=["a1", "a2"], B=[0, 1, 3])

    #type(MultinomialFactor(dict(A=["a1", "a2"]), values=[0.2, 0.8], vtype=vtype).store)

    # testing factors
    marg = MultinomialFactor(dict(A=["a1", "a2"]), values=[0.2, 0.8], vtype=vtype)
    cond = MultinomialFactor(domain, values=[[0.2, 0.1, 0.7], [0.3, 0.6, 0.1]], right_vars=["A"], vtype=vtype)
    join = MultinomialFactor(domain, values= [[0.1, 0.15, 0.15], [0.2, 0.4, 0.0]], vtype=vtype)
    marg.store
    return cond, marg, join


@pytest.mark.parametrize("vtype", store_types)
def test_default_store(vtype):
    cond, _, _ = build_factors(vtype)
    assert type(cond.store) == store_dict[vtype]
    DataStore.set_default(vtype)
    cond, _, _ = build_factors()
    assert type(cond.store) == store_dict[vtype]


@pytest.mark.parametrize("vtype", store_types)
def test_sum_values(vtype):
    cond, marg, join = build_factors(vtype)
    assert join.store.sum_all() == 1.0
    assert marg.store.sum_all() == 1.0
    assert cond.store.sum_all() == 2.0

@pytest.mark.parametrize("vtype", store_types)
def test_multiply(vtype):
    cond, marg, join = build_factors(vtype)

    f = cond * marg
    actual = f.values
    expected = [0.04, 0.02, 0.14, 0.24, 0.48, 0.08]
    assert_array_almost_equal(actual, expected)

@pytest.mark.parametrize("vtype", store_types)
def test_marginalize(vtype):
    cond, marg, join = build_factors(vtype)
    f = join**"A"
    actual = f.values
    expected = [0.3, 0.55, 0.15]
    assert_array_almost_equal(actual, expected)

    f = join**("A","B")
    actual = f.values
    expected = [1.0]
    assert_array_almost_equal(actual, expected)

@pytest.mark.parametrize("vtype", store_types)
def test_maxmarginalize(vtype):

    try:
        cond, marg, join = build_factors(vtype)

        # max-marginalize
        f = join^"A"
        actual = f.values
        expected = [0.2, 0.4, 0.15]
        assert_array_almost_equal(actual, expected)

        f = join^("A","B")
        actual = f.values
        expected = [0.4]
        assert_array_almost_equal(actual, expected)
    except NotImplementedError:
        print("Warning: operation not implemented")


@pytest.mark.parametrize("vtype", store_types)
def test_addition(vtype):
    cond, marg, join = build_factors(vtype)
    # sum
    f = cond + marg
    actual = f.values
    expected = [0.4, 0.3, 0.9, 1.1, 1.4, 0.9]
    assert_array_almost_equal(actual, expected)

@pytest.mark.parametrize("vtype", store_types)
def test_subtract(vtype):
    cond, marg, join = build_factors(vtype)

    # subtract
    f = cond - marg
    actual = f.values
    expected = [0.0, -0.1, 0.5, -0.5, -0.2,-0.7]
    assert_array_almost_equal(actual, expected)

@pytest.mark.parametrize("vtype", store_types)
def test_sampling(vtype):
    cond, marg, join = build_factors(vtype)

    np.random.seed(0)
    dataset = join.sample(100)
    assert sum(join.log_prob(dataset)) == -151.26474464995295

@pytest.mark.parametrize("vtype", store_types)
def test_values_array(vtype):
    cond, marg, join = build_factors(vtype)

    actual = marg.values_array()
    expected = np.array([0.2, 0.8])
    assert_array_almost_equal(actual, expected)

    actual = cond.values_array()
    expected = np.array([[0.2, 0.1, 0.7],[0.3, 0.6, 0.1]])


@pytest.mark.parametrize("vtype", store_types)
def test_values_dict(vtype):
    cond, marg, join = build_factors(vtype)

    actual = cond.values_dict[('A', 'a1'), ('B', 1)]
    expected = 0.1
    assert actual==expected

    actual = cond.values_dict[('A', 'a2'), ('B', 3)]
    expected = 0.1
    assert actual==expected

    actual = marg.values_dict[(('A', 'a2'),)]
    expected = 0.8
    assert actual==expected


@pytest.mark.parametrize("vtype", store_types)
def test_values_dict(vtype):

    cond, marg, join = build_factors(vtype)

    actual = cond.values_list
    expected = [0.2, 0.1, 0.7, 0.3, 0.6, 0.1]
    assert actual==expected


    actual = join.values_list
    expected = [0.1, 0.15, 0.15, 0.2, 0.4, 0.0]
    assert actual==expected


    actual = marg.values_list
    expected = [0.2, 0.8]
    assert actual==expected