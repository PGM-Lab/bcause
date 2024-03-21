import warnings

import pandas as pd

from bcause import DiscreteFactor, MultinomialFactor
from bcause.inference.probabilistic.datainference import LaplaceInference



def mutual_info_dist(joint:DiscreteFactor, marg1:DiscreteFactor, marg2:DiscreteFactor) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ((joint / (marg1 * marg2)).log() * joint).sum_all(masked_invalid=True)



def mutual_info_data(data, X, Y, Z=None, domains=None):
    variables = [X, Y]

    if Z is not None:
        if not isinstance(Z, list):
            variables += Z
        else:
            variables += [Z]

    # Determine the domains
    domains = domains or {v: list(data[v].unique()) for v in variables}
    inf = LaplaceInference(data, domains)

    # Inferece from the data
    inf = LaplaceInference(data, domains)

    # compute the joint and the marginals
    joint = inf.query([X, Y], conditioning=Z)
    marg1 = inf.query(X, conditioning=Z)
    marg2 = inf.query(Y, conditioning=Z)



    # compute the mutual information
    return mutual_info_dist(joint, marg1, marg2)



joint = MultinomialFactor(domain=dict(X=[0,1,2], Y=[0,1], Z=[0,1]), right_vars="Z", values=[0,.2,.2, .1,.0,.5, 0,.2,.2, .1,.0,.5])
marg1 = joint.marginalize("X")
marg2 = joint.marginalize("Y")

data = pd.concat([joint.R(Z=z).sample(1000, as_pandas=True) for z in (0,1)], ignore_index=True)
data.loc[0:1000, "Z"] = 0
data.loc[1000:2000, "Z"] = 1

mutual_info_data(data, "X","Y","Z")
