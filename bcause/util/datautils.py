from bcause.factors import MultinomialFactor
import bcause.util.domainutils as dutils


def to_counts(domains, data=None, normalize=False):
    domains = domains.copy()
    if data is not None:
        data = data[list(domains.keys())]
        dcounts = data.value_counts(dropna=False).to_dict()
        # Add None where missing values
        for x in [v for v in data.columns if data.isna().any()[v]]:
            if not all([type(v)==str for v in domains[x]]):
                raise ValueError("Missing values are only compatible in variables with str-like domains")
            domains[x] = domains[x] + ["nan"]
        data = data.fillna("nan")
    else:
        dcounts = dict()

    data_counts = [0 if k not in dcounts else dcounts[k] for k in dutils.state_space(domains)]
    if normalize:
        N = sum(data_counts)
        data_counts = [c/N for c in data_counts]
    return MultinomialFactor(domains, data_counts)


def filter_data(data, obs):
    for k,v in obs.items():
        data = data.loc[data[k]==v]
    return data
