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


def to_numeric_dataset(data, dom):
    data = data.copy()

    for v in data.columns:
        data[v] = data[v].apply(lambda x: dom[v].index(x))

    return data


###

def to_uai_data(data, model):
    out_data = to_numeric_dataset(data, model.domains)

    var_order = list(reversed(list(model.domains.keys())))
    var_order = [v for v in var_order if v in model.endogenous]

    out_data = out_data[var_order]

    new_names = {var_order[i]: len(var_order) - i - 1 for i in range(len(var_order))}
    out_data = out_data.rename(columns=new_names).reset_index(drop=True)
    return out_data

