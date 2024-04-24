from bcause.util.datautils import filter_and, filter_not


def discr_score(data, positive_class, deprived_group):
    data_priv = filter_not(data, **deprived_group)
    data_depr = filter_and(data, **deprived_group)

    ppriv = len(filter_and(data_priv, **positive_class)) / len(data_priv)
    pdepr = len(filter_and(data_depr, **positive_class)) / len(data_depr)

    return ppriv - pdepr