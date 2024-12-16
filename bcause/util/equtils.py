import bcause.util.domainutils as dutils

import pandas as pd
import pandas as pd

import bcause.util.domainutils as dutils


def seq_to_pandas(f, exovar):

    endoVars = [v for v in f.variables if v != exovar]
    endoDom = dutils.subdomain(f.domain, *endoVars)
    exocols = [f"{exovar}={s}" for s in f.domain[exovar]]
    colnames = endoVars + exocols

    df = pd.DataFrame(columns=colnames)

    for x in dutils.assingment_space(endoDom):
        values = [int(v) for v in f.R(**x).values]
        df = df._append({**x, **dict((zip(exocols, values)))}, ignore_index=True)

    return df