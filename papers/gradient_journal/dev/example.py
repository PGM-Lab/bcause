import pandas as pd

from bcause.models.cmodel import StructuralCausalModel
from bcause.util.equtils import seq_to_pandas

modelfile = "papers/gradient_journal/models/literature/pearl_small.bif"
datapath = "papers/gradient_journal/models/literature/pearl_small.csv"

model = StructuralCausalModel.read(modelfile)
data = pd.read_csv(datapath)

# show the equations


seq_to_pandas(model.factors["S"], exovar="U")
seq_to_pandas(model.factors["T"], exovar="V")

