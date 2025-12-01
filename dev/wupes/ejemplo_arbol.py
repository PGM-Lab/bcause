from bcause.learning.parameter.expectation_maximization import ExpectationMaximizationTrees
from bcause.util import randomUtil
from bcause.models.cmodel import StructuralCausalModel
import pandas as pd

filepath = "./models/WUPES/g2_model_27.bif"
datapath = "./models/WUPES/g2_data_27.csv"
model = StructuralCausalModel.read(filepath)
data = pd.read_csv(datapath, dtype='str')
models = []
for i in range(20):
    randomUtil.seed(i)
    em = ExpectationMaximizationTrees(model.randomize_factors(model.exogenous, allow_zero=False),ignore_convergence=True)
    em.run(data, max_iter=20)
    models.append(em.model)