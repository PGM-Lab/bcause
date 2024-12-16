from bcause.inference.causal.multi import EMCC, GDCC
from papers.gradient_journal.code.prepro import load_and_preprocess

model, data, exact_queries,_ = load_and_preprocess("./papers/gradient_journal/models/synthetic/simple/simple_nparents2_nzr02_zdr05_10.uai")

remove_outliers = True # test with both, True and False



# dataset with some exact queries
exact_queries

#inf = EMCC(model,data,num_runs=10, max_iter=100)
inf = GDCC(model, data, num_runs=20, tol=10e-5, outliers_removal=remove_outliers)

inf.compile()

inf.prob_necessity("X1","Y")



