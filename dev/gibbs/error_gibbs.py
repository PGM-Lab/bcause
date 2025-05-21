from bcause.inference.causal.multi import EMCC, GibbsCausal
from bcause.models.cmodel import StructuralCausalModel
import pandas as pd
modelfile = "./dev/gibbs/simple_nparents1_nzr02_zdr05_ysize3_3.uai"

#### Código para cargar el modelo #####
model = StructuralCausalModel.read(modelfile)
data = pd.read_csv(modelfile.replace(".uai", ".csv"), index_col=0)

non_root = [v for v in model.endogenous if len(model.get_edogenous_parents(v)) > 0]
root = sorted([v for v in model.endogenous if len(model.get_edogenous_parents(v)) == 0])

# Rename variables
model_endo_names = {v: v.replace("V", "X") for v in root}
model_endo_names["V0"] = "Y"
data_names = {k.replace("V", ""): v for k, v in model_endo_names.items()}
model_exo_names = {u: "U" + model_endo_names[model.get_children(u)[0]].lower() for u in model.exogenous}

model = model.rename_vars({**model_endo_names, **model_exo_names})
data = data.rename(columns=data_names)

######


#inf1 = EMCC(model,data)
#inf1.prob_sufficiency("X1", "Y", true_false_cause=(1,0), true_false_effect=(1,0))


model = model.randomize_factors(model.exogenous)

inf2 = GibbsCausal(model, data, num_runs=10, burnin_iter=10)

for _ in inf2.compile_incremental(step_runs=1):

    p = inf2.prob_sufficiency("X1", "Y", true_false_cause=(1,0), true_false_effect=(1,0))
    print(p)

    print(inf2.models[-1].factors["Uy"])

#inf2.prob_necessity("X1", "Y", true_false_cause=(1,0), true_false_effect=(1,0))




inf2.models[-1].factors["Uy"]

