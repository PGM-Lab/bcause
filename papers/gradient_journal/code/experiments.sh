

sbatch  --export=method=EMCC_50 experiments.sbs
sbatch  --export=method=EMCC_100 experiments.sbs
sbatch  --export=method=EMCC_150 experiments.sbs

sbatch  --export=method=GDCC_1e-05 experiments.sbs
sbatch  --export=method=GDCC_1e-07 experiments.sbs
sbatch  --export=method=GDCC_1e-09 experiments.sbs

sbatch  --export=method=GSCC experiments.sbs