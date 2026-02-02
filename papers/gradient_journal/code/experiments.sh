

METHOD=EMCC_50; sbatch  --job-name=$METHOD --export=method=$METHOD experiments.sbs
METHOD=EMCC_100; sbatch  --job-name=$METHOD --export=method=$METHOD experiments.sbs
METHOD=EMCC_150; sbatch  --job-name=$METHOD --export=method=$METHOD experiments.sbs


METHOD=GDCC_1e-05; sbatch --job-name=$METHOD --export=method=$METHOD experiments.sbs
METHOD=GDCC_1e-07; sbatch --job-name=$METHOD --export=method=$METHOD experiments.sbs
METHOD=GDCC_1e-09; sbatch --job-name=$METHOD --export=method=$METHOD experiments.sbs


METHOD=GSCC; sbatch --job-name=$METHOD --export=method=$METHOD experiments.sbs

