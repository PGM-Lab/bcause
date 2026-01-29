#sbatch  --export=nparents=2,nzr=10 -n 100 experiments.sbs
#sbatch  --export=nparents=2,nzr=08 -n 100 experiments.sbs
#sbatch  --export=nparents=2,nzr=06 -n 100 experiments.sbs
#sbatch  --export=nparents=2,nzr=04 -n 100 experiments.sbs
#sbatch  --export=nparents=2,nzr=02 -n 100 experiments.sbs

#sbatch  --export=nparents=3,nzr=10 -n 100 experiments.sbs
#sbatch  --export=nparents=3,nzr=08 -n 100 experiments.sbs
#sbatch  --export=nparents=3,nzr=06 -n 100 experiments.sbs
#sbatch  --export=nparents=3,nzr=04 -n 100 experiments.sbs
#sbatch  --export=nparents=3,nzr=02 -n 100 experiments.sbs

sbatch  --export=nparents=1 -n 100 experiments.sbs
sbatch  --export=nparents=3 -n 100 experiments.sbs
sbatch  --export=nparents=2 -n 100 experiments.sbs

