#!/bin/bash

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=25

#SBATCH --time=03:30:00

#SBATCH --partition=tue.default.q

#SBATCH --job-name=shuntcurrents_1bar_external

#SBATCH --output=output.log

#SBATCH --error=error.log

#SBATCH --mail-type=END,FAIL

#SBATCH --mail-user=b.e.acosta.angulo@tue.nl



module load Anaconda3

source ~/.bashrc

conda activate fenicsx-env

which python
python -m pip list | grep pymoo

srun python optim_external.py
