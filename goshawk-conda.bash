#!/bin/bash

#Account and Email Information
#SBATCH -A bwilder
#SBATCH --mail-type=end
#SBATCH --mail-user=brentwilder@boisestate.edu

#SBATCH -J CONDA                                                # job name
#SBATCH -o /bsuhome/bwilder/scratch/SPHERES/slurm/log_slurm.o%j  # output and error file name (%j expands to jobID)
#SBATCH -p bsudfq                                                   # queue (partition)
#SBATCH -N 1                                                           # Number of nodes
#SBATCH --ntasks 10                                                    # Number of tasks 
#SBATCH -t 00-00:30:00                                                 # run time (d-hh:mm:ss)  


conda env create -f conda.yml
#conda env update --file conda.yml --prune