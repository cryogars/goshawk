#!/bin/bash

#Account and Email Information
#SBATCH -A YOUR_USER_NAME_HERE
#SBATCH --mail-type=end
#SBATCH --mail-user=YOUR_EMAIL_HERE@boisestate.edu

#SBATCH -J GOSHAWK_pt2                      # job name
#SBATCH -o ./slurm/log_slurm.o%j            # output and error file name (%j expands to jobID)
#SBATCH -p bsudfq                           # queue (partition)
#SBATCH -N 4                                # Number of nodes
#SBATCH --ntasks 192                        # Number of tasks (48/node max Borah)
#SBATCH -t 00-01:00:00                      # run time (d-hh:mm:ss) 
ulimit -v unlimited
ulimit -s unlimited

module purge
module load slurm


# USER INPUTS
dem='Copernicus' # Copernicus, SRTM, or 3DEP
path_to_img_base='/bsuhome/bwilder/scratch/goshawk-mu/prisma/BECK2/PRS_20210429180418_20210429180422_0001'
path_to_libradtran_bin='/bsuhome/bwilder/scratch/SPHERES/libRadtran-2.0.4/bin'
service_account='brentonwilder1995@brent-snow.iam.gserviceaccount.com'
ee_json='/bsuhome/bwilder/scratch/SPHERES/brent-snow.json'
optimal_cosi='yes' #yes or no
impurity_type='Dust' #Dust or Soot or None
# END USER INPUTS


conda activate goshawk
mpirun -np $SLURM_NTASKS python3 -m mpi4py.futures ./scripts/pipeline_pt2.py --dem $dem --img $path_to_img_base --lrt $path_to_libradtran_bin --ee_account $service_account --ee_json $ee_json --mu $optimal_cosi --impurity $impurity_type --n_cpu $SLURM_NTASKS


