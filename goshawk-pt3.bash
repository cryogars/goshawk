#!/bin/bash

#Account and Email Information
#SBATCH -A YOUR_USER_NAME_HERE
#SBATCH --mail-type=end
#SBATCH --mail-user=YOUR_EMAIL_HERE@boisestate.edu

#SBATCH -J GOSHAWK_pt3                          # job name
#SBATCH -o ./slurm/log_slurm.o%j                # output and error file name (%j expands to jobID)
#SBATCH -p bsudfq                               # queue (partition)
#SBATCH -N 1                                    # Number of nodes
#SBATCH -n 48                                   # Number of cores
#SBATCH -t 00-00:10:00                          # run time (d-hh:mm:ss) 
ulimit -v unlimited
ulimit -s unlimited

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
python ./scripts/pipeline_pt3.py --dem $dem --img $path_to_img_base --lrt $path_to_libradtran_bin --ee_account $service_account --ee_json $ee_json --mu $optimal_cosi --impurity $impurity_type --n_cpu $SLURM_NTASKS
