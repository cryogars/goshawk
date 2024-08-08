#!/bin/bash

#Account and Email Information
#SBATCH -A YOUR_USER_NAME_HERE
#SBATCH --mail-type=end
#SBATCH --mail-user=YOUR_EMAIL_HERE@boisestate.edu

#SBATCH -J GOSHAWK_pt1                      # job name
#SBATCH -o ./slurm/log_slurm.o%j            # output and error file name (%j expands to jobID)
#SBATCH -p bsudfq                           # queue (partition)
#SBATCH -N 1                                # Number of nodes
#SBATCH --ntasks 48                         # Number of tasks 
#SBATCH -t 00-00:45:00                      # run time (d-hh:mm:ss)  
ulimit -v unlimited
ulimit -s unlimited
#SBATCH --exclusive

module load slurm
module load netcdf/gcc/64/gcc/64/4.7.3 
module load globalarrays/openmpi/gcc/64/5.7 
module load gcc/7.5.0 
module load openmpi/intel/4.0.4
module load hdf5/intel/1.10.5
module load gsl/gcc8/2.6 
cd /bsuhome/bwilder/scratch/SPHERES/libRadtran-2.0.4
PYTHON=$(which python2) ./configure --prefix=$(pwd)
make
cd /bsuhome/bwilder/scratch/goshawk-mu

conda activate goshawk

# USER INPUTS
dem='Copernicus' # Copernicus, SRTM, or 3DEP
path_to_img_base='/bsuhome/bwilder/scratch/goshawk-mu/prisma/BECK2/PRS_20210429180418_20210429180422_0001'
path_to_libradtran_bin='/bsuhome/bwilder/scratch/SPHERES/libRadtran-2.0.4/bin'
service_account='brentonwilder1995@brent-snow.iam.gserviceaccount.com'
ee_json='/bsuhome/bwilder/scratch/SPHERES/brent-snow.json'
optimal_cosi='yes' #yes or no
impurity_type='Dust' #Dust or Soot
n_cpu=48
n_nodes=4
# END USER INPUTS

python ./scripts/pipeline_pt1.py --dem $dem --img $path_to_img_base --lrt $path_to_libradtran_bin --ee_account $service_account --ee_json $ee_json --mu $optimal_cosi --impurity $impurity_type --n_cpu $n_cpu