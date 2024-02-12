# Import libraries
import os
import time
import argparse
import logging
import pickle
import numpy as np
from datetime import datetime
from spectral import *
from mpi4py.futures import MPIPoolExecutor

from optimization import compute_pixels_mpi


if __name__ == '__main__':

    # Set req args
    parser = argparse.ArgumentParser(description='Model Snow Albedo')                
    parser.add_argument('--dem', type=str, required=True,
                        help='Please select either "Copernicus", "SRTM", or "3DEP" for dem')
    parser.add_argument('--img', type=str, required=True,
                        help='Path to image base')
    parser.add_argument('--lrt', type=str, required=True,
                        help='path to libRadtran bin')
    parser.add_argument('--ee_account', type=str, required=True,
                        help='EE Service Account')
    parser.add_argument('--ee_json', type=str, required=True,
                        help='path to EE json credentials')
    parser.add_argument('--mu', type=str, required=True,
                        help='optimal terrain?')
    parser.add_argument('--n_cpu', type=int, required=True,
                        help='number of cpus')  
    # Parse user args
    args = parser.parse_args()
    dem = args.dem
    path_to_img_base = args.img
    path_to_libradtran_bin = args.lrt
    service_account = args.ee_account
    ee_json = args.ee_json
    optimal_cosi = args.mu
    n_cpu = args.n_cpu

    # Check whether log directory exists, if not, make one
    log_dir = f'{path_to_img_base}_albedo/logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create log file
    log_date = datetime.now()
    logging.basicConfig(filename=f'{log_dir}/logging_{log_date}_PT2.log', level=logging.DEBUG)
    logging.info(f'***STARTING SNOW ALBEDO MODEL_PT2***')

    # Load in tmp list file
    with open(f'{path_to_img_base}_albedo/combo_list.pkl', 'rb') as f:
        combo_list = pickle.load(f)

    # Optimize model for each pixel using MPI
    opt_time = time.time()
    combo_results = []
    with MPIPoolExecutor() as pool:
        logging.info('MPI Pool started.')
        for r in pool.map(compute_pixels_mpi, combo_list):
            combo_results.append(r)
    logging.info('MPI Pool closed. Optimization finished in %s seconds.' %(time.time() - opt_time))
 
    with open(f'{path_to_img_base}_albedo/combo_results.pkl', 'wb') as f:
        pickle.dump(combo_results, f)

    logging.info(f'***ENDING SNOW ALBEDO MODEL_PT2***')