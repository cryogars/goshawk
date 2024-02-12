# Import libraries
import os
import time
import argparse
import itertools
import logging
import pickle
from datetime import datetime
import numpy as np
import multiprocessing
from spectral import *
import hytools as ht
import rasterio as rio
from osgeo import gdal

from terrain import get_surface
from matching import match_combo
from postprocessing import snow_tifs


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
    logging.basicConfig(filename=f'{log_dir}/logging_{log_date}_PT3.log', level=logging.DEBUG)
    logging.info(f'***STARTING SNOW ALBEDO MODEL_PT3***')

    # Load sorted list of opt results
    with open(f'{path_to_img_base}_albedo/combo_results.pkl', 'rb') as f:
        combo_results = pickle.load(f)
    logging.info(f'The resulting number of optimizations in combos: {len(combo_results)}')

    uniid_array = np.array(gdal.Open(f'{path_to_img_base}_albedo/clustering/kmeans.tif').ReadAsArray())

    # get rows and cols again
    rad_path = path_to_img_base + '_rdn_prj'
    observed_rad = envi.open(rad_path+'.hdr')
    observed_rad_array = observed_rad.open_memmap(writeable=True)
    rows = observed_rad_array.shape[0]
    cols = observed_rad_array.shape[1]
    spec_bands = observed_rad_array.shape[2]
    hy_obj = ht.HyTools()
    hy_obj.read_file(rad_path)
    sensor_wavelengths = hy_obj.wavelengths
    logging.info(f'Image has {rows} rows, {cols} columns, and {spec_bands} bands.')

    # Create empty uni_id and append to get order 
    # and extract index in matching
    uniid_list = []
    for z in combo_results:
        uniid_list.append(z[0])

    # Save combo as array for matching
    combo_copy = np.array(combo_results, dtype = 'object')
    
    # Load in cosi, Land cover and canopy percent arrays
    cosi = np.array(gdal.Open(f'{path_to_img_base}_albedo/dem_{dem}/cos_i.tif').ReadAsArray())
    lc = np.array(gdal.Open(f'{path_to_img_base}_albedo/landcover/landcover_prj.tif').ReadAsArray())
    cc = np.array(gdal.Open(f'{path_to_img_base}_albedo/canopy/canopy_prj.tif').ReadAsArray()) / 100
    cloud_mask  = np.load(f'{path_to_img_base}_albedo/cloud_mask.npy')
     
    # Parallel processing
    def my_func(t, def_param=[observed_rad_array, cosi, lc, cc, uniid_array, combo_copy, uniid_list, sensor_wavelengths,cloud_mask]):
        i = t[0]
        j = t[1]
        return match_combo(i,j,observed_rad_array, cosi, lc, cc, uniid_array, combo_copy, uniid_list, sensor_wavelengths,cloud_mask)

    pixel_list = [(i,j) for i,j in itertools.product(range(0,rows), range(0,cols))]
    
    # MP
    mp_time = time.time()
    logging.info('Matching started.')
    with multiprocessing.Pool() as pool:
        results = pool.map(my_func, pixel_list)
    logging.info('Matching finished in %s seconds.' %(time.time() - mp_time))

    # Post-processing step 
    snow_tifs(results, cosi, path_to_img_base, dem, cc, optimal_cosi)
    logging.info(f'***ENDING SNOW ALBEDO MODEL_PT3***')