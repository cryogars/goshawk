# Import libraries
import os
import logging
import pickle
import argparse
import subprocess
from datetime import datetime
import numpy as np
from spectral import *
from osgeo import gdal
import dateutil.parser
from multiprocessing.dummy import Pool
import hytools as ht

from terrain import get_surface
from clouds import simple_cloud_threshold
from shadow import run_ray_pool
from earthengine import get_landcover, get_canopy_cover
from clustering import kmeans_grouping
from libradtran import lrt_prepper, write_lrt_inp,write_lrt_inp_irrad, lut_grid
from postprocessing import kmeans_tifs



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

    # Get date from file string
    if 'EMIT' in path_to_img_base:
        date_string = path_to_img_base.split("EMIT_L2A_RAD_",1)[1]
        date_string = date_string[:15]
    else: # PRISMA
        date_string = path_to_img_base.split("PRS_",1)[1]
        date_string = date_string[:14]
    
    obs_time = dateutil.parser.parse(date_string)
    year = int(obs_time.year)
    doy = int(obs_time.strftime('%j'))

    # Check whether log directory exists, if not, make one
    log_dir = f'{path_to_img_base}_albedo/logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # same for libradtran directory
    lrt_dir = f'{path_to_img_base}_albedo/libradtran'
    if not os.path.exists(lrt_dir):
        os.makedirs(lrt_dir)

    # Create log file
    log_date = datetime.now()
    logging.basicConfig(filename=f'{log_dir}/logging_{log_date}_PT1.log', level=logging.INFO)
    logging.info(f'***STARTING SNOW ALBEDO MODEL_PT1***')

    # Get DEM and other terrain information
    bounds, bounds_small= get_surface(dem, path_to_img_base, n_cpu)

    # Open all of the terrain and view arrays
    dem_dir = f'{path_to_img_base}_albedo/dem_{dem}'
    selected_elev = np.array(gdal.Open(f'{dem_dir}/dem_prj.tif').ReadAsArray())
    selected_cosi = np.array(gdal.Open(f'{dem_dir}/cos_i.tif').ReadAsArray())
    selected_cosv = np.array(gdal.Open(f'{dem_dir}/cos_v.tif').ReadAsArray())
    selected_theta = np.array(gdal.Open(f'{dem_dir}/theta.tif').ReadAsArray())
    selected_svf = np.array(gdal.Open(f'{dem_dir}/svf.tif').ReadAsArray())
    selected_slope = np.array(gdal.Open(f'{dem_dir}/slope.tif').ReadAsArray())
    selected_aspect = np.array(gdal.Open(f'{dem_dir}/aspect.tif').ReadAsArray())
    sza_array = np.array(gdal.Open(f'{dem_dir}/sza.tif').ReadAsArray())
    saa_array = np.array(gdal.Open(f'{dem_dir}/saa.tif').ReadAsArray())

    # Lat / lon mean 
    lat_mean = (bounds_small[1] + bounds_small[3]) / 2
    lon_mean = (bounds_small[0] + bounds_small[2]) / 2

    # Get WorldCover land cover dataset from earth engine
    landcover = get_landcover(path_to_img_base, bounds, dem, service_account, ee_json)
    landcover = np.array(gdal.Open(f'{path_to_img_base}_albedo/landcover/landcover_prj.tif').ReadAsArray())

    # Get canopy cover
    canopy = get_canopy_cover(path_to_img_base, bounds, dem, service_account, ee_json)

    # Get rad-array, rows, cols,#bands, and sensor_wavelengths
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
    
    # Check for clouds
    cloud_mask = simple_cloud_threshold(observed_rad_array, sensor_wavelengths)
    
    # Run shade mask ray tracing for local terrain shading
    # First fills list quickly using multiprocessing
    shade_args = []
    for a in range(selected_elev.shape[0]):
        for b in range(selected_elev.shape[1]):
            shade_args.append([a, b, selected_elev, selected_cosi[a,b],
                               sza_array[a,b], saa_array[a,b], path_to_img_base])
    ray_trace_results = run_ray_pool(shade_args, n_cpu=n_cpu)

    #Then, fills the array structure.
    shadow_arr = np.empty_like(selected_elev)
    for z in ray_trace_results:
        shadow_arr[z[0], z[1]] = z[2]

    # Get params ready for LRT
    vza, umu, phi0, phi, sza, lat_inp, lon_inp, alt_min, alt_max, atmos = lrt_prepper(path_to_img_base, 
                                                                                                sza_array, 
                                                                                                selected_elev)
    path_to_libradtran_base = os.path.dirname(path_to_libradtran_bin)

    # Run the LRT LUT pipeline
    path_to_libradtran_base = os.path.dirname(path_to_libradtran_bin)
    h20_range = [1,25, 50]
    a550_range = [0.01,0.5,1.0]
    alt_range = [alt_min, alt_max]
    lrt_inp = []
    lrt_inp_irrad = []
    for h in h20_range:
        for aod in a550_range:
            for altitude_km in alt_range:
                
                cmd = write_lrt_inp(h,aod,0, ['toa','uu'], umu, phi0, phi, sza, 
                                lat_inp, lon_inp, doy, altitude_km, atmos, path_to_libradtran_bin, 
                                lrt_dir, path_to_libradtran_base)
                lrt_inp.append([cmd,path_to_libradtran_bin])
                
                cmd = write_lrt_inp(h,aod,0, ['sur','eglo'], umu, phi0, phi, vza, 
                                lat_inp, lon_inp, doy, altitude_km, atmos, path_to_libradtran_bin, 
                                lrt_dir, path_to_libradtran_base)
                lrt_inp.append([cmd,path_to_libradtran_bin])

                cmd = write_lrt_inp(h,aod,0.15, ['sur','eglo'], umu, phi0, phi, sza, 
                                lat_inp, lon_inp, doy, altitude_km, atmos, path_to_libradtran_bin, 
                                lrt_dir, path_to_libradtran_base)
                lrt_inp.append([cmd,path_to_libradtran_bin])
                
                cmd = write_lrt_inp(h,aod,0.5, ['sur','eglo'], umu, phi0, phi, sza, 
                                lat_inp, lon_inp, doy, altitude_km, atmos, path_to_libradtran_bin, 
                                lrt_dir, path_to_libradtran_base)   
                lrt_inp.append([cmd,path_to_libradtran_bin])

                cmd = write_lrt_inp_irrad(h,aod,0, ['toa','uu'], umu, phi0, phi, sza, 
                                lat_inp, lon_inp, doy, altitude_km, atmos, path_to_libradtran_bin, 
                                lrt_dir, path_to_libradtran_base)
                lrt_inp_irrad.append([cmd,path_to_libradtran_bin])
    
    def spawn(cmd, path_to_libradtran_bin):   
        exit = subprocess.run(cmd, shell=True, cwd=path_to_libradtran_bin)

    def spawn_pool(args):
        return spawn(*args)

    pool = Pool(max(1,n_cpu-12)) # this had to be tinkered with.. RAM req of specific node    
    pool.map(spawn_pool, lrt_inp)
    pool.map(spawn_pool, lrt_inp_irrad)
    logging.info(f'Completed running LRT for this image.')

    g_l0, g_tup, g_s, g_edir, g_edn = lut_grid(h20_range,a550_range, alt_range, 
                                               path_to_img_base, sensor_wavelengths)
    
    logging.info(f'Gridded LUT created.')
 

    # Run k-means
    combo_list, cluster_matches, spectra_dict, cosi_dict = kmeans_grouping(observed_rad_array, 
                                                                        cloud_mask, 
                                                                        selected_elev,
                                                                        selected_cosi,
                                                                        selected_cosv, 
                                                                        selected_theta,
                                                                        selected_svf, 
                                                                        landcover,
                                                                        selected_slope,
                                                                        selected_aspect,
                                                                        sensor_wavelengths,
                                                                        lat_mean, lon_mean, 
                                                                        shadow_arr, 
                                                                        sza,vza, phi0, phi,
                                                                        g_l0, g_tup, g_s, g_edir, g_edn,
                                                                        optimal_cosi,
                                                                        n_cpu)

    # Save the k-means clustering map to tiff
    uniid_array = kmeans_tifs(cluster_matches, selected_cosi, path_to_img_base, dem, cloud_mask, cosi_dict)
    logging.info(f'GOSHAWK running with {len(combo_list)} clusters.')

    # Dump lists to tmp pkl file
    with open(f'{path_to_img_base}_albedo/spectra_dict.pkl', 'wb') as f:
        pickle.dump(spectra_dict, f)
    with open(f'{path_to_img_base}_albedo/combo_list.pkl', 'wb') as f:
        pickle.dump(combo_list, f)
    # Dump LRT Grids to pkl for pt3
    # g_l0, g_tup, g_s, g_edir, g_edn
    with open(f'{path_to_img_base}_albedo/g_l0.pkl', 'wb') as f:
        pickle.dump(g_l0, f)
    with open(f'{path_to_img_base}_albedo/g_tup.pkl', 'wb') as f:
        pickle.dump(g_tup, f)
    with open(f'{path_to_img_base}_albedo/g_s.pkl', 'wb') as f:
        pickle.dump(g_s, f)
    with open(f'{path_to_img_base}_albedo/g_edir.pkl', 'wb') as f:
        pickle.dump(g_edir, f)
    with open(f'{path_to_img_base}_albedo/g_edn.pkl', 'wb') as f:
        pickle.dump(g_edn, f)
    np.save(f'{path_to_img_base}_albedo/cloud_mask.npy', cloud_mask)
    np.save(f'{path_to_img_base}_albedo/shadow_arr.npy', shadow_arr)
    logging.info(f'***ENDING SNOW ALBEDO MODEL_PT1***')
    