# Import libraries
import os
import logging
import numpy as np
import pandas as pd
from spectral import *
import rasterio as rio
import elevation
from dem_stitcher import stitch_dem
from osgeo import gdal
import py3dep
from dateutil import parser
from datetime import timezone
from pyproj import CRS
from pysolar import solar

from viewf import run_viewf_pool


def get_surface(dem, path_to_img_base, n_cpu):
    '''
    Select the Digital Elevation Model (DEM) and compute relevant parameters.

    Choices for DEM currently include 'SRTM', '3DEP', 'Copernicus', and 'User'.

    If 'User' is selected, the DEM must be projected to the correct UTM and named 'dem.tif',
    saved next to the input images.

    Args:
        dem (str): Choice of DEM.
        path_to_img_base (str): Base path for the input images.

    Returns:
        tuple: Tuple containing the bounding coordinates, selected cosine of incidence angle (cos_i),
               selected elevation array, and solar zenith angle (sza) array.
    '''

    # Check whether dem directory exists, if not, make one
    dem_dir = f'{path_to_img_base}_albedo/dem_{dem}'
    if not os.path.exists(dem_dir):
        os.makedirs(dem_dir)

    # Get the ENVI paths from SISTER
    loc_path = path_to_img_base + '_loc_prj'
    obs_path = path_to_img_base + '_obs_prj'

    # Check to make sure these files exist
    if os.path.isfile(loc_path+'.hdr'):
        observed_loc = envi.open(loc_path+'.hdr')
        observed_loc_array = observed_loc.open_memmap(writeable=True)
        # Find extents of the image
        west = np.min(np.delete(observed_loc_array[:, :, 0].flatten(), np.where(observed_loc_array[:, :, 0].flatten() == -9999)))
        east = np.max(np.delete(observed_loc_array[:, :, 0].flatten(), np.where(observed_loc_array[:, :, 0].flatten() == -9999)))
        south = np.min(np.delete(observed_loc_array[:, :, 1].flatten(), np.where(observed_loc_array[:, :, 1].flatten() == -9999)))
        north = np.max(np.delete(observed_loc_array[:, :, 1].flatten(), np.where(observed_loc_array[:, :, 1].flatten() == -9999)))
        bounds = west - 0.1, south - 0.1, east + 0.1, north + 0.1
        bounds_small = [float(west), float(south), float(east), float(north)]

    else:
        raise Exception('see GITHUB readme for the exact filenames needed at this time...')
    
    # Check to make sure these files exist
    if os.path.isfile(obs_path+'.hdr'):
        observed_obs = envi.open(obs_path+'.hdr')
        observed_obs_array = observed_obs.open_memmap(writeable=True)
    else:
        raise Exception('see GITHUB readme for the exact filenames needed at this time...')


    # Update elevation and local solar zenith angle based on DEM selection
    if dem == 'SRTM' or dem == '3DEP' or dem == 'Copernicus' or dem == 'User':

        # Set paths for dem outputs
        dem_download = f'{dem_dir}/dem.tif'
        dem_prj = f'{dem_dir}/dem_prj.tif'
        slope = f'{dem_dir}/slope.tif'
        aspect = f'{dem_dir}/aspect.tif'     
        cos_i = f'{dem_dir}/cos_i.tif'
        cos_v = f'{dem_dir}/cos_v.tif'
        theta_path =f'{dem_dir}/theta.tif' 

        if os.path.exists(cos_i):
            dem = True
        
        if dem is True:
            pass

        # Download SRTM DEM
        elif dem == 'SRTM':
            # Download SRTM using 'elevation' library
            elevation.clip(bounds=bounds, output=dem_download, product='SRTM1')

        # Download 3DEP DEM
        elif dem == '3DEP':
            dem_3dep = py3dep.get_map('DEM', bounds, resolution=30, crs='EPSG:4326')
            dem_3dep.rio.to_raster(dem_download)

        # Copy over DEM from User
        elif dem =='User':
            user_dem_path = os.path.normpath(path_to_img_base + os.sep + os.pardir)
            os.system(f'gdal_calc.py -A {user_dem_path}dem.tif --A_band=1 \
                        --outfile={dem_download} --overwrite \
                        --calc="A" --quiet')

        # Download Copernicus
        else:
            X, p = stitch_dem(bounds,
                              dem_name='glo_30')  # Global Copernicus 30 meter resolution DEM
            with rio.open(dem_download, 'w', **p) as ds:
                 ds.write(X, 1)

        ref_raster = rio.open(loc_path)
        crs = ref_raster.crs  # ASSUMING USED UTM PROJECTION OPTION IN SISTER
        west, south, east, north = ref_raster.bounds
        
        # if first three letters are EMIT, 60m.. else is PRISMA at 30m
        if 'EMIT' in path_to_img_base:
            dem_spacing = 60
        else: # PRISMA
            dem_spacing = 30
        
        os.system(f'gdalwarp -r bilinear -t_srs {crs} -te {west} {south} {east} {north} \
                    -tr {dem_spacing} {dem_spacing} -overwrite {dem_download} {dem_prj} -q')           

        # Compute slope and aspect with gdal
        os.system(f'gdaldem slope -compute_edges {dem_prj} {slope} -q')
        os.system(f'gdaldem aspect -compute_edges -zero_for_flat {dem_prj} {aspect} -q')

        # Compute SVF with topo-calc from USDA-ARS (modified using Dozier 2022 equation)
        svf_path = f'{dem_dir}/svf.tif'
        dem_svf = (np.array(gdal.Open(f'{dem_dir}/dem_prj.tif').ReadAsArray())).astype('double')
        svf = run_viewf_pool(n_cpu, dem_svf, dem_spacing)
        ref_raster = rio.open(dem_prj)
        ras_meta = ref_raster.profile
        with rio.open(svf_path, 'w', **ras_meta) as dst:
            dst.write(svf, 1)

        # Compute solar and view angles
        saa_path = f'{dem_dir}/saa.tif'
        sza_path = f'{dem_dir}/sza.tif'

        # Compute cos_i 
        # ~~~~~~~~~~~~~
        ref_raster = rio.open(dem_prj)
        crs = ref_raster.crs  # ASSUMING USED UTM PROJECTION OPTION IN SISTER
        ras_meta = ref_raster.profile

        # Create mask for sza and saa raster
        array = observed_loc_array[:, :, 0]
        mask = np.copy(array)
        mask[mask != -9999] = 1
        mask[mask == -9999] = 0

        # Get date from file string
        if 'EMIT' in path_to_img_base:
            date_string = path_to_img_base.split("EMIT_L2A_RAD_",1)[1]
            date_string = date_string[:15]
        else: # PRISMA
            date_string = path_to_img_base.split("PRS_",1)[1]
            date_string = date_string[:14]
        obs_time = parser.parse(date_string).replace(tzinfo=timezone.utc)

        # Compute SAA and SZA rasters. Assumes first UTM time.
        solar_az = solar.get_azimuth(observed_loc_array[:, :, 1],
                                        observed_loc_array[:, :, 0],
                                        obs_time)
        
        solar_zn = 90-solar.get_altitude(observed_loc_array[:, :, 1],
                                        observed_loc_array[:, :, 0],
                                        obs_time)

        #nodata mask for this is 0
        ras_meta['nodata'] = 0
        solar_zn = solar_zn * mask
        solar_az = solar_az * mask

        with rio.open(sza_path, 'w', **ras_meta) as dst:
            dst.write(solar_zn, 1)

        with rio.open(saa_path, 'w', **ras_meta) as dst:
            dst.write(solar_az, 1)

        os.system(f'gdal_calc.py -A {saa_path} -B {sza_path} -C {slope} -D {aspect} \
                    --outfile={cos_i} --overwrite \
                    --calc="sin(B*(pi/180))*sin(C*(pi/180))*cos(A*(pi/180)-D*(pi/180))+cos(B*(pi/180))*cos(C*(pi/180))" --quiet') 
        
        # Create cos_v array (cosine of local view zenith angle)
        s = np.array(gdal.Open(f'{dem_dir}/slope.tif').ReadAsArray())
        a = np.array(gdal.Open(f'{dem_dir}/aspect.tif').ReadAsArray())
        cos_v_array = np.sin(observed_obs_array[:,:,2]*(np.pi/180))*np.sin(s*(np.pi/180))*np.cos(observed_obs_array[:,:,1]*(np.pi/180)-a*(np.pi/180))+np.cos(observed_obs_array[:,:,2]*(np.pi/180))*np.cos(s*(np.pi/180))
        cos_v_array[solar_zn==0] = -9999
        ras_meta['nodata'] = -9999
        with rio.open(cos_v, 'w', **ras_meta) as dst:
                        dst.write(cos_v_array, 1)


        # Create THETA array (used in computation for snow reflectance)
        # This is scattering angle with RAA notation to match ART usage
        # see PyICE and the rest of their work.
        # RAA = 180 (vaa - saa)
        cosv = np.copy(cos_v_array)
        cos_i = np.array(gdal.Open(f'{dem_dir}/cos_i.tif').ReadAsArray())
        cosi = np.copy(cos_i)
        cos_raa = np.cos(np.radians(180 - (observed_obs_array[:,:,3] - observed_obs_array[:,:,1])))
        cosi[cosi<=0.0] = 0.0
        cosi[cosi>=1.0] = 1.0
        sini = np.sin(np.arccos(cosi))
        cosv[cosv<=0.0] = 0.0
        cosv[cosv>=1.0] = 1.0
        sinv = np.sin(np.arccos(cosv))
        theta = np.degrees(np.arccos(-cosi*cosv + sini*sinv*cos_raa)) 
        theta[solar_zn==0] = -9999
        with rio.open(theta_path, 'w', **ras_meta) as dst:
                dst.write(theta, 1)           

    # Exception if input string does not match the req dem string
    else:
        raise Exception('Please select either "Copernicus", "SRTM","3DEP", or "User" for dem.')


    logging.info(f'DEM is saved to disk.')

    return bounds, bounds_small

