# Import libraries
import os
import shutil
import logging
import numpy as np
import ee
from osgeo import gdal
import rasterio as rio




def get_landcover(path_to_img_base, bounds, dem, service_account, ee_json):
    '''
    Downloads the ESA WorldCover v100 product from Earth Engine and performs nearest-neighbor resampling 
    to match the extent/resolution of the DEM.

    Args:
        path_to_img_base (str): Path to the base image directory.
        bounds (list): List of bounding coordinates [west, south, east, north].
        dem (str): DEM identifier.
        service_account (str): Google Earth Engine service account.
        ee_json (str): Path to the Earth Engine JSON key file.

    Returns:
        landcover (numpy.ndarray): Resampled landcover data as a NumPy array.
    '''

    # If file exists don't redownload it
    if os.path.exists(f'{path_to_img_base}_albedo/landcover/landcover_prj.tif'):
        return

    # This is specific to my Google server account to be able to run on linux cluster
    service_account = service_account
    credentials = ee.ServiceAccountCredentials(service_account, ee_json)
    ee.Initialize(credentials)

    # Define our area of interest (based on bounds from init script)
    area = ee.Feature(ee.Geometry.Rectangle(bounds))

    # Load image based on bounds
    # For now this is just a static land cover dataset of 2020
    col = 'ESA/WorldCover/v100'
    lc = ee.ImageCollection(col).first().clip(area)

    # Max download size is 50.33 MB
    if 'EMIT' in path_to_img_base: 
        url = lc.getDownloadUrl({
        'name': 'landcover',
        'region': area.geometry(),
        'scale': 60,
        })
    else:
        url = lc.getDownloadUrl({
        'name': 'landcover',
        'region': area.geometry(),
        'scale': 20,
        })

    # Download it locally
    os.system(f'wget --directory-prefix={path_to_img_base}_albedo/landcover {url}')

    # get the zip file
    zip_filename = url.rsplit('/', 1)[1]

    # Unzip landcover downloaded from earth engine
    shutil.unpack_archive(f'{path_to_img_base}_albedo/landcover/{zip_filename}',
                          f'{path_to_img_base}_albedo/landcover',
                          format='zip')
    # Remove the zip file
    if os.path.exists(f'{path_to_img_base}_albedo/landcover/{zip_filename}'):
        os.remove(f'{path_to_img_base}_albedo/landcover/{zip_filename}')

    # resample it using gdal
    lc_prj = f'{path_to_img_base}_albedo/landcover/landcover_prj.tif'
    lc_download = f'{path_to_img_base}_albedo/landcover/landcover.Map.tif'
    ref_raster = rio.open(f'{path_to_img_base}_albedo/dem_{dem}/cos_i.tif')
    crs = ref_raster.crs  # ASSUMING USED UTM PROJECTION OPTION IN SISTER
    west, south, east, north = ref_raster.bounds
    
    if 'EMIT' in path_to_img_base:
        os.system(f'gdalwarp -r nearest -t_srs {crs} -te {west} {south} {east} {north} \
                    -tr 60 60 -overwrite {lc_download} {lc_prj}')
    else:
        os.system(f'gdalwarp -r nearest -t_srs {crs} -te {west} {south} {east} {north} \
                    -tr 30 30 -overwrite {lc_download} {lc_prj}')

    landcover = np.array(gdal.Open(lc_prj).ReadAsArray())

    logging.info('Landcover dataset successfully downloaded and resampled to image.')

    return landcover




def get_canopy_cover(path_to_img_base, bounds, dem, service_account, ee_json):
    '''
    Downloads the canopy cover data from MOD44B.006 Terra Vegetation Continuous Fields Yearly Global 250m on Earth Engine 
    and performs nearest-neighbor resampling to match the extent/resolution of the DEM.

    Args:
        path_to_img_base (str): Path to the base image directory.
        bounds (list): List of bounding coordinates [west, south, east, north].
        dem (str): DEM identifier.
        service_account (str): Google Earth Engine service account.
        ee_json (str): Path to the Earth Engine JSON key file.

    Returns:
        canopy (numpy.ndarray): Resampled canopy cover data as a NumPy array.
    '''

    # If file exists don't redownload it
    if os.path.exists(f'{path_to_img_base}_albedo/canopy/canopy_prj.tif'):
        return

    # This is specific to my Google server account to be able to run on linux cluster
    service_account = service_account
    credentials = ee.ServiceAccountCredentials(service_account, ee_json)
    ee.Initialize(credentials)

    # Define our area of interest (based on bounds from init script)
    area = ee.Feature(ee.Geometry.Rectangle(bounds))

    # Load image based on bounds
    # For now this is just a static canopy cover from 2015
    col = 'MODIS/006/MOD44B'
    cc = ee.ImageCollection(col).select('Percent_Tree_Cover').mean()

    # Max download size is 50.33 MB
    if 'EMIT' in path_to_img_base: 
        url = cc.getDownloadUrl({
        'name': 'canopycover',
        'region': area.geometry(),
        'scale': 200,
        })
    else: #PRISMA
        url = cc.getDownloadUrl({
        'name': 'canopycover',
        'region': area.geometry(),
        'scale': 200,
        })

    # Download it locally
    os.system(f'wget --directory-prefix={path_to_img_base}_albedo/canopy {url}')

    # get the zip file
    zip_filename = url.rsplit('/', 1)[1]

    # Unzip landcover downloaded from earth engine
    shutil.unpack_archive(f'{path_to_img_base}_albedo/canopy/{zip_filename}',
                          f'{path_to_img_base}_albedo/canopy',
                          format='zip')
    # Remove the zip file
    if os.path.exists(f'{path_to_img_base}_albedo/canopy/{zip_filename}'):
        os.remove(f'{path_to_img_base}_albedo/canopy/{zip_filename}')

    # resample it using gdal
    cc_prj = f'{path_to_img_base}_albedo/canopy/canopy_prj.tif'
    cc_download = f'{path_to_img_base}_albedo/canopy/canopycover.Percent_Tree_Cover.tif'
    ref_raster = rio.open(f'{path_to_img_base}_albedo/dem_{dem}/cos_i.tif')
    crs = ref_raster.crs  # ASSUMING USED UTM PROJECTION OPTION IN SISTER
    west, south, east, north = ref_raster.bounds
    
    if 'EMIT' in path_to_img_base:
        os.system(f'gdalwarp -r nearest -t_srs {crs} -te {west} {south} {east} {north} \
                    -tr 60 60 -overwrite {cc_download} {cc_prj}')
    else:
        os.system(f'gdalwarp -r nearest -t_srs {crs} -te {west} {south} {east} {north} \
                    -tr 30 30 -overwrite {cc_download} {cc_prj}')

    canopy = np.array(gdal.Open(cc_prj).ReadAsArray())

    logging.info('Canopy cover dataset successfully downloaded and resampled to image.')

    return canopy
