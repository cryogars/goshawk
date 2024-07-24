# Import libraries
import os
import shutil
import logging
import numpy as np
import ee
from osgeo import gdal
import rasterio as rio
from dateutil.relativedelta import relativedelta



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
    # For now this is just a static land cover dataset of 2021
    col = 'ESA/WorldCover/v200'
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




def get_o3(obs_time, bounds, service_account, ee_json):
    '''
    Uses Sentinel-5P NRTI O3: Near Real-Time Ozone dataset 
    to exctract an average O3 in units of DU over the image.

    Underlying assumption is that the average o3 over the 30 km swath does not vary too much.

    It is also averaged over a 2 week span surrounding the observation.

    Args:
        obs_time (datetime object): timestamp of the image
        path_to_img_base (str): Path to the base image directory.
        bounds (list): List of bounding coordinates [west, south, east, north].
        dem (str): DEM identifier.
        service_account (str): Google Earth Engine service account.
        ee_json (str): Path to the Earth Engine JSON key file.

    Returns:
        o3 (float): average ozone for entire image
    '''

    # This is specific to my Google server account to be able to run on linux cluster
    service_account = service_account
    credentials = ee.ServiceAccountCredentials(service_account, ee_json)
    ee.Initialize(credentials)

    # Define our area of interest (based on bounds from init script)
    area = ee.Feature(ee.Geometry.Rectangle(bounds))

    # 1 week delta
    time_before = obs_time - relativedelta(weeks=1)
    time_after = obs_time + relativedelta(weeks=1)

    # Compute mean o3
    o3_col = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_O3').select('O3_column_number_density').filterDate(time_before, time_after)
    o3_img = o3_col.mean().setDefaultProjection(o3_col.first().projection()).clip(area).unmask(-9999) #add mask
    o3_img = o3_img.reduceRegion(reducer=ee.Reducer.toList(),geometry=area.geometry()) #clip to geom
    o3 = np.array((ee.Array(o3_img.get("O3_column_number_density")).getInfo())) #convert to array
    o3 = o3[o3 >- 9999] #mask NaN
    o3 = np.mean(o3)/ (4.4615*10**-4) # convert to mol/m2 to DU
    print(o3)

    return o3



def get_canopy_cover(path_to_img_base, bounds, dem, service_account, ee_json, col='NASA/MEASURES/GFCC/TC/v3'):
    '''
    Downloads the canopy cover data from MOD44B.006 Terra Vegetation Continuous Fields Yearly Global 250m on Earth Engine 
    and performs nearest-neighbor resampling to match the extent/resolution of the DEM.

    Args:
        col (str) : Earth engine data collection (currently accepts 'NASA/MEASURES/GFCC/TC/v3' or 'MODIS/006/MOD44B')
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
    if col == 'MODIS/006/MOD44B':
        # For now this is just a static canopy cover from 2015
        cc = ee.ImageCollection(col).select('Percent_Tree_Cover').mean().unmask(-9999)
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
        cc_download = f'{path_to_img_base}_albedo/canopy/canopycover.Percent_Tree_Cover.tif'

    
    else: # col = 'NASA/MEASURES/GFCC/TC/v3'
        # For now this is just a static canopy cover from 2015
        cc = ee.ImageCollection(col).select('tree_canopy_cover').filter(ee.Filter.date('2015-01-01', '2015-12-31')).mean().unmask(-9999)
        # Max download size is 50.33 MB
        if 'EMIT' in path_to_img_base: 
            url = cc.getDownloadUrl({
            'name': 'canopycover',
            'region': area.geometry(),
            'scale': 60,
            })
        else: #PRISMA
            url = cc.getDownloadUrl({
            'name': 'canopycover',
            'region': area.geometry(),
            'scale': 30,
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

        cc_download = f'{path_to_img_base}_albedo/canopy/canopycover.tree_canopy_cover.tif'


    cc_prj = f'{path_to_img_base}_albedo/canopy/canopy_prj_missing.tif'
    cc_fill = f'{path_to_img_base}_albedo/canopy/canopy_prj.tif'

    # Reference raster
    ref_raster = rio.open(f'{path_to_img_base}_albedo/dem_{dem}/cos_v.tif')
    crs = ref_raster.crs  # ASSUMING USED UTM PROJECTION OPTION IN SISTER
    west, south, east, north = ref_raster.bounds


    if 'EMIT' in path_to_img_base:
        os.system(f'gdalwarp -r nearest -srcnodata -9999 -t_srs {crs} -te {west} {south} {east} {north} \
                    -tr 60 60 -overwrite {cc_download} {cc_prj}')
    else:
        os.system(f'gdalwarp -r nearest -srcnodata -9999 -t_srs {crs} -te {west} {south} {east} {north} \
                    -tr 30 30 -overwrite {cc_download} {cc_prj}')


    # fill no data
    os.system(f'gdal_fillnodata.py -md 1000 -q {cc_prj} {cc_fill}')

    canopy = np.array(gdal.Open(cc_prj).ReadAsArray())

    logging.info('Canopy cover dataset successfully downloaded and resampled to image.')

    return canopy
