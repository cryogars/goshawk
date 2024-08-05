# Import libraries
import os
import numpy as np
import pandas as pd
import rasterio as rio
from osgeo import gdal




def cloud_tif(cloud_array, path_to_img_base, dem):
    '''
    Outputs cloud tiff
    

    '''

    # Check whether directory exists, if not, make one
    cloud_dir = f'{path_to_img_base}_albedo/clouds'
    if not os.path.exists(cloud_dir):
        os.makedirs(cloud_dir)

    # load in initial dataset for reference
    with rio.open(f'{path_to_img_base}_albedo/dem_{dem}/cos_i.tif') as src:
        ras_data = src.read()
        ras_meta = src.profile


    # Save all arrays to tifs
    gdal_np_to_raster(cloud_array, 'clouds', ras_data, ras_meta, src, cloud_dir, path_to_img_base, dem)

    return 



def kmeans_tifs(cluster_matches, selected_cosi,path_to_img_base, dem, cloud_mask, cosi_dict):
    '''
    Outputs kmeans tif and RMSE from matching to centroid
     
    columns=['uni_id', 'i', 'j', 'elev','cosi', 'cosv','theta', 'slope','svf','lc','shadow','rmse'])

    Returns:
        None

    '''

    # Check whether snow directory exists, if not, make one
    kmeans_dir = f'{path_to_img_base}_albedo/clustering'
    if not os.path.exists(kmeans_dir):
        os.makedirs(kmeans_dir)

    # Expand output_list (created in initialize)
    kmeans_arr = np.ones_like(selected_cosi)
    rmse_arr = np.ones_like(selected_cosi)
    d_cosi_arr = np.ones_like(selected_cosi)

    # load in initial dataset for reference
    with rio.open(f'{path_to_img_base}_albedo/dem_{dem}/cos_i.tif') as src:
        ras_data = src.read()
        ras_meta = src.profile

    # Get each of the complete arrays from match-cluster list
    # columns=['uni_id', 'i', 'j', 'elev','cosi', 'cosv','theta', 'slope','aspect','svf','lc','shadow','rmse'])
    for p in cluster_matches:
        
        # Save to the arrays
        cosij = selected_cosi[int(p[1]), int(p[2])]

        if cosij > 100 or cloud_mask[int(p[1]), int(p[2])] == 1:
            d_cosi_arr[int(p[1]), int(p[2])] = -9999
            kmeans_arr[int(p[1]), int(p[2])] =-9999
            rmse_arr[int(p[1]), int(p[2])] = -9999
        else:
            # get kmeans data
            cosi_kmeans = cosi_dict[p[0]]
            d_cosi_arr[int(p[1]), int(p[2])]  = cosi_kmeans - cosij
            kmeans_arr[int(p[1]), int(p[2])] = p[0]
            rmse_arr[int(p[1]), int(p[2])] = p[12]

    # Hard set NaN values
    d_cosi_arr[(selected_cosi > 10) | (cloud_mask == 1)] = -9999
    kmeans_arr[(selected_cosi > 10) | (cloud_mask == 1)] = -9999
    rmse_arr[(selected_cosi > 10) | (cloud_mask == 1)] = -9999

    # Save all arrays to tifs
    gdal_np_to_raster(kmeans_arr, 'kmeans', ras_data, ras_meta, src, kmeans_dir, path_to_img_base, dem )
    gdal_np_to_raster(rmse_arr, 'rmse', ras_data, ras_meta, src, kmeans_dir, path_to_img_base, dem)
    gdal_np_to_raster(d_cosi_arr, 'dcosi', ras_data, ras_meta, src, kmeans_dir, path_to_img_base, dem)

    return kmeans_arr


def snow_tifs(results, selected_cosi, path_to_img_base, dem, cc, optimal_cosi):
    '''
    Outputs all tifs to the snow directory (wherever input images are saved).
    

    # [i,j,f_snow, f_shade, f_em1, f_em2, ssa, lap,lwc, h20,aod, broadband, rmse, l_toa, rho_s, r, s_total]
    Args:
        results (list): List of optimization results.
        SELECTED COSI
        path_to_img_base (str): Base path to the input images.
        dem (str): DEM identifier.
        cc (numpy.ndarray): Canopy cover array.
    
    Returns:
        None
    '''
    # Check whether snow directory exists, if not, make one
    snow_dir = f'{path_to_img_base}_albedo/snow'
    if not os.path.exists(snow_dir):
        os.makedirs(snow_dir)

    # Expand output_list (created in initialize)
    f_snow_arr = np.ones_like(selected_cosi)
    grain_arr = np.ones_like(selected_cosi)
    broad_arr = np.ones_like(selected_cosi)
    rmse_arr = np.ones_like(selected_cosi)
    f1_arr = np.ones_like(selected_cosi)
    f2_arr = np.ones_like(selected_cosi)
    f3_arr = np.ones_like(selected_cosi)
    f4_arr = np.ones_like(selected_cosi)
    lap_arr = np.ones_like(selected_cosi)
    lwc_arr = np.ones_like(selected_cosi)
    aod_arr = np.ones_like(selected_cosi)
    h20_arr = np.ones_like(selected_cosi)

    if optimal_cosi == 'yes':
        cosi_arr = np.ones_like(selected_cosi)

    # load in initial dataset for reference
    with rio.open(f'{path_to_img_base}_albedo/dem_{dem}/cos_i.tif') as src:
        ras_data = src.read()
        ras_meta = src.profile

    # Get each of the complete arrays from optimization list
    for val in results:
        row = val[0]
        col = val[1]
        # Save to the arrays
        if val[2] < 0.0 or val[3] < 0.0: #NaN value
            f_snow_arr[row, col] = val[2]  # snow NaN
        else:
            # T.H. Painter et al. / Remote Sensing of Environment 85 (2003) 64–77
            # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9290428 
            # BAIR et al.: SNOW PROPERTY INVERSION FROM REMOTE SENSING (SPIReS)
            f_snow_arr[row, col] = min(1, val[2] / (1 - val[3] - cc[row,col]))
            if f_snow_arr[row, col] < 0:
                f_snow_arr[row, col] = -9999 # negative returned... NaN

        grain_arr[row, col] = val[6]
        broad_arr[row, col] = val[11]
        rmse_arr[row,col] = val[12]
        f1_arr[row,col] = val[2]
        f2_arr[row,col] = val[3]
        f3_arr[row,col] = val[4]
        f4_arr[row,col] = val[5]
        lap_arr[row,col] = val[7]
        lwc_arr[row,col] = val[8]
        h20_arr[row,col] = val[9]      
        aod_arr[row,col] = val[10]
        if optimal_cosi == 'yes':
            cosi_arr[row,col] = val[17]     

    # Save all arrays to tifs
    gdal_np_to_raster(f_snow_arr, 'FSCA', ras_data, ras_meta, src, snow_dir, path_to_img_base, dem)
    gdal_np_to_raster(grain_arr, 'SSA', ras_data, ras_meta,src, snow_dir, path_to_img_base, dem)
    gdal_np_to_raster(broad_arr, 'BA', ras_data, ras_meta, src, snow_dir, path_to_img_base, dem)
    gdal_np_to_raster(rmse_arr, 'RMSE', ras_data, ras_meta,src, snow_dir, path_to_img_base, dem)
    gdal_np_to_raster(f1_arr, 'F1', ras_data, ras_meta,src, snow_dir, path_to_img_base, dem)
    gdal_np_to_raster(f2_arr, 'F2', ras_data, ras_meta,src, snow_dir, path_to_img_base, dem)
    gdal_np_to_raster(f3_arr, 'F3', ras_data, ras_meta,src, snow_dir, path_to_img_base, dem)
    gdal_np_to_raster(f4_arr, 'F4', ras_data, ras_meta,src, snow_dir, path_to_img_base, dem)
    gdal_np_to_raster(lap_arr, 'LAP', ras_data, ras_meta,src, snow_dir, path_to_img_base, dem)
    gdal_np_to_raster(lwc_arr, 'LWC', ras_data, ras_meta,src, snow_dir, path_to_img_base, dem)
    gdal_np_to_raster(h20_arr, 'H2O', ras_data, ras_meta,src, snow_dir, path_to_img_base, dem)
    gdal_np_to_raster(aod_arr, 'AOD550', ras_data, ras_meta,src, snow_dir, path_to_img_base, dem)
    if optimal_cosi == 'yes':
        # make any necessary changes to raster properties, e.g.:
        ras_meta['nodata'] = -9999
        # Write tif file
        tif_file = f'{snow_dir}/GOSHAWK-MU.tif'
        with rio.open(tif_file, 'w', **ras_meta) as dst:
            dst.write(cosi_arr, 1)

    return


def gdal_np_to_raster(array, tif_name, ras_data, ras_meta, src, snow_dir, path_to_img_base, dem, interp_res=100):
    '''
    Converts a NumPy array to a raster file using GDAL library.
    Optionally performs interpolation to 100m spatial resolution.

    Args:
        array (numpy.ndarray): Input array to be written as raster.
        tif_name (str): Name of the output TIFF file.
        ras_data: Raster data.
        ras_meta: Metadata for the raster.
        src: Source data.
        snow_dir (str): Directory path where the TIFF file will be saved.
        path_to_img_base (str): Base path for the input images.
        dem (str): DEM identifier.
        interp_res: default is 100m

    Returns:
        None
    '''

    # make any necessary changes to raster properties, e.g.:
    ras_meta['nodata'] = -9999

    # Write tif file
    tif_file = f'{snow_dir}/{tif_name}.tif'
    ref_file = f'{path_to_img_base}_albedo/dem_{dem}/cos_i.tif'
    with rio.open(tif_file, 'w', **ras_meta) as dst:
        dst.write(array, 1)

    # Reproject the tif to 100m spatial res
    crs = src.crs
    west, south, east, north = src.bounds
    tif_interp_file = f'{snow_dir}/{tif_name}_{interp_res}m.tif'
    ref_interp_file = f'{path_to_img_base}_albedo/dem_{dem}/cos_i_{interp_res}m.tif'
    os.system(f'gdalwarp -r bilinear -t_srs {crs} -te {west} {south} {east} {north} \
                    -tr {interp_res} {interp_res} -overwrite {tif_file} {tif_interp_file} -q')
    os.system(f'gdalwarp -r bilinear -t_srs {crs} -te {west} {south} {east} {north} \
                    -tr {interp_res} {interp_res} -overwrite {ref_file} {ref_interp_file} -q')  
    # gdal fill no data and interpolation
    tif_interp_fill_file = f'{snow_dir}/{tif_name}_{interp_res}m_fill.tif'
    os.system(f'gdal_fillnodata.py -md 1000 -q {tif_interp_file} {tif_interp_fill_file}')
    # Only keep fill values where there is elevation data (exclude the edges)
    tif_interp_clean_file = f'{snow_dir}/{tif_name}_{interp_res}_fill_clean.tif'
    os.system(f'gdal_calc.py -A {tif_interp_fill_file} -B {ref_interp_file} \
                --outfile={tif_interp_clean_file} --type Float64 --overwrite \
                --calc="where((B>-2500),A,B)" --quiet')

    return



def snow_csvs(results, cc, path_to_img_base, sensor_wavelengths):
    '''
    Saves all CSV files to the snow directory.

    # [i,j,f_snow, f_shade, f_em1, f_em2, ssa, lap,lwc, h20,aod, broadband, rmse, l_toa, rho_s, r, s_total]

    Args:
        path_to_img_base (str): Base path for the input images.
        sensor_wavelengths (list): List of sensor wavelengths.

    Returns:
        None
    '''
    # Check whether snow directory exists, if not, make one
    snow_dir = f'{path_to_img_base}_albedo/snow'
    if not os.path.exists(snow_dir):
        os.makedirs(snow_dir)
    # create variables for results needed 
    i = results[0]
    j = results[1]

    # Create a directory for this i,j pair and save
    ij_dir = f'{snow_dir}/pixel_{i}_{j}'
    if not os.path.exists(ij_dir):
        os.makedirs(ij_dir)
    # Save all of the csvs
    df_ltoa = pd.DataFrame({'Wavelength [nm]': sensor_wavelengths, 'L_TOA_microWcm-2sr-1nm-1': results[13]})
    df_ltoa.to_csv(f'{ij_dir}/l_toa.csv', index=False)

    df_snowr = pd.DataFrame({'Wavelength [nm]': sensor_wavelengths, 'Snow_Reflectance': results[14]})
    df_snowr.to_csv(f'{ij_dir}/snow_reflectance.csv', index=False)

    df_snowa = pd.DataFrame({'Wavelength [nm]': sensor_wavelengths, 'Snow_Albedo': results[15]})
    df_snowa.to_csv(f'{ij_dir}/snow_albedo.csv', index=False)

    df_irrad = pd.DataFrame({'Wavelength [nm]': sensor_wavelengths, 'Total_Irradiance_microWcm-2sr-1nm-1': results[16]})
    df_irrad.to_csv(f'{ij_dir}/total_irrad.csv', index=False)

    # Finally, save all of the other findings to disk
    fsca = min(1, results[2] / (1 - results[3] - cc[i,j]))
    if fsca < 0:
        fsca = -9999 # negative returned... NaN
    # [i,j,f_snow, f_shade, f_em1, f_em2, ssa, lap,lwc, h20,aod, broadband, rmse, l_toa, rho_s, r, s_total]
    with open(f'{ij_dir}/optimization_outputs.txt', 'w') as f:
        f.write(f'fSCA: {fsca}\n')  # 
        f.write(f'row,col: {i,j}\n')  # 
        f.write(f'f_snow: {results[2]}\n')  # 
        f.write(f'f_shade: {results[3]}\n')  # 
        f.write(f'f_endmember1: {results[4]}\n')  # 
        f.write(f'f_endmember2: {results[5]}\n')  # 
        f.write(f'SSA: {results[6]}\n')  # 
        f.write(f'lap: {results[7]}\n')  # 
        f.write(f'LWC: {results[8]}\n')  #   
        f.write(f'h20MM: {results[9]}\n')  # 
        f.write(f'AOD550: {results[10]}\n')  # 
        f.write(f'Snow Broadband Albedo: {results[11]}\n')  # 
        f.write(f'RMSE: {results[12]}\n')  # 
    
    
    return




def special_cases(i,j,lc_ij,cc_ij,combo_ij,ndsi,ndvi):
    '''
    Identify special cases where the model breaks down and requires intervention.

   #    # FOR REFERENCE FROM OPTIMIZATION.PY
   #     # Save all the outputs to a list
   # combo_opt = [i, #0
   #              f_snow, #1 
   #              f_shade, #2 
   #              f_em1, #3
   #              f_em2, #4
   #              ssa_opt, #5
   #              lap_opt, #6
   #              lwc_opt, #7   
   #              h20_opt, #8
   #              aod_opt, #9
   #              broadband, #10
   #              rmse, #11
   #              l_toa, #12 estimated TOA Radiance
   #              rho_s, #13 snow reflectance
   #              alb_snow, #14 Plane albedo
   #              s_total,#15 total irrad
   #              rho_surface,#16 surface reflectance
   #              cosi] #17 cosi 


    TODO

    '''

    # for pixles with canopy cover fractions above 0.5
    # https://tc.copernicus.org/articles/17/567/2023/ 
    # Landsat, MODIS, and VIIRS snow cover mapping algorithm performance as validated by airborne lidar datasets
    if cc_ij >= 0.5 and lc_ij == 10:
        opt_out = [i, j, -9999, -9999,-9999,-9999, -9999,-9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999]
        return opt_out

    # For no snow, ndsi less than 0, assign 0.25 as BBA
    if ndsi <= 0.0:
        opt_out = [i, j, 0, 0,0, 0,0,0, 0, 0, 0, 0.25, combo_ij[11], 0, 0, 0, 0, -9999]
        return opt_out

    # Keeping only very confident snow pixels , where f_snow >75%
    if combo_ij[1] <= 0.75:
        opt_out = [i, j, combo_ij[1], combo_ij[2], combo_ij[3],combo_ij[4], -9999,-9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999]
        return opt_out

    # Errors for low snow and shade where optimization returns max or min SSA.
    if combo_ij[5] >= 155.5 or combo_ij[5] <= 2.5:
        opt_out = [i, j, combo_ij[1], combo_ij[2], combo_ij[3],combo_ij[4], -9999,-9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999]
        return opt_out

    # if none then pixel is fine
    opt_out = 1

    return opt_out


