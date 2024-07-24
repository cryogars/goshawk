# Import libraries
import numpy as np
import pandas as pd
from spectral import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean as _mean


def kmeans_grouping(rad_og, cloud_mask, dem, cosi, cosv,
                    theta, svf, lc, 
                    slope_array, aspect_array,
                    sensor_wavelengths,
                    lat_mean, lon_mean, shadow_arr,
                    sza, vza, saa, vaa, 
                    g_l0, g_tup, g_s, g_edir, g_edn,
                    optimal_cosi, cpu,
                    cosi_decimals=4, max_clusters=10, 
                    max_iterations=100):
    '''
    Computes k-means grouping for the entire image. It leverages PySpark in order to compute spatial 
    statistics for each class (#classes > 50,000). These averages are then saved with the clustered spectra,
    to be outputted and passed along for optimization.  
    
    The key thing here is that I am assuming that areas with high/low cosine of solar illumination angle 
    will have higher spatial variability (e.g., mountains). Therefore,the algorithm will have the opportunity 
    to assign 10 clusters for say cosi=0.8345 (4 decimals). While areas that are flatter will still be given the 
    opportunity to solve for 10 clusters, but most likely will result in less because of the lower variation. In
    this way, the clustering is adaptive, in that it samples more where it needs to account for terrain differences.

    One could tweak the cosi_decimals, max_clusters, and/or max_iterations to achieve different levels of clustering
    resolution. However, these value were tested to achieve desired accuracy 
    (less than 0.01 RMSE in spectral and cosi space) AND to ensure number of clusters remains low enough 
    to not create too many inversions later on in optimization.py. 



    Args:
        rad_array (ndarray): topographically corrected raster of observed radiance from image spec
        cloud_mask (ndarray): Output raster of Zhai function (1=cloud, 0=none)
        cosi (ndarray): raster of selected cosine of solar illumination angle
        svf (ndarray): raster of selected skyview fraction
        lc (ndarray): raster of WorldCover Land cover dataset
        density (ndarray): snow density resampled to image from ERA-Land
        sza_array (ndarry): solar zenith angle array
        sensor_wavelengths (ndarray): 1d array of the sensor wavelengths in nanometers
        path_to_img_base (str): the path to the image spec data
        wet (Bool): bool to determine of snow is wet or dry conditions in optimization
        lat_mean (float): mean latitude across the image
        lon_mean (float): mean longitude across the image
        shadow_arr (ndarray): computed using ray tracing to determine areas that are shaded by local terrain
        cosi_decimals (float): the number of decimals to round the cosine of solar illumination - used for grouping in kmeans (default=4)
        max_clusters (float): the maximum number kmeans clusters that can be solved for during iterations in a given group (default=10)
        max_iterations (float): the maximum number of iterations allowed for searching given chunk for clusters (default=100)

    Returns:
        TODO

    '''



    #######################################################
    # TODO: this prepping section is a little slow and is also single core
    # takes about 59 seconds. Can be done in a way to use many cores.
    #######################################################
    # Clean data
    rad_array = np.copy(rad_og)
    rad_array[rad_array <= -9999] = 0.0
    rad_array[cloud_mask == 1] = 0.0

    # Create an I,J array for indexing later in the loop back to the orignal image
    i_array = np.ones_like(rad_array[:,:,10])
    j_array = np.ones_like(rad_array[:,:,10])

    for i in range(0,i_array.shape[0]):
        for j in range(0,i_array.shape[1]):
            i_array[i,j] = i
            j_array[i,j] = j

    # flatten the arrays to 1d (2d for spectra)
    rad_array_flat = np.vstack(rad_array)
    i_array_flat = i_array.flatten()
    j_array_flat = j_array.flatten()
    cosi_array_flat = np.round(cosi.flatten(), cosi_decimals)
    unique_cosi = np.unique(cosi_array_flat)

    # Now going to break off the data into chunks based on 0.0001
    # Save each of the different datasets using the same idex
    chunks_list = []
    i_chunks_list = []
    j_chunks_list = []
    for uni in unique_cosi:
        chunks_list.append(rad_array_flat[cosi_array_flat==uni,:])
        i_chunks_list.append(i_array_flat[cosi_array_flat==uni])
        j_chunks_list.append(j_array_flat[cosi_array_flat==uni])
    
    # loop through all chunks and apply k-means clustering
    # in the end, they will be compiled into the whole image
    uni_id = 0
    cluster_matches = []
    spectra_dict = {}
    for chunk_num in range(0,len(chunks_list)):
        chunk = chunks_list[chunk_num]
        i_chunk = i_chunks_list[chunk_num]
        j_chunk = j_chunks_list[chunk_num]

        # Remove all zero (NAN) data from the chunk
        chunk_clean = []
        i_chunk_clean = []
        j_chunk_clean = []
        for t in range(0,len(chunk)):
            if np.sum(chunk[t]) != 0:
                chunk_clean.append(chunk[t])
                i_chunk_clean.append(i_chunk[t])
                j_chunk_clean.append(j_chunk[t])
        chunk_clean = np.array(chunk_clean)
        i_chunk_clean = np.array(i_chunk_clean)
        j_chunk_clean = np.array(j_chunk_clean)
    #######################################################
    #######################################################
    #######################################################
    #######################################################
        # TODO: current single core this seciton in 37 seconds,
        # Can be sped up to only take a few seconds if i get it going in parallel.
        if chunk_clean.size != 0:
            chunk_clean = np.expand_dims(chunk_clean, axis=0)
            (m, c) = kmeans(chunk_clean, max_clusters, max_iterations)
            for q in range(c.shape[0]): 
                locs_in_chunk = np.argwhere(m == q)
                if locs_in_chunk.size != 0:
                    spectra_dict[uni_id] = c[q]
                    for loc in locs_in_chunk:
                        idx_kmeans = int(loc[1])
                        i_exact = int(i_chunk_clean[idx_kmeans])
                        j_exact = int(j_chunk_clean[idx_kmeans])
                        dem_ij = dem[i_exact, j_exact] / 1000 #km   
                        cosi_ij = cosi[i_exact, j_exact]
                        cosv_ij = cosv[i_exact,j_exact]
                        theta_ij = theta[i_exact,j_exact]
                        svf_ij = svf[i_exact, j_exact]
                        slope_ij = slope_array[i_exact, j_exact]
                        aspect_ij = aspect_array[i_exact, j_exact]
                        lc_ij = lc[i_exact, j_exact]
                        shadow_ij = shadow_arr[i_exact, j_exact]
                        rmse_ij = np.sqrt(((rad_array[i_exact,j_exact] - c[q])**2).mean())
                        cluster_matches.append([uni_id,i_exact,j_exact, dem_ij, cosi_ij, cosv_ij,
                                                theta_ij, slope_ij, aspect_ij, svf_ij,
                                                lc_ij,shadow_ij, rmse_ij])
                    uni_id+=1
    #######################################################
    #######################################################
    # this section is fairly quick and scales nicely with Pyspark
    # currently takes 45 seconds on 10 cores.
    #######################################################
    # turn to pandas dataframe as same dtype
    cluster_matches = np.array(cluster_matches)
    cluster_matches = cluster_matches.astype(float)
    pdf = pd.DataFrame(data=cluster_matches, columns=['uni_id', 'i', 'j', 'elev','cosi', 'cosv',
                                                      'theta', 'slope','aspect','svf','lc','shadow','rmse'])

    # Averaging aspect, need to account for discontinuity
    pdf['n'] = np.cos(np.radians(pdf['aspect']))
    pdf['e'] = np.sin(np.radians(pdf['aspect']))

    # Create SparkSession with cores=n_cpu
    spark = SparkSession.builder.master(f'local[{cpu}]').config("spark.driver.memory", "15g").appName('goshawk-app').getOrCreate()

    # Save as a spark dataframe
    sdf = spark.createDataFrame(pdf)

    # perform groupby operations and save back to pandas dataframe
    pdf = sdf.groupBy('uni_id') \
             .agg(_mean('i').alias('i'), \
                  _mean('j').alias('j'), \
                  _mean('elev').alias('elev'), \
                  _mean('cosi').alias('cosi'), \
                  _mean('cosv').alias('cosv'), \
                  _mean('theta').alias('theta'), \
                  _mean('slope').alias('slope'), \
                  _mean('aspect').alias('aspect'), \
                  _mean('svf').alias('svf'), \
                  _mean('lc').alias('lc'), \
                  _mean('shadow').alias('shadow'), \
                  _mean('rmse').alias('rmse'), \
                  _mean('n').alias('n'), \
                  _mean('e').alias('e')).toPandas()
    
    pdf = pdf.round({'lc':-1}) #round to nearest 10 for LC
    pdf = pdf.round({'shadow': 0}) #round to nearest 1 for Shadow (0 or 1)

    # Convert back to aspect (0-360)
    pdf['aspect'] = np.degrees(np.arctan2(pdf['e'], pdf['n']))
    pdf['aspect'] = pdf['aspect'].apply(lambda x: x+360 if x < 0.0 else x)

    # Vectorize for better looping below
    np_mean = pdf.to_numpy()

    # Assign uni_id index, rad_array, cosi, LC for each combo,
    # As well as other things needed for optimizations
    combo_list = []
    cosi_dict = {}
    for row in range(0, np_mean.shape[0]):
        row_array = np_mean[row]
        # Spectra that goes with uni_id
        cosi_dict[row_array[0]] = row_array[4]
        combo_list.append([row_array[0], #i
                        spectra_dict[row_array[0]], #r
                        row_array[3], #elev-km
                        row_array[4], #cosi
                        row_array[5], #cosv
                        row_array[6], #theta
                        row_array[7], #slope 
                        row_array[8], #aspect
                        row_array[9], #svf 
                        int(row_array[10]), #LC
                        int(row_array[11]), #shadow
                        sensor_wavelengths,
                        lat_mean, 
                        lon_mean,
                        sza, vza,
                        saa, vaa,
                        g_l0, g_tup, 
                        g_s, g_edir, g_edn,
                        optimal_cosi])

    return combo_list, cluster_matches, spectra_dict, cosi_dict