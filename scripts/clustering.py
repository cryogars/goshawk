# Import libraries
import numpy as np
import pandas as pd
from spectral import *
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import logging


def kmeans_grouping(rad_og, cloud_mask, dem, cosi, cosv,
                    theta, svf, lc, 
                    slope_array, aspect_array,
                    sensor_wavelengths,
                    lat_mean, lon_mean, shadow_arr,
                    sza, vza, saa, vaa, 
                    g_l0, g_tup, g_s, g_edir, g_edn,
                    optimal_cosi, impurity_type, cpu,
                    cosi_decimals=4, max_clusters=10, 
                    max_iterations=100):
    '''
    Computes k-means grouping for the entire image. It leverages Dask in order to compute spatial 
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


    '''
    # Locate the SPy logger and suppress info on each iteration..
    spy_logger = logging.getLogger('spectral')
    spy_logger.setLevel(logging.CRITICAL)


    #from datetime import datetime
    #startTime = datetime.now()

    #######################################################
    # Section speed on single core is 5.8 seconds
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
        idx = np.where(cosi_array_flat==uni)
        chunks_list.append(rad_array_flat[idx,:])
        i_chunks_list.append(i_array_flat[idx])
        j_chunks_list.append(j_array_flat[idx])
    #print('FINISHED SORTING UNIQUE VALUES',datetime.now() - startTime)


    #######################################################
    #######################################################
    #######################################################
    #######################################################
    # TODO: current single core this seciton in 34 seconds,
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
        
        if chunk_clean.size != 0:
            #chunk_clean = np.expand_dims(chunk_clean, axis=0)
            (m, c) = kmeans(chunk_clean, max_clusters, max_iterations)
            for q in range(c.shape[0]): 
                locs_in_chunk = np.argwhere(m == q)
                if locs_in_chunk.size != 0:
                    spectra_dict[uni_id] = c[q]
                    for loc in locs_in_chunk:
                        idx_kmeans = int(loc[0])
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
    #print('FINISHED RUNNING K-MEANS',datetime.now() - startTime)

    #######################################################
    #######################################################
    # this section is fairly quick and scales nicely with DASK
    #######################################################
    # turn to pandas dataframe as same dtype
    cluster_matches = np.array(cluster_matches)
    cluster_matches = cluster_matches.astype(float)
    pdf = pd.DataFrame(data=cluster_matches, columns=['uni_id', 'i', 'j', 'elev','cosi', 'cosv',
                                                      'theta', 'slope','aspect','svf','lc','shadow','rmse'])

    # Averaging aspect, need to account for discontinuity
    pdf['n'] = np.cos(np.radians(pdf['aspect']))
    pdf['e'] = np.sin(np.radians(pdf['aspect']))


    # Creating dask with N cpu partiions
    ddf = dd.from_pandas(pdf, npartitions=cpu)

    # Perform groupby operations
    result = ddf.groupby('uni_id').agg({
        'i': 'mean',
        'j': 'mean',
        'elev': 'mean',
        'cosi': 'mean',
        'cosv': 'mean',
        'theta': 'mean',
        'slope': 'mean',
        'aspect': 'mean',
        'svf': 'mean',
        'lc': 'mean',
        'shadow': 'mean',
        'rmse': 'mean',
        'n': 'mean',
        'e': 'mean'
    })

    # Convert back to pandas DataFrame
    with ProgressBar():
        pdf = result.compute()
    
    # Perform rounding for LC and Shadow
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
                        optimal_cosi, impurity_type])

    #print('FINISHED DASK',datetime.now() - startTime)

    return combo_list, cluster_matches, spectra_dict, cosi_dict