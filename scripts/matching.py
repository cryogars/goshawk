# Import libraries
import numpy as np
from postprocessing import special_cases



def match_combo(i,j,rad_array, cosi, lc, cc, uniid_array, 
                combo_copy, uniid_list, sensor_wavelengths, cloud_mask):
    '''
    This function assigns the closest fit from the computed combos to each pixel in the actual image
    based on the data for that pixel. It also includes special cases to clean the data.

    Args:
        i (int): Row index of the pixel.
        j (int): Column index of the pixel.
        observed_rad_array (numpy array): Array of observed reflectance values.
        cosi (numpy array): Array of cosi values.
        lc (numpy array): Array of lc values.
        cc (numpy array): Array of cc values.
        cosi_list (numpy array): Array of cosi values from the computed combos.
        lc_list (numpy array): Array of lc values from the computed combos.
        combo_copy (numpy array): Copy of the computed combos.
        spectra_list (numpy array): List of spectra.
        sensor_wavelengths (numpy array): Array of sensor wavelengths.
        ix (numpy index) : indices of bands to remove from matching.

    Returns:
        opt_out (list): List of assigned values for the pixel.
    '''
    
    # Load data array for i,j
    rad_ij = rad_array[i, j, :] 
    cosi_ij = cosi[i, j]
    uniid_ij = uniid_array[i,j]
    lc_ij = int(lc[i, j])
    cc_ij = cc[i,j]

    # Case for no-data values
    if cosi_ij > 10000 or lc_ij == 0: # no data for lc is 0
        opt_out = [i, j, -9999, -9999, -9999, -9999, -9999,-9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999]
        return opt_out
    
    if cloud_mask[i,j] == 1:
        opt_out = [i, j, -9999, -9999, -9999, -9999, -9999,-9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999]
        return opt_out     

    # Find index where uni_ID is in relation to combos
    match_idx = uniid_list.index(uniid_ij)
    
    # Match found
    combo_ij = combo_copy[match_idx]

    #Compute exact RMSE
    # Remove bands not in inversion window for snow and atmos.
    ix_1 = np.argwhere((sensor_wavelengths >= 2450) & (sensor_wavelengths <= 2700))
    ix_2 = np.argwhere((sensor_wavelengths >= 1780) & (sensor_wavelengths <= 1950))
    ix_3 = np.argwhere((sensor_wavelengths >= 1300) & (sensor_wavelengths <= 1450))
    ix_4 = np.argwhere((sensor_wavelengths >= 300) & (sensor_wavelengths <= 500))
    ix = np.concatenate((ix_1, ix_2, ix_3, ix_4))
    rad_ij = np.delete(rad_ij, ix)
    l_toa = np.delete(combo_ij[12], ix)
    rmse = np.sqrt(((rad_ij - l_toa)**2).mean())

    # Special cases
    idx_1600nm = (np.abs(sensor_wavelengths - 1600)).argmin()
    idx_550nm = (np.abs(sensor_wavelengths - 550)).argmin()
    idx_850nm = (np.abs(sensor_wavelengths - 850)).argmin()
    idx_660nm = (np.abs(sensor_wavelengths - 660)).argmin()
    rho_surface = combo_ij[16]
    ir = rho_surface[idx_850nm]
    red = rho_surface[idx_660nm]
    green = rho_surface[idx_550nm]
    swir = rho_surface[idx_1600nm]
    ndvi = (ir-red)/(ir+red)
    ndsi = (green-swir)/(green+swir)
    opt_out = special_cases(i,j,lc_ij,cc_ij,combo_ij,ndsi,ndvi)
    
    # if no special cases
    # For reference these indices i am pulling from combo_ij can be identified via optimization.py
    # [i,j,f_snow, f_shade, f_em1, f_em2, ssa, lap,lwc, h20,aod, broadband, rmse, l_toa, rho_s, r, s_total]
    # EXACT RMSE is used, minus the bands not used , see IX above.
    if opt_out == 1:         
        opt_out = [i,j, combo_ij[1], combo_ij[2], 
                   combo_ij[3], combo_ij[4],
                   combo_ij[5], combo_ij[6], 
                   combo_ij[7], combo_ij[8], 
                   combo_ij[9], combo_ij[10], 
                   rmse, combo_ij[12], 
                   combo_ij[13], combo_ij[14], 
                   combo_ij[15], combo_ij[17]]
        

   # FOR REFERENCE FROM OPTIMIZATION.PY
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


    return opt_out