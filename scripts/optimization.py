# Import libraries
import math
import numpy as np
from scipy import optimize
from initialize import initial_endmembers
from snow import art
from libradtran import lrt_reader


# Define mpi wrapper function
def compute_pixels_mpi(args):
    return invert_snow_and_atmos_props(*args)


def rmse_calc(residual):
    '''
    Calculate the root mean squared error (RMSE) of the residual.

    Args:
        residual (numpy array): Array of residual values.

    Returns:
        rmse (float): Root mean squared error.
    '''
    # Remove NaN values from the residual
    residual = residual[~np.isnan(residual)]

    # Calculate RMSE
    rmse = np.sqrt(np.sum(np.square(residual)) / residual.shape)
    
    return rmse


def new_aspect(east, north):
    '''
    TODO
    '''
    aspect = np.degrees(math.atan2(east, north))
    if (aspect < 0.0):
        aspect += 360.0

    return aspect 


def new_terrain(slope, aspect, sza, vza, saa, vaa):

    '''
    TODO
    '''

    cosi = np.sin(sza*(np.pi/180))*np.sin(slope*(np.pi/180))*np.cos(saa*(np.pi/180)-aspect*(np.pi/180))+np.cos(sza*(np.pi/180))*np.cos(slope*(np.pi/180))
    cosv = np.sin(vza*(np.pi/180))*np.sin(slope*(np.pi/180))*np.cos(vaa*(np.pi/180)-aspect*(np.pi/180))+np.cos(vza*(np.pi/180))*np.cos(slope*(np.pi/180))
    if cosi <= 1e-16:
        cosi = 1e-16
    if cosv <= 1e-16:
        cosv = 1e-16
    if cosi >= 1:
        cosi = 1
    if cosv >= 1:
        cosv = 1
    sini = np.sin(np.arccos(cosi))
    sinv = np.sin(np.arccos(cosv))
    if sini <= 1e-16:
        sini = 1e-16
    if sinv <= 1e-16:
        sinv = 1e-16
    if sini >= 1:
        sini = 1
    if sinv >= 1:
        sinv = 1
    cos_raa = np.cos(np.radians(180 - (vaa - saa)))
    theta = np.degrees(np.arccos(-cosi*cosv + sini*sinv*cos_raa)) 

    return cosi, cosv, theta



def fun_min(x0, cosi, cosv, theta, sensor_wavelengths, endmembers_opt,
            sza, vza, saa, vaa, 
            alt, shadow, svf, slope, aspect, 
            observed_radiance_transpose,
            g_l0, g_tup, g_s, g_edir, g_edn, 
            ix, optimal_cosi):
    '''
    Define the objective function.
    
    Args:


    Returns:
        rmse (float): Root mean squared error.
    '''
    # Get x0 ready
    fractional_area_opt = np.array([[x0[0]], [x0[1]], [x0[2]], [x0[3]]])
    ssa = x0[4] * 100
    lap = x0[5] * 1e-5
    lwc = x0[6]
    h = x0[7] * 100
    aod = x0[8]

    # Optimal terrain
    if optimal_cosi == 'yes':
        aspect = new_aspect(x0[9], x0[10])
        cosi, cosv, theta = new_terrain(slope, aspect, sza, vza, saa, vaa)

    # Call the snow model
    reflectance, _ = art(ssa, lap, lwc, cosi, cosv, theta, sensor_wavelengths)
    snow_reflectance_transpose = reflectance.T

    # Replace the new inversion into the endmember array
    endmembers_opt[:, 0] = snow_reflectance_transpose

    # estimate rho_surface
    rho_surface = np.matmul(endmembers_opt, fractional_area_opt).flatten()

    # Get the local irrad and atmos params.
    # This is loading the data from the precomputed LUTs.
    l0, t_up, sph_alb, s_total = lrt_reader(h, aod, alt,cosi, sza, 
                                            shadow,svf, slope, rho_surface,
                                            g_l0, g_tup, g_s, g_edir, g_edn,
                                            sensor_wavelengths) 

    # Compute L_TOA
    l_toa = l0 +  (1/np.pi) * ((rho_surface * s_total* t_up) / (1 - sph_alb * rho_surface))

    # Input the values into the system of equations
    residual =  l_toa - observed_radiance_transpose

    # Remove selected bands from residual
    residual = np.delete(residual, ix)

    # RMSE for minimization
    rmse = rmse_calc(residual)
    
    #print(x0)
    #import matplotlib.pyplot as plt
    #plt.scatter(sensor_wavelengths, l_toa, c='magenta', s=10)
    #plt.scatter(sensor_wavelengths, observed_radiance_transpose, c='k', s=10)
    #plt.scatter(sensor_wavelengths, rho_surface, c='magenta', s=5)
    #plt.scatter(sensor_wavelengths, endmembers_opt[:,2], c='k', s=5)
    #plt.xlabel('Wavelength')
    #plt.ylabel('$L_{TOA} microW cm-2 nm-1 sr-1$')
    #plt.legend(['Modeled','Observed'])
    #plt.show()
    #print(x0)
    #print(rmse)

    return rmse



def fun_min_light(x0, cosi, cosv, theta, sensor_wavelengths, endmembers_opt,
            sza, vza, saa, vaa, 
            alt, shadow, svf, slope, aspect, 
            observed_radiance_transpose,
            g_l0, g_tup, g_s, g_edir, g_edn, 
            ix, lap, lwc, fractional_area_opt,h,aod):
    '''
    Define the objective function.
    
    Args:


    Returns:
        rmse (float): Root mean squared error.
    '''

    # Call the snow model
    reflectance, _ = art(x0, lap, lwc, cosi, cosv, theta, sensor_wavelengths)
    snow_reflectance_transpose = reflectance.T

    # Replace the new inversion into the endmember array
    endmembers_opt[:, 0] = snow_reflectance_transpose

    # estimate rho_surface
    rho_surface = np.matmul(endmembers_opt, fractional_area_opt).flatten()

    # Get the local irrad and atmos params.
    # This is loading the data from the precomputed LUTs.
    l0, t_up, sph_alb, s_total = lrt_reader(h, aod, alt,cosi, sza, 
                                            shadow,svf, slope, rho_surface,
                                            g_l0, g_tup, g_s, g_edir, g_edn,
                                            sensor_wavelengths) 

    # Compute L_TOA
    l_toa = l0 +  (1/np.pi) * ((rho_surface * s_total* t_up) / (1 - sph_alb * rho_surface))

    # Input the values into the system of equations
    residual =  l_toa - observed_radiance_transpose

    # Remove selected bands from residual
    residual = np.delete(residual, ix)

    # RMSE for minimization
    rmse = rmse_calc(residual)

    return rmse




def invert_snow_and_atmos_props(i, r, alt, cosi, cosv, 
                                theta,slope,aspect, svf, 
                                lc,shadow,
                                sensor_wavelengths, 
                                lat_mean, lon_mean, 
                                sza, vza, saa, vaa,
                                g_l0, g_tup, g_s, g_edir, g_edn, optimal_cosi):
    '''
    Compute from the combo list found from kmeans clustering (or for a single pixel)


    '''
    
    # cosi was allowed to vary from -1 to 1 during clustering to create more groups
    # But now I am bringing it back into physical reality by making it range from 0-1
    # Note on TARTES, it cannot be exactly 0.0 here so using a very small val (1e-16)
    if cosi <= 0.0:
        cosi = 1e-16

    # Transpose the observed reflectance to 1 column
    observed_radiance_transpose = r.T

    # Correct the shape to be (bands,1)
    observed_radiance_transpose = observed_radiance_transpose.reshape(-1, 1).flatten()

    # bring initial endmembers for optimization
    endmembers_opt = initial_endmembers(sensor_wavelengths, lc, lat_mean, lon_mean) 

    # Remove bands not in inversion window for snow and atmos.
    sensor_wavelengths_transpose = sensor_wavelengths.T
    ix_1 = np.argwhere((sensor_wavelengths_transpose >= 2450) & (sensor_wavelengths_transpose <= 2700))
    ix_2 = np.argwhere((sensor_wavelengths_transpose >= 1780) & (sensor_wavelengths_transpose <= 1950))
    ix_3 = np.argwhere((sensor_wavelengths_transpose >= 1300) & (sensor_wavelengths_transpose <= 1450))
    ix_4 = np.argwhere((sensor_wavelengths_transpose >= 300) & (sensor_wavelengths_transpose <= 400))
    ix = np.concatenate((ix_1, ix_2, ix_3, ix_4))


    # fractional covers sum to 1
    def constraint(x):
        return 1 - sum(x[0:4])
    con = {'type': 'eq', 'fun': constraint}
    opt = {'maxiter': 50}  # TESTING disp:True
    
    # lower upper bounds for SLSQP
    # NOTE: ssa and LAP are scaled in order to improve gradient during search
    lb_snow = 0.0
    up_snow = 1.0
    lb_shade = 0.0
    up_shade = 1.0
    lb_em1 = 0.0
    up_em1 = 1.0
    lb_em2 = 0.0
    up_em2 = 1.0
    lb_ssa = 0.02
    up_ssa = 1.56
    lb_lap = 0.0
    up_lap = 0.5
    lb_lwc = 0.0
    up_lwc = 0.5
    lb_h = 0.01
    up_h = 0.5
    lb_aod = 0.01
    up_aod = 1
    
    north_start = np.cos(np.radians(aspect))
    east_start = np.sin(np.radians(aspect))

    lb_east_north = -1.0
    up_east_north = 1.0

    # Run 
    # initial conniditons in order from left to right
    if optimal_cosi == 'yes':
        #(fsnow,fshade,fem1,fem2,ssa,LAP,LWC,h,aod,east, north)
        x0 = [0.1, 0.2, 0.5,0.2,0.4,0.0, 0.02, 0.01, 0.1, east_start, north_start]
        lb = [lb_snow, lb_shade, lb_em1 , lb_em2, lb_ssa, lb_lap, lb_lwc, lb_h, lb_aod, lb_east_north, lb_east_north]
        ub = [up_snow, up_shade, up_em1 , up_em2, up_ssa, up_lap, up_lwc, up_h, up_aod, up_east_north, up_east_north]

    else:
        #(fsnow,fshade,fem1,fem2,ssa,LAP,LWC,h,aod)
        x0 = [0.1, 0.2, 0.5,0.2,0.4,0.0, 0.02, 0.01, 0.1]
        lb = [lb_snow, lb_shade, lb_em1 , lb_em2, lb_ssa, lb_lap, lb_lwc, lb_h, lb_aod]
        ub = [up_snow, up_shade, up_em1 , up_em2, up_ssa, up_lap, up_lwc, up_h, up_aod]

    bounds =  optimize.Bounds(lb=lb, ub=ub, keep_feasible=True)
    opt_result = optimize.minimize(fun_min, x0,
                            args=(cosi, cosv, theta, sensor_wavelengths, endmembers_opt,
                                  sza, vza, saa, vaa, 
                                  alt, shadow,
                                  svf, slope, aspect, 
                                  observed_radiance_transpose,
                                  g_l0, g_tup, g_s, g_edir, g_edn, 
                                  ix, optimal_cosi),
                                  method='SLSQP',
                                  constraints=con, bounds=bounds, options=opt)
    
    # Save updated model results
    xfinal = opt_result.x
    f_snow = xfinal[0]
    f_shade = xfinal[1]
    f_em1 = xfinal[2]
    f_em2 = xfinal[3]
    ssa_opt = xfinal[4]*100
    lap_opt = xfinal[5]*1e-5 * 5.50e9 #scaling to physically meaninful values, Bond,2006 [ng/g]
    lwc_opt = xfinal[6]
    h20_opt = xfinal[7]*100
    aod_opt = xfinal[8]
    if optimal_cosi == 'yes':
        aspect = new_aspect(xfinal[9], xfinal[10])
        cosi, cosv, theta = new_terrain(slope, aspect, sza, vza, saa, vaa)
    rmse = opt_result.fun

    fractional_area_opt = np.array([[f_snow], [f_shade], [f_em1], [f_em2]])


    # Run one more time but this time to solve closer for SSA
    ix_1 = np.argwhere((sensor_wavelengths_transpose >= 2450) & (sensor_wavelengths_transpose <= 2700))
    ix_2 = np.argwhere((sensor_wavelengths_transpose >= 1780) & (sensor_wavelengths_transpose <= 1950))
    ix_3 = np.argwhere((sensor_wavelengths_transpose >= 1300) & (sensor_wavelengths_transpose <= 1450))
    ix_4 = np.argwhere((sensor_wavelengths_transpose >= 300) & (sensor_wavelengths_transpose <= 900))
    ix = np.concatenate((ix_1, ix_2, ix_3, ix_4))

    # (ssa)
    x0 = [ssa_opt]

    opt_result = optimize.minimize(fun_min_light, x0,
                            args=(cosi, cosv, theta, sensor_wavelengths, endmembers_opt,
                                sza, vza, saa, vaa, 
                                alt, shadow,
                                svf, slope, aspect, 
                                observed_radiance_transpose,
                                g_l0, g_tup, g_s, g_edir, g_edn, 
                                ix, lap_opt, lwc_opt, 
                                fractional_area_opt,
                                h20_opt,aod_opt),
                                method='Nelder-Mead')
    
    # Save updated model results
    xfinal = opt_result.x
    ssa_opt = xfinal[0]
    rmse = opt_result.fun



    # If everything looks good continue with calling ART one more time to save outputs
    rho_s , alb_snow = art(ssa_opt, lap_opt, lwc_opt, cosi, cosv, theta, sensor_wavelengths)
    snow_reflectance_transpose = rho_s.T
    endmembers_opt[:, 0] = snow_reflectance_transpose
    rho_surface = np.matmul(endmembers_opt, fractional_area_opt).flatten()

    # Get the local irrad and atmos params.
    # This is loading the data from the precomputed LUTs.
    l0, t_up, sph_alb, s_total = lrt_reader(h20_opt, aod_opt, alt, 
                                            cosi, sza, shadow,
                                            svf, slope, rho_surface,
                                            g_l0, g_tup, g_s, g_edir, g_edn,
                                            sensor_wavelengths)                                          

    # Compute L-TOA again
    l_toa = l0 + (1/np.pi) * ((rho_surface * s_total* t_up) / (1 - sph_alb * rho_surface))

    # Calculate broadband albedo
    broadband = np.trapz(alb_snow * s_total, dx=1) / np.trapz(s_total, dx=1)  

    # Save all the outputs to a list
    combo_opt = [i, #0
                 f_snow, #1 
                 f_shade, #2 
                 f_em1, #3
                 f_em2, #4
                 ssa_opt, #5
                 lap_opt, #6
                 lwc_opt, #7   
                 h20_opt, #8
                 aod_opt, #9
                 broadband, #10
                 rmse, #11
                 l_toa, #12 estimated TOA Radiance
                 rho_s, #13 snow reflectance
                 alb_snow, #14 Plane albedo
                 s_total,#15 total irrad
                 rho_surface,#16 surface reflectance
                 cosi] #17 cosi

    return combo_opt

