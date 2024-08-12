# Import libraries
from spectral import *
import numpy as np
from simple_snow import art
from scipy import optimize



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




def fun_min(x0, spectra, cosi_ij, cosv_ij, theta_ij, 
            sensor_wavelengths,ix):
    '''
    Obj func

    '''
    
    # call AART
    reflectance, _ = art(x0[2]*100, 0, x0[3], cosi_ij, cosv_ij, theta_ij, sensor_wavelengths)
    reflectance = reflectance * x0[0]

    # Input the values into the system of equations
    residual =  spectra - reflectance

    # Remove selected bands from residual
    residual = np.delete(residual, ix)

    # RMSE for minimization
    rmse = rmse_calc(residual)


    return rmse



# Define mpi wrapper function
def ssa_fun_pool(args):
    return ssa_fun(*args)





def ssa_fun(i, j, spectra, cosi_ij, cosv_ij, theta_ij, ndsi_ij, sensor_wavelengths):
    '''
    produce ssa
    
    '''
    if ndsi_ij < 0.90:
        return [i,j, -9999, -9999]

    # Remove bands not in inversion window for snow and atmos.
    sensor_wavelengths_transpose = sensor_wavelengths.T
    ix_1 = np.argwhere((sensor_wavelengths_transpose >= 2450) & (sensor_wavelengths_transpose <= 2700))
    ix_2 = np.argwhere((sensor_wavelengths_transpose >= 1780) & (sensor_wavelengths_transpose <= 1950))
    ix_3 = np.argwhere((sensor_wavelengths_transpose >= 1300) & (sensor_wavelengths_transpose <= 1450))
    ix_4 = np.argwhere((sensor_wavelengths_transpose >= 300) & (sensor_wavelengths_transpose <= 900))
    ix = np.concatenate((ix_1, ix_2, ix_3, ix_4))

     #(fsnow,ssa,LWC)
    x0 = [0.9,0.1, 0.25, 0.01]
    
    # fractional covers sum to 1
    def constraint(x):
        return 1 - sum(x[0:2])
    con = {'type': 'eq', 'fun': constraint}
    opt = {'maxiter': 50}  # TESTING disp:True

    bounds = optimize.Bounds(lb=[0.0,0.0, 0.02, 0.0], ub=[1.0,1.0, 1.56,0.5], keep_feasible=True)

    opt_result = optimize.minimize(fun_min, x0, 
                                   args=(spectra, cosi_ij, cosv_ij, theta_ij, sensor_wavelengths,ix),
                                   method='SLSQP',
                                   bounds=bounds, options=opt, constraints=con)
        
    # Save updated model results
    xfinal = opt_result.x
    ssa_ij = xfinal[2]*100
    lwc = xfinal[3]
    fsnow = xfinal[0]

    # no data
    if ssa_ij >= 155 or ssa_ij <=2:
        ssa_ij = -9999
        lwc = -9999


    return [i, j, ssa_ij,lwc]