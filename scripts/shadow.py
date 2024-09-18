import numpy as np
from spectral import *
import scipy
import multiprocessing as mp


# run in parallel
def run_ray_pool(list_args, n_cpu):
    with mp.Pool(n_cpu) as pool:
        results = pool.map(ray_trace_mp, list_args)
    return results


# Define wrapper function
def ray_trace_mp(args):
    return ray_trace(*args)


def ray_trace(i, j, dem_arr, cosi, sza, saa, path_to_img_base):

    '''
    loop through all pixels in parallel and compute simple ray tracing
                
    '''


    #start = time.time()

    # Get pix size from satellite and projection
    if 'EMIT' in path_to_img_base:
        pix_size = 60
    else: #PRISMA
        pix_size = 30

    i_lim = dem_arr.shape[0]
    j_lim = dem_arr.shape[1]  
    
    # Compute projected solar directions
    tan_theta_e = np.tan(np.radians(90-sza))
    tan_sundir = -1 * np.tan(np.radians(saa)) # direction flipped because negative is pointing right
    
    # Assumes image is taken mid-day so mover_i(row-wise / N-S) will always be larger than mover_j
    # starts with high sampling and decreases resolution with increasing distance  
    y_mover = np.arange(0,300.1,1)
    x_mover = np.round(y_mover * tan_sundir).astype(int)

    # Southern hemisphere (need to move back up the image instead of down the rows)
    if saa>270 or saa<90:
        y_mover = y_mover * -1

    if sza == 0.0: # NaN data, skip
        return [i,j,-9999]
    
    if cosi <= 0.0: # Already know this is 100% shadow , no direct
        return [i,j,1]
    
    # Run ray tracing
    y = i + y_mover
    x = j + x_mover

    # make sure y_mover and x_mover did not put outside img range
    ix_remove = np.argwhere((y>=i_lim-1) | (x>=j_lim-1) | (y<0) | (x<0))
    y = np.round(np.delete(y, ix_remove))
    x = np.round(np.delete(x, ix_remove))

    # Extract the values along the line, using cubic interpolation
    zi = scipy.ndimage.map_coordinates(dem_arr, np.vstack((y,x)), order=1)

    # Create the sun ray, from the pixel
    h = (np.sqrt(((y-i)*pix_size)**2 + ((x-j)*pix_size)**2)) * tan_theta_e + dem_arr[i,j]

    if ((h[1:] < zi[1:]).any()) == True:
        return [i,j,0]
    else:
        return [i,j,1]

    return

