# Import libraries
import numpy as np


def simple_cloud_threshold(rad_array, sensor_wavelengths, 
                    cloud_wl1 = 1994, 
                    cloud_wl2 = 2490, 
                    min_threshold=13): 
    
    '''
    TODO

    cloud-wl - nm

    min-threshold - prisma microW cm-2 nm-1 sr-1


    '''

    clouds = np.zeros_like(rad_array[:,:,1])

    # find closest to cloud_wl
    rad_cloud1 = rad_array[:,:,np.argmin(np.abs(sensor_wavelengths - cloud_wl1))]
    rad_cloud2 = rad_array[:,:,np.argmin(np.abs(sensor_wavelengths - cloud_wl2))]

    # Find max rad
    max_rad = np.amax(rad_array, axis=2)

    # assign cloud
    clouds[(max_rad>min_threshold) & (rad_cloud1 > 0.13)  & (rad_cloud2 > 0.12)] = 1

    # Assign Nan
    clouds[(rad_array[:,:,0] == -9999)] = -9999

    return clouds