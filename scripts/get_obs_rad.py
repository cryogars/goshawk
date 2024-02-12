# Just a short utility script to pull the radiance of a given pixel and save to a csv
import numpy as np
import pandas as pd
from spectral import *
import hytools as ht
from initialize import find_pixel

# USER INPUTS HERE
lat = 39.216816 # from GIS (note this must be exact from the SISTER loc file)
lon = -108.183655  # from GIS
path_to_img_base = '/Users/brent/Documents/Albedo/PRISMA/20210406_GRANDMESA/albedo/PRS_20210406180740_20210406180744_0001'
rad_csv_output = '/Users/brent/Code/SPHERES/test/rad/cloud_test_yes.csv'
# END USER INPUTS

# Get the ENVI paths from SISTER/ISOFIT
rad_path = path_to_img_base + '_rdn_prj'
loc_path = path_to_img_base + '_loc_prj'

# Use Spectral python library to do the actual calcs
observed_rad = envi.open(rad_path+'.hdr')
observed_rad_array = observed_rad.open_memmap(writeable=True)
observed_loc = envi.open(loc_path+'.hdr')
observed_loc_array = observed_loc.open_memmap(writeable=True)

# Create a HyTools container object
hy_obj = ht.HyTools()

# Read and load ENVI file metadata
hy_obj.read_file(rad_path)
sensor_wavelengths = hy_obj.wavelengths

# Find pixel
i,j = find_pixel(lat,lon,path_to_img_base)
rad = observed_rad_array[i,j,:]

# Save to CSV
df = pd.DataFrame({"Wavelength" : sensor_wavelengths, "microWcm-2nm-1sr-1" : rad})
df.to_csv(rad_csv_output, index=False)
