from sister.sensors import emit
import netCDF4 as nc

img_file = '/Users/brent/Documents/EMIT_L2A_RFL_001_20230203T184925_2303413_007.nc'
out_dir =  '/Users/brent/Documents/'
temp_dir =  '/Users/brent/Documents/'
export_loc = True




emit.nc_to_envi(img_file,out_dir,temp_dir,
                obs_file=None,export_loc=True,
                crid='000')
