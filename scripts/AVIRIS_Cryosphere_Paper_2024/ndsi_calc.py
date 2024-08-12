# Import libraries
from spectral import *
import hytools as ht
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from simple_snow import art
from scipy import optimize
import matplotlib.pyplot as plt



# FOR each of the 4 aviris flightlienes
aviris = ['/Users/brent/Documents/Albedo/AVIRIS/ang20210429t185512_rfl_v2z1',
          '/Users/brent/Documents/Albedo/AVIRIS/ang20210429t185927_rfl_v2z1',
          '/Users/brent/Documents/Albedo/AVIRIS/ang20210429t190537_rfl_v2z1',
          '/Users/brent/Documents/Albedo/AVIRIS/ang20210429t191025_rfl_v2z1', 
]

for av in aviris:

    # Open
    rfl = envi.open(av+'.hdr')
    rfl = rfl.open_memmap(writeable=True)
    hy_obj = ht.HyTools()
    hy_obj.read_file(av)
    sensor_wavelengths = hy_obj.wavelengths
    print(f'{av[-27:]}: file started.')

    # Create NDSI
    wave_600nm = hy_obj.get_wave(600)
    wave_1500nm = hy_obj.get_wave(1500)
    ndsi = (wave_600nm - wave_1500nm) / (wave_600nm + wave_1500nm)
    ndsi[ndsi < -0.7] = -9999
    print(f'{av[-27:]}: ndsi computed.')


    # write SSA raster to disk
    # Create a new hy-tools object for snow properties output
    snow_header = hy_obj.get_header()
    snow_header['bands'] = 1
    snow_header['band_names'] = 'ssa'
    snow_header['wavelength units'] = np.nan
    snow_header['wavelength'] = np.nan
    writer = ht.io.WriteENVI(f'/Users/brent/Code/AVIRIS/data/{av[-27:]}_ndsi', snow_header)
    writer.write_band(ndsi,0)
    writer.close()
