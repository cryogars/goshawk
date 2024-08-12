# Import libraries
from spectral import *
import hytools as ht
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from simple_snow import art
from scipy import optimize
import matplotlib.pyplot as plt
import itertools

import multiprocessing

from ssa_functions import ssa_fun_pool







if __name__ == '__main__':


    # FOR each of the 4 aviris flightlienes
    aviris = ['/Users/brent/Documents/Albedo/AVIRIS/ang20210429t185512_rfl_v2z1',
            #'/Users/brent/Documents/Albedo/AVIRIS/ang20210429t185927_rfl_v2z1',
            #'/Users/brent/Documents/Albedo/AVIRIS/ang20210429t190537_rfl_v2z1',
            #'/Users/brent/Documents/Albedo/AVIRIS/ang20210429t191025_rfl_v2z1', 
    ]

    for av in aviris:

        # Open
        rfl = envi.open(av+'.hdr')
        rfl = rfl.open_memmap(writeable=True)
        hy_obj = ht.HyTools()
        hy_obj.read_file(av)
        sensor_wavelengths = hy_obj.wavelengths
        print(f'{av[-27:]}: file started.')

        # get the cosi, cosv, and theta from the obs file
        obs = envi.open(av+'_obs_ort.hdr')
        obs = obs.open_memmap(writeable=True)
        s = np.copy(obs[:,:,6])
        a = np.copy(obs[:,:,7])
        cosi = np.copy(obs[:,:,8])
        cosv = np.copy(np.sin(obs[:,:,2]*(np.pi/180))*np.sin(s*(np.pi/180))*np.cos(obs[:,:,1]*(np.pi/180)-a*(np.pi/180))+np.cos(obs[:,:,2]*(np.pi/180))*np.cos(s*(np.pi/180)))
        cos_raa = np.copy(np.cos(np.radians(180 - (obs[:,:,3] - obs[:,:,1]))))
        cosi[cosi<=0.0] = 0.0
        cosi[cosi>=1.0] = 1.0
        sini = np.sin(np.arccos(cosi))
        cosv[cosv<=0.0] = 0.0
        cosv[cosv>=1.0] = 1.0
        sinv = np.sin(np.arccos(cosv))
        theta = np.degrees(np.arccos(-cosi*cosv + sini*sinv*cos_raa)) 
        print(f'{av[-27:]}: angles computed.')

        # Create NDSI
        wave_600nm = hy_obj.get_wave(600)
        wave_1500nm = hy_obj.get_wave(1500)
        ndsi = (wave_600nm - wave_1500nm) / (wave_600nm + wave_1500nm)
        print(f'{av[-27:]}: ndsi computed.')

        # create args for parallel
        myArgs = []
        for i in range(rfl.shape[0]):
            for j in range(rfl.shape[1]):
                myArgs.append([i,j, rfl[i,j,:], cosi[i,j], cosv[i,j], theta[i,j], ndsi[i,j], sensor_wavelengths])
        
        print(f'{av} starting opts...')
        with multiprocessing.Pool(5) as pool:
            results = pool.map(ssa_fun_pool, myArgs)


        # SLSQP optimization for wavelengths around 1000-1200
        # Iff SSA <2.5 or >155.5 == -9999
        ssa = np.empty_like(ndsi)
        lwc = np.empty_like(ndsi)
        for r in results:
            i = int(r[0])
            j = int(r[1])
            ssa_ij = r[2]
            lwc_ij = r[3]
            if ndsi[i,j] >= 0.90:
                ssa[i,j]= ssa_ij
                lwc[i,j] = lwc_ij
            else:
                ssa[i,j] = -9999
                lwc[i,j] = -9999

        # write SSA raster to disk
        # Create a new hy-tools object for snow properties output
        snow_header = hy_obj.get_header()
        snow_header['bands'] = 1
        snow_header['band_names'] = 'ssa'
        snow_header['wavelength units'] = np.nan
        snow_header['wavelength'] = np.nan
        writer = ht.io.WriteENVI(f'/Users/brent/Code/AVIRIS/data/ssa_{av[-27:]}', snow_header)
        writer.write_band(ssa,0)
        writer.close()

        # write LWC raster to disk
        # Create a new hy-tools object for snow properties output
        snow_header = hy_obj.get_header()
        snow_header['bands'] = 1
        snow_header['band_names'] = 'lwc'
        snow_header['wavelength units'] = np.nan
        snow_header['wavelength'] = np.nan
        writer = ht.io.WriteENVI(f'/Users/brent/Code/AVIRIS/data/lwc_{av[-27:]}', snow_header)
        writer.write_band(lwc,0)
        writer.close()
