# Import libraries
import os
import logging
import numpy as np
import pandas as pd
from scipy import interpolate
from spectral import *



def find_pixel(lat,lon,path_to_img_base):
    '''
    Finds the pixel coordinates (i, j) based on the given latitude and longitude.

    Args:
        lat (float): Latitude of the pixel.
        lon (float): Longitude of the pixel.
        path_to_img_base (str): The base path of the image.

    Returns:
        tuple: A tuple containing the pixel coordinates (i, j).
    '''

    loc_path = path_to_img_base + '_loc_prj'
    observed_loc_array =  envi.open(loc_path+'.hdr')
    observed_loc_array = observed_loc_array.open_memmap(writeable=True)
    
    # Find the pixel that matches the user input lat/lon (center)
    # For right now this requires a perfect match (use GIS - rfl to find)
    i_j = np.where((observed_loc_array[:, :, 1] == lat) & (observed_loc_array[:, :, 0] == lon))
    i = i_j[0][0]
    j = i_j[1][0]
    
    return i,j




def initial_endmembers(sensor_wavelengths, landcover_value, lat, lon):
    '''
    Sets four endmembers based on the landcover class for pixel i,j. The endmembers are derived from ecospeclib-all.

    As a note! , I assume no BRDF impacts from the endmembers, and rely solely on the ART model to contribute directional
    impacts from snow only. With this being focused on winter aquisitions, this seems to be a fair assumption.

    However, incorperating BRDF effects of forests and rocks can be revisited later..


    Args:
        sensor_wavelengths (list): List of sensor wavelengths.
        landcover_value (int): Landcover value.
        lat (float): Latitude.
        lon (float): Longitude.

    Returns:
        tuple: A tuple containing the endmember arrays.

    This area can be improved in the future! On a drive through Caribou Poker and looking at all of the beautiful mixed forests,
    I can have a bunch of iff statements here based on the location to switch up for different regions, while not increasing computational cost.

    For example, I would like to soon include Spruce spectra for "boreal" regions. However, there is no spruce in the curent version of ecospec-lib.

    Additionally, as of right now a lot of these classes just end up being granite rock and conifer. Simply because the other more specifc ones like
    water and ice were not working as expected. I think this is fine for now, because in an optically thick setting, you would need see these
    surfaces in this pixel. However, for patchy snow this could be an issue...
    
    Need to find better reference spectra and is an area of improvements for this alg.

    '''

    # CREATE INITIAL ENDMEMBERS...
    eco_dir = './ecospeclib-all'
    if not os.path.exists(eco_dir):
        eco_dir = '../ecospeclib-all'
        if not os.path.exists(eco_dir):
            logging.exception('Please download EcoSpeclib-all from ECOSTRESS and save to working directory.')
            raise Exception('Please download EcoSpeclib-all from ECOSTRESS and save to working directory.')

    # Four endmembers in total (Snow, Shade, Em1, Em2)
    # Endmembers 1&2 are decided based on the landcover value
    # ESA WorldCover 10m v100
    # 10 Trees
    # 20 Shrubland
    # 30 Grassland
    # 40 Cropland
    # 50 Built-up
    # 60 Barren / sparse vegetation
    # 70 Snow and ice
    # 80 Open water
    # 90 Herbaceous wetland
    # 95 Mangroves
    # 100 Moss and lichen

    # Define boreal areas with more complex forest structures
    if (lat>=46 and lon<0) or (lat>55 and lon>0):
        boreal = True
    else:
        boreal = False

    if landcover_value == 10 and boreal is False:
        # Ponderosa Pine
        endmember_1 = pd.read_csv(f'{eco_dir}/vegetation.tree.pinus.ponderosa.vswir.vh249.ucsb.asd.spectrum.txt',
                                  skiprows=20, sep='\s+', header=None)
        endmember_1.columns = ['Wavelength', 'Reflectance']

        # rock outcrop.. granite
        endmember_2 = pd.read_csv(f'{eco_dir}/rock.igneous.felsic.solid.all.granite_h1.jhu.becknic.spectrum.txt',
                                  skiprows=20, sep='\s+', header=None)
        endmember_2.columns = ['Wavelength', 'Reflectance']
    
    elif landcover_value == 10 and boreal is True:
        # Ponderosa Pine (TRY AND FIND SPRUCE SPECTRA!)
        endmember_1 = pd.read_csv(f'{eco_dir}/vegetation.tree.pinus.ponderosa.vswir.vh249.ucsb.asd.spectrum.txt',
                                  skiprows=20, sep='\s+', header=None)
        endmember_1.columns = ['Wavelength', 'Reflectance']

        # Branches
        endmember_2 = pd.read_csv(f'{eco_dir}/nonphotosyntheticvegetation.branches.ceanothus.megacarpus.vswir.vh331.ucsb.asd.spectrum.txt',
                                  skiprows=20, sep='\s+', header=None)
        endmember_2.columns = ['Wavelength', 'Reflectance']

    elif landcover_value == 20:
        # baccharis.pilularis.
        endmember_1 = pd.read_csv(f'{eco_dir}/vegetation.shrub.baccharis.pilularis.vswir.vh006.ucsb.asd.spectrum.txt',
                                  skiprows=20, sep='\s+', header=None)
        endmember_1.columns = ['Wavelength', 'Reflectance']
        # soil
        endmember_2 = pd.read_csv(f'{eco_dir}/soil.entisol.quartzipsamment.none.all.87p706.jhu.becknic.spectrum.txt',
                                  skiprows=20, sep='\s+', header=None)
        endmember_2.columns = ['Wavelength', 'Reflectance']

    elif landcover_value == 30:
        # baccharis.pilularis.
        endmember_1 = pd.read_csv(f'{eco_dir}/vegetation.shrub.baccharis.pilularis.vswir.vh006.ucsb.asd.spectrum.txt',
                                  skiprows=20, sep='\s+', header=None)
        endmember_1.columns = ['Wavelength', 'Reflectance']
        # soil
        endmember_2 = pd.read_csv(f'{eco_dir}/soil.entisol.quartzipsamment.none.all.87p706.jhu.becknic.spectrum.txt',
                                  skiprows=20, sep='\s+', header=None)
        endmember_2.columns = ['Wavelength', 'Reflectance']

    elif landcover_value == 40:
        # baccharis.pilularis.
        endmember_1 = pd.read_csv(f'{eco_dir}/vegetation.shrub.baccharis.pilularis.vswir.vh006.ucsb.asd.spectrum.txt',
                                  skiprows=20, sep='\s+', header=None)
        endmember_1.columns = ['Wavelength', 'Reflectance']
        # rock outcrop.. granite
        endmember_2 = pd.read_csv(f'{eco_dir}/rock.igneous.felsic.solid.all.granite_h1.jhu.becknic.spectrum.txt',
                                  skiprows=20, sep='\s+', header=None)
        endmember_2.columns = ['Wavelength', 'Reflectance']

    elif landcover_value == 50:
        # Asphalt road
        endmember_1 = pd.read_csv(f'{eco_dir}/manmade.road.pavingasphalt.solid.all.0674uuuasp.jhu.becknic.spectrum.txt', #urban
                                  skiprows=20, sep='\s+', header=None)
        endmember_1.columns = ['Wavelength', 'Reflectance']
        # rock outcrop.. granite
        endmember_2 = pd.read_csv(f'{eco_dir}/rock.igneous.felsic.solid.all.granite_h1.jhu.becknic.spectrum.txt',
                                  skiprows=20, sep='\s+', header=None)
        endmember_2.columns = ['Wavelength', 'Reflectance']

    elif landcover_value == 60:
        # soil
        endmember_1 = pd.read_csv(f'{eco_dir}/soil.entisol.quartzipsamment.none.all.87p706.jhu.becknic.spectrum.txt',
                                  skiprows=20, sep='\s+', header=None)
        endmember_1.columns = ['Wavelength', 'Reflectance']
        # rock outcrop.. granite
        endmember_2 = pd.read_csv(f'{eco_dir}/rock.igneous.felsic.solid.all.granite_h1.jhu.becknic.spectrum.txt',
                                  skiprows=20, sep='\s+', header=None)
        endmember_2.columns = ['Wavelength', 'Reflectance']

    elif landcover_value == 70:
        # soil
        endmember_1 = pd.read_csv(f'{eco_dir}/soil.entisol.quartzipsamment.none.all.87p706.jhu.becknic.spectrum.txt',
                                  skiprows=20, sep='\s+', header=None)
        endmember_1.columns = ['Wavelength', 'Reflectance']
        
        # rock outcrop.. granite
        endmember_2 = pd.read_csv(f'{eco_dir}/rock.igneous.felsic.solid.all.granite_h1.jhu.becknic.spectrum.txt',
                                  skiprows=20, sep='\s+', header=None)
        endmember_2.columns = ['Wavelength', 'Reflectance']

    elif landcover_value == 80:
        # Ponderosa Pine
        endmember_1 = pd.read_csv(f'{eco_dir}/vegetation.tree.pinus.ponderosa.vswir.vh249.ucsb.asd.spectrum.txt',
                                  skiprows=20, sep='\s+', header=None)
        endmember_1.columns = ['Wavelength', 'Reflectance']

        # soil
        endmember_2 = pd.read_csv(f'{eco_dir}/soil.entisol.quartzipsamment.none.all.87p706.jhu.becknic.spectrum.txt',
                                  skiprows=20, sep='\s+', header=None)
        endmember_2.columns = ['Wavelength', 'Reflectance']

    elif landcover_value == 90:
        # Ponderosa Pine
        endmember_1 = pd.read_csv(f'{eco_dir}/vegetation.tree.pinus.ponderosa.vswir.vh249.ucsb.asd.spectrum.txt',
                                  skiprows=20, sep='\s+', header=None)
        endmember_1.columns = ['Wavelength', 'Reflectance']

        # soil
        endmember_2 = pd.read_csv(f'{eco_dir}/soil.entisol.quartzipsamment.none.all.87p706.jhu.becknic.spectrum.txt',
                                  skiprows=20, sep='\s+', header=None)
        endmember_2.columns = ['Wavelength', 'Reflectance']

    elif landcover_value == 95:
        # Ponderosa Pine
        endmember_1 = pd.read_csv(f'{eco_dir}/vegetation.tree.pinus.ponderosa.vswir.vh249.ucsb.asd.spectrum.txt',
                                  skiprows=20, sep='\s+', header=None)
        endmember_1.columns = ['Wavelength', 'Reflectance']

        # soil
        endmember_2 = pd.read_csv(f'{eco_dir}/soil.entisol.quartzipsamment.none.all.87p706.jhu.becknic.spectrum.txt',
                                  skiprows=20, sep='\s+', header=None)
        endmember_2.columns = ['Wavelength', 'Reflectance']
    else:
        # Ponderosa Pine
        endmember_1 = pd.read_csv(f'{eco_dir}/vegetation.tree.pinus.ponderosa.vswir.vh249.ucsb.asd.spectrum.txt',
                                  skiprows=20, sep='\s+', header=None)
        endmember_1.columns = ['Wavelength', 'Reflectance']

        # soil
        endmember_2 = pd.read_csv(f'{eco_dir}/soil.entisol.quartzipsamment.none.all.87p706.jhu.becknic.spectrum.txt',
                                  skiprows=20, sep='\s+', header=None)
        endmember_2.columns = ['Wavelength', 'Reflectance']

    # Match sensor_wavelength using the spline function
    wave_range = sensor_wavelengths / 1000  # convert to um
    f_1 = interpolate.interp1d(endmember_1['Wavelength'], endmember_1['Reflectance'], kind='slinear', fill_value='extrapolate')
    em1_spline_endmember = (f_1(wave_range)).T

    f_2 = interpolate.interp1d(endmember_2['Wavelength'], endmember_2['Reflectance'], kind='slinear', fill_value='extrapolate')
    em2_spline_endmember = (f_2(wave_range)).T

    # Provide initial snow reflectance for speeding up optimization
    snow_endmember = pd.read_csv(f'{eco_dir}/water.snow.mediumgranular.medium.all.medgran_snw_.jhu.becknic.spectrum.txt',
                                 skiprows=20, sep='\s+', header=None)
    snow_endmember.columns = ['Wavelength', 'Reflectance']
    f_snow = interpolate.interp1d(snow_endmember['Wavelength'], snow_endmember['Reflectance'], kind='slinear', fill_value='extrapolate')
    snow_spline_endmember = (f_snow(wave_range)).T

    # Getting shade reflectance
    # From Bair et al 2022 
    # "This study emphasizes the difficulties in modeling lighting conditions on the snow surface. 
    # Because of these difficulties, a recommendation is to always use a shade endmember in unmixing models, 
    # even for in situ spectroscopic measurements. Likewise, snow albedo models should produce apparent albedos
    # by accounting for the shade fraction."
    em_shade = snow_spline_endmember * 0.0

    # Stack all of the endmember arrays .. divide by 100 because of the ECOSPECLIB format
    endmembers_og = np.vstack((snow_spline_endmember, em_shade, em1_spline_endmember, em2_spline_endmember)).T / 100

    return endmembers_og