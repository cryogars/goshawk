# Import libraries
import os
import subprocess
import numpy as np
import pandas as pd
from spectral import *
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator


def lrt_prepper(path_to_img_base, sza_array, selected_elev):

    '''
    TODO

    '''

    # Load loc file
    loc_path = path_to_img_base + '_loc_prj'
    observed_loc_array =  envi.open(loc_path+'.hdr')
    observed_loc_array = observed_loc_array.open_memmap(writeable=True)

    # Load obs file
    obs_path = path_to_img_base + '_obs_prj'
    observed_obs_array =  envi.open(obs_path+'.hdr')
    observed_obs_array = observed_obs_array.open_memmap(writeable=True)

    # Get avg values for scene 
    lat = observed_loc_array[:, :, 1].flatten()
    lat[(lat==-9999) | (lat>500)] = np.nan
    lat = lat[~np.isnan(lat)]
    lat = np.mean(lat)

    lon = observed_loc_array[:, :, 0].flatten()
    lon[(lon==-9999) | (lon>500)] = np.nan
    lon = lon[~np.isnan(lon)]
    lon = np.mean(lon)

    # Have to drop -NaN before taking mean
    sza_f = sza_array.flatten()
    sza_f[(sza_f<=0) | (sza_f>90)] = np.nan
    sza_f = sza_f[~np.isnan(sza_f)]
    sza = np.mean(sza_f)  
    
    elev = selected_elev[:, :].flatten()
    elev = elev[elev < 8848] #mt everest
    elev = elev[elev >= 0] #ocean
    elev = elev[~np.isnan(elev)]
    elev_min = np.min(elev)
    elev_mean = np.mean(elev)
    elev_max = np.max(elev)
    alt_min = elev_min / 1000 # convert to km
    alt_mean = elev_mean / 1000
    alt_max = elev_max / 1000 # convert to km

    # Check to use subarctic or midlat winter atmosphere
    if abs(lat) >= 60:
        atmos = 'sw'
    else:
        atmos = 'mw'

    # Assign N / S / E / W
    if lat >= 0:
        lat_inp = str(f'N {abs(lat)}')
    else:
        lat_inp = str(f'S {abs(lat)}')

    if lon >= 0:
        lon_inp = str(f'E {abs(lon)}')
    else:
        lon_inp = str(f'W {abs(lon)}')

    # Calculate all the angles
    vza = observed_obs_array[:, :, 2].flatten()
    vza[(vza==-9999) | (vza>500)] = np.nan
    vza = vza[~np.isnan(vza)]
    vza = np.mean(vza)  

    umu = np.cos(np.radians(vza))

    phi0 = observed_obs_array[:, :, 3].flatten()
    phi0[(phi0==-9999) | (phi0>500)] = np.nan
    phi0 = phi0[~np.isnan(phi0)]
    phi0 = np.mean(phi0) 

    phi = observed_obs_array[:, :, 1].flatten()
    phi[(phi==-9999) | (phi>500)] = np.nan
    phi = phi[~np.isnan(phi)]
    phi = np.mean(phi)  

    return vza, umu, phi0, phi, sza, lat_inp, lon_inp, alt_min, alt_max, atmos



def write_lrt_inp(h , aod, a, out_str, umu, phi0, phi, sza, lat_inp, lon_inp, doy, altitude_km,
              atmos, path_to_libradtran_bin, lrt_dir, path_to_libradtran_base):
    '''

    adapted from: https://github.com/MarcYin/libradtran


    '''
    foutstr = out_str[0] + out_str[1]
    fname = f'{lrt_dir}/lrt_h20_{h}_aot_{aod}_alb_{a}_alt_{round(altitude_km*1000)}_{foutstr}'
    with open(f'{fname}.INP', 'w') as f:
        f.write('source solar\n')  # extraterrestrial spectrum
        f.write('wavelength 340 2510\n')  # set range for lambda
        f.write(f'atmosphere_file {path_to_libradtran_base}/data/atmmod/afgl{atmos}.dat\n')
        f.write(f'albedo {a}\n')  # TODO
        f.write(f'umu {umu}\n') # Cosine of the view zenith angle
        f.write(f'phi0 {phi0}\n') # SAA
        f.write(f'phi {phi}\n') # VAA
        f.write(f'sza {sza}\n')  # solar zenith angle
        f.write('rte_solver disort\n')  # set twostr as the RTM
        f.write('pseudospherical\n')# computed with spherical instead of plane parallel
        f.write(f'latitude {lat_inp}\n')
        f.write(f'longitude {lon_inp}\n')
        f.write(f'day_of_year {doy}\n')  # DOY
        f.write(f'mol_modify O3 300 DU\n')  #  
        f.write(f'mol_abs_param reptran coarse\n')  #  
        f.write(f'mol_modify H2O {h} MM\n')  #  
        f.write(f'crs_model rayleigh bodhaine \n')  # 
        f.write(f'zout {out_str[0]}\n')  # sat
        f.write(f'altitude {altitude_km}\n')  # sat   
        f.write(f'aerosol_default\n')  # 
        f.write(f'aerosol_species_file continental_average\n')  # 
        f.write(f'aerosol_set_tau_at_wvl 550 {aod}\n')  #    
        f.write(f'output_quantity transmittance\n')  #outputs
        f.write(f'output_user lambda {out_str[1]}\n')  #outputs  
        f.write('quiet')
    cmd = f'{path_to_libradtran_bin}/uvspec < {fname}.INP > {fname}.out'
    return cmd




def write_lrt_inp_irrad(h , aod, a, out_str, umu, phi0, phi, sza, lat_inp, lon_inp, doy, altitude_km,
                        atmos, path_to_libradtran_bin, lrt_dir, path_to_libradtran_base):
    # Run here manually for irrad
    fname = f'{lrt_dir}/lrt_h20_{h}_aot_{aod}_alt_{round(altitude_km*1000)}_IRRAD'
    with open(f'{fname}.INP', 'w') as f:
        f.write('source solar\n')  # extraterrestrial spectrum
        f.write('wavelength 340 2510\n')  # set range for lambda
        f.write(f'atmosphere_file {path_to_libradtran_base}/data/atmmod/afgl{atmos}.dat\n')
        f.write(f'sza {sza}\n')  # solar zenith angle
        f.write('rte_solver disort\n')  # set 
        f.write('pseudospherical\n')# computed with spherical instead of plane parallel
        f.write(f'latitude {lat_inp}\n')
        f.write(f'longitude {lon_inp}\n')
        f.write(f'day_of_year {doy}\n')  # DOY
        f.write(f'zout {altitude_km}\n')  # 
        f.write(f'aerosol_default\n')  # 
        f.write(f'aerosol_species_file continental_average\n')  # 
        f.write(f'aerosol_set_tau_at_wvl 550 {aod}\n')  #   
        f.write(f'mol_modify O3 300 DU\n')  #  
        f.write(f'mol_abs_param reptran coarse\n')  #  
        f.write(f'mol_modify H2O {h} MM\n')  #  
        f.write(f'crs_model rayleigh bodhaine \n')  # 
        f.write(f'output_user lambda edir edn \n')  #outputs  
        f.write('quiet')
    cmd = f'{path_to_libradtran_bin}/uvspec < {fname}.INP > {fname}.out'
    return cmd




def lut_grid(h20_range,a550_range, alt_range, path_to_img_base, sensor_wavelengths):
    '''

    Grid the LUT so they are continuous variables for numerical optimization.

    '''
    # LRT dir
    lrt_dir = f'{path_to_img_base}_albedo/libradtran'

    # 
    l0_arr = np.empty(shape=(len(h20_range),len(a550_range),len(alt_range), len(sensor_wavelengths)))
    t_up_arr = np.empty(shape=(len(h20_range),len(a550_range),len(alt_range), len(sensor_wavelengths)))
    s_arr  = np.empty(shape=(len(h20_range),len(a550_range),len(alt_range), len(sensor_wavelengths)))
    edir_arr = np.empty(shape=(len(h20_range),len(a550_range),len(alt_range), len(sensor_wavelengths)))
    edn_arr = np.empty(shape=(len(h20_range),len(a550_range),len(alt_range), len(sensor_wavelengths)))

    for i in range(0, len(h20_range)):
        for j in range(0, len(a550_range)):
            for k in range(0, len(alt_range)):

                h = h20_range[i]
                aod = a550_range[j]
                altitude_km = alt_range[k]

                # Now load in each of them into pandas to perform math.
                df_r = pd.read_csv(f'{lrt_dir}/lrt_h20_{h}_aot_{aod}_alb_0_alt_{round(altitude_km*1000)}_toauu.out', delim_whitespace=True, header=None)
                df_r.columns = ['Wavelength','uu']

                df_t = pd.read_csv(f'{lrt_dir}/lrt_h20_{h}_aot_{aod}_alb_0_alt_{round(altitude_km*1000)}_sureglo.out', delim_whitespace=True, header=None)
                df_t.columns = ['Wavelength', 'eglo']

                df_s1 = pd.read_csv(f'{lrt_dir}/lrt_h20_{h}_aot_{aod}_alb_0.15_alt_{round(altitude_km*1000)}_sureglo.out', delim_whitespace=True, header=None)
                df_s1.columns = ['Wavelength', 'eglo']

                df_s2 = pd.read_csv(f'{lrt_dir}/lrt_h20_{h}_aot_{aod}_alb_0.5_alt_{round(altitude_km*1000)}_sureglo.out', delim_whitespace=True, header=None)
                df_s2.columns = ['Wavelength', 'eglo']

                df_irr = pd.read_csv(f'{lrt_dir}/lrt_h20_{h}_aot_{aod}_alt_{round(altitude_km*1000)}_IRRAD.out', delim_whitespace=True, header=None)
                df_irr.columns = ['Wavelength', 'edir', 'edn']

                # Fit spline to match  for L_0 (path radiance)
                fun_r = interpolate.interp1d(df_r['Wavelength'], df_r['uu'], kind='slinear')
                l0 = fun_r(sensor_wavelengths)

                # Compute t_up (upward transmittance)
                fun_t = interpolate.interp1d(df_t['Wavelength'], df_t['eglo'], kind='slinear')
                t_up = fun_t(sensor_wavelengths)   

                # Compute S (atmos sphere albedo)
                df_s2['sph_alb'] = (df_s2['eglo'] - df_s1['eglo']) / (0.5 * df_s2['eglo'] -  0.15 * df_s1['eglo'])
                fun_s = interpolate.interp1d(df_s2['Wavelength'], df_s2['sph_alb'], kind='slinear')
                s = fun_s(sensor_wavelengths)

                # Fit spline to match edir and edn
                f_dir = interpolate.interp1d(df_irr['Wavelength'], df_irr['edir'], kind='slinear')
                edir = f_dir(sensor_wavelengths)

                f_edn = interpolate.interp1d(df_irr['Wavelength'], df_irr['edn'], kind='slinear')
                edn = f_edn(sensor_wavelengths)

                # append the results
                l0_arr[i,j,k,:] = l0
                t_up_arr[i,j,k,:] = t_up
                s_arr[i,j,k,:] = s
                edir_arr[i,j,k,:] = edir
                edn_arr[i,j,k,:] = edn
    
    # Now prep for new grid
    Z = np.copy(sensor_wavelengths)
    W = np.array(h20_range)
    X = np.array(a550_range)
    Y = np.array(alt_range)

    # Create grid functions
    g_l0 = RegularGridInterpolator((W, X, Y, Z), l0_arr, method='linear')
    g_tup = RegularGridInterpolator((W, X, Y, Z), t_up_arr, method='linear')
    g_s = RegularGridInterpolator((W, X, Y, Z), s_arr, method='linear')
    g_edir = RegularGridInterpolator((W, X, Y, Z), edir_arr, method='linear')
    g_edn = RegularGridInterpolator((W, X, Y, Z), edn_arr, method='linear')


    return  g_l0, g_tup, g_s, g_edir, g_edn





def lrt_reader(h, aod, alt, cosi, sza, shadow,svf, slope, rho_surface,
               g_l0, g_tup, g_s, g_edir, g_edn,sensor_wavelengths):
    
    '''
    TODO

    '''
    # Ensure optimization stays in bounds
    if h <= 1:
        h=1
    if aod <= 0.01:
        aod=0.01

    # Setup arrays
    Z = np.copy(sensor_wavelengths)
    W = np.array(h)
    X = np.array(aod)
    Y = np.array(alt)

    # GRID INTERPS
    l0 = g_l0((W, X, Y, Z))
    t_up = g_tup((W, X, Y, Z))
    s = g_s((W, X, Y, Z))
    edir0 = g_edir((W, X, Y, Z))
    edn0 = g_edn((W, X, Y, Z))

    # Correct to local conditions
    #############################  
    t_up = t_up / np.cos(np.radians(sza)) 

    # Adjust local Fdir and Fdiff
    edir =  edir0 * cosi  * shadow #shadow: 0=shadow, 1=sun
    edn =  edn0  * svf
    
    # Combine diffuse and direct into S_total
    s_total = edir + edn

    # Add in adjacent pixels estimate (terrain influence)
    # https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020JD034294
    ct = max(0,((1 + np.cos(np.radians(slope))) / 2 ) - svf)
    s_total = s_total + ((edir0+edn0)*rho_surface * ct)

    # Correct units to be microW/cm2/nm/sr
    s_total = s_total / 10
    l0 = l0 / 10


    return l0, t_up, s, s_total



