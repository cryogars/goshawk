from sister.sensors import prisma

base = '/Users/brent/Documents/Albedo/PRISMA/20221214_MAMMOTH'

zipname = 'PRS_L1_STD_OFFL_20221214184840_20221214184845_0001.zip'

l1_zip = f'{base}/{zipname}'

out_dir = f'{base}/rad'

temp_dir = f'{base}/temp'

elev_dir='https://copernicus-dem-30m.s3.amazonaws.com/'

s2 = f'{base}/s2/sentinel'

shift = '/Users/brent/Code/sister/data/prisma/PRISMA_Mali1_wavelength_shift_surface_smooth.npz'

rad_coeff = '/Users/brent/Code/sister/data/prisma/PRS_Mali1_radcoeff_surface.npz'


prisma.he5_to_envi(l1_zip, out_dir, temp_dir, elev_dir, shift = shift, rad_coeff = rad_coeff,
                match=s2, proj = True, res = 30)