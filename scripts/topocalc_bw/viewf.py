import numpy as np
import multiprocessing as mp

from topocalc_bw.gradient import gradient_d8
from topocalc_bw.horizon import horizon

#    **************************
#    ***UPDATE JULY 17, 2024***
#    ***  by Brent Wilder   ***
#
#    Now includes the self-slope obscuring the horizon. 
#    
#    Also brings in multiprocessing/parallel computing.
#
#    These modifications were presented by Jeff Dozier, 2022 IEEE paper.
#
#    ***************************
#    ***************************


# Define wrapper function
def viewf_mp(args):
    return viewf(*args)



# run in parallel
def run_viewf_pool(n_cpu,dem, spacing, n_angles=72):

    # prep the data for correct format for computation
    angles, aspect, cos_slope, sin_slope, tan_slope = viewf_prep(dem=dem, 
                                                                 spacing=spacing, 
                                                                 nangles=n_angles, 
                                                                 sin_slope=None, 
                                                                 aspect=None)
    
    # create list_args
    list_args = []
    for a in angles:
        list_args.append([a, dem, spacing, aspect, cos_slope, sin_slope, tan_slope])
    
    # Run n-angles in parallel (n=72 by default)
    with mp.Pool(n_cpu) as pool:
        results = pool.map(viewf_mp, list_args)
    
    # and so now, we have a list object of 72, 2-d arrays
    # and  can complete integration for svf
    svf = sum(results) / len(angles)

    return svf




def viewf_prep(dem, spacing, nangles=72, sin_slope=None, aspect=None):
    """
    Calculate the sky view factor of a dem. (preps it for multipool)

    Args:
        dem: numpy array for the DEM
        spacing: grid spacing of the DEM
        nangles: number of angles to estimate the horizon, defaults
                to 72 angles
        sin_slope: optional, will calculate if not provided
                    sin(slope) with range from 0 to 1
        aspect: optional, will calculate if not provided
                Aspect as radians from south (aspect 0 is toward
                the south) with range from -pi to pi, with negative
                values to the west and positive values to the east.

    Returns:
        angles, aspect, cos_slope, sin_slope, tan_slope

    """  

    if dem.ndim != 2:
        raise ValueError('viewf input of dem is not a 2D array')

    if nangles < 16:
        raise ValueError('viewf number of angles should be 16 or greater')

    if sin_slope is not None:
        if np.max(sin_slope) > 1:
            raise ValueError('slope must be sin(slope) with range from 0 to 1')

    # calculate the gradient if not provided
    # The slope is returned as radians so convert to sin(S)
    if sin_slope is None:
        slope, aspect = gradient_d8(
            dem, dx=spacing, dy=spacing, aspect_rad=True)
        sin_slope = np.sin(slope)
        cos_slope = np.cos(slope)
        tan_slope = np.tan(slope)

    # -180 is North
    angles = np.linspace(-180, 180, num=nangles, endpoint=False)

    # perform the integral
    cos_slope = np.sqrt((1 - sin_slope) * (1 + sin_slope))

    return angles, aspect, cos_slope, sin_slope, tan_slope



def viewf(angle, dem, spacing, aspect, cos_slope, sin_slope, tan_slope):
    '''
    See above, but this is running each horizon angle in parallel (n=72 , or user input)
    
    '''

    # horizon angles
    hcos = horizon(angle, dem, spacing)
    azimuth = np.radians(angle)
    h = np.arccos(hcos)

    # cosines of difference between horizon aspect and slope aspect
    cos_aspect = np.cos(aspect - azimuth)

    # check for slope being obscured
    # EQ 3 in Dozier et al. 2022
    #     H(t) = min(H(t), acos(sqrt(1-1./(1+tand(slopeDegrees)^2*cos(azmRadian(t)-aspectRadian).^2))));
    t = cos_aspect<0
    h[t] = np.fmin(h[t], 
                        np.arccos(np.sqrt(1 - 1/(1 + cos_aspect[t]**2 * tan_slope[t]**2))))

    # integral in Dozier 2022
    # qIntegrand = (cosd(slopeDegrees)*sin(H).^2 + sind(slopeDegrees)*cos(aspectRadian-azmRadian).*(H-cos(H).*sin(H)))/2
    svf = (cos_slope * np.sin(h)**2 + sin_slope*cos_aspect * (h - np.sin(h)*np.cos(h)))

    svf[svf<0] = 0

    return svf
