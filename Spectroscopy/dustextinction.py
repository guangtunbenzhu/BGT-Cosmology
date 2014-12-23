
""" 
Useful small tools for spectroscopic analysis
"""

import numpy as np
from numpy.polynomial.polynomial import polyval
from scipy import interpolate
import warnings

# Create a center wavelength grid with constant width in log (i.e., velocity) space:
# Input is in Angstrom, output is log10(lambda/Angstrom)
def get_loglam(minwave=448., maxwave=10402., dloglam=1.E-4):
    """Return a central wavelength grid uniform in velocity space
    """
    if maxwave>minwave:
       raise ValueError("Your maximum wavelength is smaller than the minimum wavelength.")
    return np.arange(np.log10(minwave), np.log10(maxwave), dloglam)+0.5*dloglam

#get_loglambda = lambda minwave=400., maxwave=12800., dloglam=5.E-5: np.arange(np.log10(minwave), np.log10(maxwave)+dloglam, dloglam)

def dust_gordon(wave, RV=3.41):
    pass

def dust_calzetti(wave, RV=3.1):
    pass

def dust_ccm(wave, RV=3.1, ODONNELL=False):
    """
    Cardelli, Clayton, Mathis (1989)
    Extinction Curve k_lambda = A_lambda/E(B-V) for star-forming galaxies
    """
    
    
    x = 10000./np.ravel(wave)
    a = np.zeros(x.size, dtype=np.float64)
    b = np.zeros(x.size, dtype=np.float64)

    # Infrared
    i_IR = np.logical_and(x>0.3, x<1.1)
    if (np.count_nonzero(i_IR)>0):
       a[i_IR] = 0.574*pow(x[i_IR], 1.61)
       b[i_IR] = -0.527*pow(x[i_IR], 1.61)

    # Optical/Near-infrared
    i_OP = np.logical_and(x>=1.1, x<3.3)
    if (np.count_nonzero(i_OP)>0):
       if ODONNELL:
          c1 = np.array([1., 0.104, -0.609, 0.701, 1.137, -1.718, -0.827, 1.647, -0.505])
          c2= np.array([0., 1.952, 2.908, -3.989, -7.985, 11.102, 5.491, -10.805, 3.347])
       else:
          c1 = np.array([1., 0.17699, -0.50447, -0.02427, 0.72085, 0.01979, -0.77530, 0.32999])
          c2= np.array([0., 1.41338, 2.28305, 1.07233, -5.38434, -0.62251, 5.30260, -2.09002])
       a[i_OP] = polyval(x[i_OP]-1.82, c1)
       b[i_OP] = polyval(x[i_OP]-1.82, c2)

#      c1 = np.array([1., 0.104, -0.609, 0.701, 1.137, -1.718, -0.827, 1.647, -0.505])
#      c2= np.array([0., 1.952, 2.908, -3.989, -7.985, 11.102, 5.491, -10.805, 3.347])

    # Mid-UV
    i_NUV = np.logical_and(x>=3.3, x<8.)
    if (np.count_nonzero(i_NUV)):
       Fa = -0.04473*pow(x[i_NUV]-5.9, 2)-0.009779*pow(x[i_NUV]-5.9, 3)
       Fb = 0.2130*pow(x[i_NUV]-5.9, 2)+0.1207*pow(x[i_NUV]-5.9, 3)

       a[i_NUV] = 1.752-0.316*x[i_NUV]-0.104/(pow(x[i_NUV]-4.67,2)+0.341)+np.where(x[i_NUV]>5.9, Fa, 0)
       b[i_NUV] = -3.090+1.825*x[i_NUV]+1.206/(pow(x[i_NUV]-4.62,2)+0.263)+np.where(x[i_NUV]>5.9, Fb, 0)

    # Far-UV
    i_FUV = np.logical_and(x>=8., x<=11.)
    if (np.count_nonzero(i_FUV)>0):
       c1 = np.array([-1.073, -0.628, 0.137, -0.070])
       c2 = np.array([13.670, 4.257, -0.420, 0.374])

       a[i_FUV] = polyval(x[i_FUV]-8., c1)
       b[i_FUV] = polyval(x[i_FUV]-8., c2)

    # Out of range
    i_NO = np.logical_or(x>11., x<0.3)
    if (np.count_nonzero(i_NO)>0):
       warnings.warn('Some wavelengths are out of range. Return 0 for them.')

    return RV*(a+b/RV)

def dust_fitz(wave, RV=3.1, kind='MW'):
    """
    Fitzpatrick (1999)
    Extinction Curve k_lambda = A_lambda/E(B-V) for Milky Way 
    recommended by Schlafly & Finkbeiner 2011
    """

    x = 1E4/np.ravel(wave)
    k_lambda = np.zeros(x.size, dtype=np.float64)

    # Set default values of c1, c2, c3, c4, gamma, and x0 parameters
    if kind.upper() == 'MW': 
       x0 = 4.596
       gamma = 0.99
       c4 = 0.41
       c3 = 3.23
       c2 = -0.824+4.717/RV
       c1 = 2.030-3.007*c2
    elif kind.upper() == 'LMC2': 
       x0 = 4.626
       gamma = 1.05
       c4 = 0.42
       c3 = 1.92
       c2 = 1.31
       c1 = -2.16
    elif kind.upper() == 'AVGLMC':
       x0 = 4.596
       gamma = 0.91
       c4 = 0.64
       c3 = 2.73
       c2 = 1.11
       c1 = -1.28
    else:
       raise ValueError("Sorry! Cannot find the right extinction curver.")

    xcutuv = 1E4/2700.0
    x_spl_UV = 1E4/np.array([2700.0, 2600.0])
    i_UV = x>=xcutuv
    # No need to check if i_UV is all False
    xuv = np.append(x_spl_UV, x[i_UV])
    yuv = c1+c2*xuv+c3*pow(xuv,2)/(pow(pow(xuv,2)-pow(x0,2), 2)+pow(xuv*gamma,2)) \
        + c4*(0.5392*pow(np.maximum(xuv, 5.9)-5.9, 2)+0.05644*pow(np.maximum(xuv, 5.9)-5.9,3)) \
        + RV
    y_spl_UV = yuv[0:2]
    if (np.count_nonzero(i_UV)>0):
       k_lambda[i_UV] = yuv[2:]

    # Cubic spline anchor points in the optical and infrared
    x_spl_OPIR = np.append([0.], 1E4/np.ravel([26500.0, 12200.0, 6000., 5470., 4670., 4110.0]))
    y_spl_IR = np.array([0., 0.26469, 0.82925])*RV/3.1

    poly_OP_c1 = np.array([-4.22809E-1, 1.00270, 2.13572E-4])
    poly_OP_c2 = np.array([-5.1354E-2, 1.00216, -7.35778E-5])
    poly_OP_c3 = np.array([7.00127E-1, 1.00184, -3.32598E-5])
    poly_OP_c4 = np.array([1.19456, 1.01707, -5.46959E-3, 7.97809E-4, -4.45636E-5])
    y_spl_OP = np.ravel([polyval(RV, poly_OP_c1), polyval(RV, poly_OP_c2), polyval(RV, poly_OP_c3), polyval(RV, poly_OP_c4)])

    y_spl = np.append(np.append(y_spl_IR, y_spl_OP), y_spl_UV)
    x_spl = np.append(x_spl_OPIR, x_spl_UV)

    i_OPIR = x<xcutuv
    if (np.count_nonzero(i_OPIR)>0):
       f_k = interpolate.InterpolatedUnivariateSpline(x_spl, y_spl, k=3)
       k_lambda[i_OPIR] = f_k(x[i_OPIR])

    return k_lambda
