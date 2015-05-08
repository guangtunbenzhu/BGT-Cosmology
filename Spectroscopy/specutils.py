
""" 
Useful small tools for spectroscopic analysis
"""

# Python 3 vs. 2
from __future__ import print_function, division

# Standard Library Modules
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
from scipy import constants

# Third Party Modules

# Your Own Modules


#####################
# Code starts here  #
#####################

# Create a center wavelength grid with constant width in log (i.e., velocity) space:
# Input is in Angstrom, output is log10(lambda/Angstrom)
def get_loglam(minwave=448., maxwave=10402., dloglam=1.E-4, pivot=None):
    """Return a central wavelength grid uniform in velocity space
    """
    if minwave>maxwave:
        raise ValueError("Your maximum wavelength is smaller than the minimum wavelength.")
    if not pivot: 
        return np.arange(np.log10(minwave), np.log10(maxwave), dloglam)+0.5*dloglam
    else:
        # assert pivot.size == 1, "I can only handle one pivotal wavelength"
        log10_pivot_minwave = np.log10(minwave)+(np.log10(pivot)-np.log10(minwave))%dloglam \
                              -0.5*dloglam
        return np.arange(log10_pivot_minwave, np.log10(maxwave), dloglam)+0.5*dloglam

def get_velgrid(dloglam=1.E-4, dvel=None, noffset=100):
    """
    \pm noffset*dvel km/s
    """
    outvel = np.arange(2*noffset)
    if not dvel:
       dvel = np.log(10.)*dloglam*constants.c/1E3
    return (np.arange(2*noffset)-noffset)*dvel

def resample_spec(inloglam, influx, inivar, newloglam):
    """
    resample a spectrum/many spectra: convolution then interpolation?
    given f(lambda)dlambda
    """
    pass

# Need to vectorize this
def interpol_spec(inloglam, influx, inivar, newloglam):
    """interpolate the spectra onto a desired wavelength grid
    a simpler, faster and not worse version of combine1fiber
    works if the sampling of newloglam is similar to that of inloglam
    to resample a spectrum, use resample_spec which does convolution properly
    fine-tuned for logarithmic binning 
    hardcoded parameters: binsize, maxsep; 1 pixel around ivar==0
    to do: select bkptbin, maxsep
    inloglam: sorted
    newloglam: sorted

    Only works for one spectrum for now.
    """

    # Quickcheck of the inputs
    # assert inloglam.size == influx.size
    # assert inloglam.size == inivar.size
    # assert inloglam.ndim == 1

    if (inloglam.size != influx.size) or (inloglam.size != inivar.size):
       raise ValueError("The shapes of inputs don't match")

    if (inloglam.ndim != 1):
       raise ValueError("I can only take one spectrum for now.")

    # Initialization
    newflux = np.zeros(newloglam.size)
    newivar = np.zeros(newloglam.size)

    if inivar[inivar>0].size<5:
       print("input spectrum invalid, no interpolation is performed.")
       return (newflux, newivar)

    # choosing break point binning
    # in_inbetween_out = (np.where(np.logical_and(inloglam>=np.min(newloglam), inloglam<=np.max(newloglam))))[0]
    # this shouldn't matter if binning is in logarithmic scale
    # binsize = np.median(inloglam[in_inbetween_out[1:]]-inloglam[in_inbetween_out[:-1]]) 
    # bkptbin = 1.2*binsize
    # Break the inputs into groups based on maxsep
    # maxsep = 2.0*binsize

    # Check boundary
    inbetween = (np.where(np.logical_and(newloglam>np.amin(inloglam), newloglam<np.amax(inloglam))))[0]
    if inbetween.size == 0:
       print("newloglam not in range, no interpolation is necessary.")
       return (newflux, newivar)
    
    # print(inbetween[0],inbetween[1])

    # Spline
    # s is determined by difference between input and output wavelength, minimum==2
    # desired_nknots = np.ceil(np.median(newloglam[1:]-newloglam[:-1])/np.median(inloglam[1:]-inloglam[:-1]))
    #if desired_nknots<3: # No smoothing in this case
    #   s = 0
    #else: 
    #   s = int(np.floor(inivar.size/desired_nknots))
    #s = 1000
    #print(s)

    # Smoothing does not working properly, forced interpolation
    #s = 0

    # See combine1fiber
    # 1. Break the inputs into groups based on maxsep
    # 2. Choose knots ourselves

    # try except needed here
    # Let's choose the knots ourselves
    # f = LSQUnivariateSpline(inloglam, influx, knots, w=inivar)

    # No smoothing
    #f = UnivariateSpline(inloglam[inivar>0], influx[inivar>0], w=inivar[inivar>0], k=3, s=0)
    tmploglam = inloglam[inivar>0]
    tmpflux = influx[inivar>0]
    tmpivar = inivar[inivar>0]
    f = UnivariateSpline(tmploglam, tmpflux, w=tmpivar, k=3, s=0)
    newflux[inbetween] = f(newloglam[inbetween])

    # Linear
    # print(inloglam.shape)
    # print(inivar.shape)
    g = interp1d(inloglam, inivar, kind='linear')
    newivar[inbetween] = g(newloglam[inbetween])
    #  

    # set newivar=0 where inivar=0
    izero_inivar = (np.where(inivar==0))[0]
    if izero_inivar.size>0:
       # find those pixels that use inivar==0
       index = np.searchsorted(inloglam, newloglam[inbetween])
       newivar[inbetween[np.in1d(index, izero_inivar)]] = 0.
       newivar[inbetween[np.in1d(index-1, izero_inivar)]] = 0.

    newflux[newivar==0] = 0.

    return (newflux, newivar)

def airtovac(air0):
    air = np.ravel(np.array(air0))
    vac = np.zeros(air.shape)
    vac[:] = air[:]
    ig = (np.where(air>=2000.))[0]
    if ig.size>0:
       sigma2 = 1E8/np.power(vac[ig],2)
       factor = 1E0+5.792105E-2/(238.0185E0 - sigma2)+1.67917E-3/(57.362E0 - sigma2)
       vac[ig] = air[ig]*factor
    return vac

def vactoair(vac):
    return vac/(1.0+2.735182E-4+131.4182/np.power(vac,2) + 2.76249E8/np.power(vac,4))

