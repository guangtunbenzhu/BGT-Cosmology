
""" 
Useful small tools for spectroscopic analysis
"""

from __future__ import print_function, division
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d

# Create a center wavelength grid with constant width in log (i.e., velocity) space:
# Input is in Angstrom, output is log10(lambda/Angstrom)
def get_loglam(minwave=448., maxwave=10402., dloglam=1.E-4):
    """Return a central wavelength grid uniform in velocity space
    """
    if minwave>maxwave:
       raise ValueError("Your maximum wavelength is smaller than the minimum wavelength.")
    return np.arange(np.log10(minwave), np.log10(maxwave), dloglam)+0.5*dloglam

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

# Multiple Gaussians with identical width
# As this is for fitting purpose, I don't know how to generalize this efficiently yet
# Maybe use a wrapper over all these custom-made functions
# single, double, triple, quadruple, quintuple, sextuple, septuple, octuple, noncuple, and decuple
def double_gaussian_onesigma(x, amplitude1,  amplitude2, center, 
                                separation2, width):
    """ double gaussian with identical width
    amplitutde1: of the first Gaussians
    amplitutde2: of the second Gaussians
    center: of the first Gaussian
    separation2: between the second and the first Gaussian
    width: identical for two Gaussians
    """
    return (amplitude1/np.sqrt(2.*np.pi)/width*np.exp(-((x-center)**2)/(2*width**2))
            +amplitude2/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation))**2)/(2*width**2)))

def triple_gaussian_onesigma(x, amplitude1,  amplitude2,  amplitude3, center, 
                                separation2, separation3, width):
    """ tribple gaussian with identical width
    amplitutde[1-3]: of the Gaussians
    center: of the first Gaussian
    separation[2-3]: between the second/third and the first Gaussian
    width: identical for all Gaussians
    """
    return (amplitude1/np.sqrt(2.*np.pi)/width*np.exp(-((x-center)**2)/(2*width**2))
           +amplitude2/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation2))**2)/(2*width**2))
           +amplitude3/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation3))**2)/(2*width**2)))

def quadruple_gaussian_onesigma(x, amplitude1,  amplitude2,  amplitude3, amplitude4, center, 
                                   separation2, separation3, separation4, width):
    """ quadruple gaussian with identical width
    amplitutde[1-4]: of the Gaussians
    center: of the first Gaussian
    separation[2-4]: between the second/third/fourth and the first Gaussian
    width: identical for all Gaussians
    """
    return (amplitude1/np.sqrt(2.*np.pi)/width*np.exp(-((x-center)**2)/(2*width**2))
           +amplitude2/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation2))**2)/(2*width**2))
           +amplitude3/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation3))**2)/(2*width**2)) 
           +amplitude4/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation4))**2)/(2*width**2)))

def quintuple_gaussian_onesigma(x, amplitude1,  amplitude2,  amplitude3,  amplitude4,  amplitude5, center, 
                                   separation2, separation3, separation4, separation5, width):
    """ quintuple gaussian with identical width
    amplitutde[1-5]: of the Gaussians
    center: of the first Gaussian
    separation[2-5]: between the second/third/fourth and the first Gaussian
    width: identical for all Gaussians
    """
    return (amplitude1/np.sqrt(2.*np.pi)/width*np.exp(-((x-center)**2)/(2*width**2))
           +amplitude2/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation2))**2)/(2*width**2))
           +amplitude3/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation3))**2)/(2*width**2)) 
           +amplitude4/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation4))**2)/(2*width**2))
           +amplitude5/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation5))**2)/(2*width**2))

def sextuple_gaussian_onesigma(x, amplitude1,  amplitude2,  amplitude3,  amplitude4,   amplitude5,  amplitude6, center, 
                                   separation2, separation3, separation4, separation5, separation6, width):
    """ quintuple gaussian with identical width
    amplitutde[1-5]: of the Gaussians
    center: of the first Gaussian
    separation[2-5]: between the second/third/fourth and the first Gaussian
    width: identical for all Gaussians
    """
    return (amplitude1/np.sqrt(2.*np.pi)/width*np.exp(-((x-center)**2)/(2*width**2))
           +amplitude2/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation2))**2)/(2*width**2))
           +amplitude3/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation3))**2)/(2*width**2)) 
           +amplitude4/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation4))**2)/(2*width**2))
           +amplitude5/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation5))**2)/(2*width**2))

