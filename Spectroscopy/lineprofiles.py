
""" 
Useful small tools for spectroscopic analysis
"""

from __future__ import print_function, division
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d

# Multiple Gaussians with identical width
# As this is for fitting purpose, I don't know how to generalize this efficiently yet
# Maybe use a wrapper over all these custom-made functions
# single, double, triple, quadruple, quintuple, sextuple, septuple, octuple, noncuple, and decuple
def double_gaussian_fixwidth(x, center, width, 
                             amplitude1,  amplitude2, 
                             separation2):
    """ double gaussian with identical width
    amplitutde1: of the first Gaussians
    amplitutde2: of the second Gaussians
    center: of the first Gaussian
    separation2: between the second and the first Gaussian
    width: identical for two Gaussians
    """
    return (amplitude1/np.sqrt(2.*np.pi)/width*np.exp(-((x-center)**2)/(2*width**2))
            +amplitude2/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation2))**2)/(2*width**2)))

def triple_gaussian_fixwidth(x, center, width, 
                             amplitude1,  amplitude2,  amplitude3, 
                             separation2, separation3):
    """ tribple gaussian with identical width
    amplitutde[1-3]: of the Gaussians
    center: of the first Gaussian
    separation[2-3]: between the second/third and the first Gaussian
    width: identical for all Gaussians
    """
    return (amplitude1/np.sqrt(2.*np.pi)/width*np.exp(-((x-center)**2)/(2*width**2))
           +amplitude2/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation2))**2)/(2*width**2))
           +amplitude3/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation3))**2)/(2*width**2)))

def quadruple_gaussian_fixwidth(x, center, width, 
                                amplitude1,  amplitude2,  amplitude3, amplitude4, 
                                separation2, separation3, separation4):
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

def quintuple_gaussian_fixwidth(x, center, width, 
                                amplitude1,  amplitude2,  amplitude3,  amplitude4,  amplitude5, 
                                separation2, separation3, separation4, separation5):
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
           +amplitude5/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation5))**2)/(2*width**2)))

def sextuple_gaussian_fixwidth(x, center, width, 
                               amplitude1,  amplitude2,  amplitude3,  amplitude4,   amplitude5,  amplitude6, 
                               separation2, separation3, separation4, separation5,  separation6):
    """ sextuple gaussian with identical width
    amplitutde[1-6]: of the Gaussians
    center: of the first Gaussian
    separation[2-6]: between the second/third/fourth and the first Gaussian
    width: identical for all Gaussians
    """
    return (amplitude1/np.sqrt(2.*np.pi)/width*np.exp(-((x-center)**2)/(2*width**2))
           +amplitude2/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation2))**2)/(2*width**2))
           +amplitude3/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation3))**2)/(2*width**2)) 
           +amplitude4/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation4))**2)/(2*width**2))
           +amplitude5/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation5))**2)/(2*width**2))
           +amplitude6/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation6))**2)/(2*width**2)))

def septuple_gaussian_fixwidth(x, center, width, 
                               amplitude1,  amplitude2,  amplitude3,  amplitude4,   amplitude5,  amplitude6,  amplitutde7, 
                               separation2, separation3, separation4, separation5,  separation6, separation7): 
    """ septuple gaussian with identical width
    amplitutde[1-7]: of the Gaussians
    center: of the first Gaussian
    separation[2-7]: between the second/third/fourth and the first Gaussian
    width: identical for all Gaussians
    """
    return (amplitude1/np.sqrt(2.*np.pi)/width*np.exp(-((x-center)**2)/(2*width**2))
           +amplitude2/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation2))**2)/(2*width**2))
           +amplitude3/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation3))**2)/(2*width**2)) 
           +amplitude4/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation4))**2)/(2*width**2))
           +amplitude5/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation5))**2)/(2*width**2))
           +amplitude6/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation6))**2)/(2*width**2))
           +amplitude7/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation7))**2)/(2*width**2)))

def octuple_gaussian_fixwidth(x, center, width, 
                              amplitude1,  amplitude2,  amplitude3,  amplitude4,   amplitude5,  amplitude6,  amplitude7,  amplitutde8, 
                              separation2, separation3, separation4, separation5, separation6,  separation7, separation8):
    """ octuple gaussian with identical width
    amplitutde[1-8]: of the Gaussians
    center: of the first Gaussian
    separation[2-8]: between the second/third/fourth and the first Gaussian
    width: identical for all Gaussians
    """
    return (amplitude1/np.sqrt(2.*np.pi)/width*np.exp(-((x-center)**2)/(2*width**2))
           +amplitude2/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation2))**2)/(2*width**2))
           +amplitude3/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation3))**2)/(2*width**2)) 
           +amplitude4/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation4))**2)/(2*width**2))
           +amplitude5/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation5))**2)/(2*width**2))
           +amplitude6/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation6))**2)/(2*width**2))
           +amplitude7/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation7))**2)/(2*width**2))
           +amplitude8/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation8))**2)/(2*width**2)))

def nonuple_gaussian_fixwidth(x, center, width,
                              amplitude1,  amplitude2,  amplitude3,  amplitude4,   amplitude5,  amplitude6,  amplitude7,  amplitutde8,  amplitude9,
                              separation2, separation3, separation4, separation5, separation6,  separation7, separation8, separation9): 
    """ nonuple gaussian with identical width
    amplitutde[1-9]: of the Gaussians
    center: of the first Gaussian
    separation[2-9]: between the second/third/fourth and the first Gaussian
    width: identical for all Gaussians
    """
    return (amplitude1/np.sqrt(2.*np.pi)/width*np.exp(-((x-center)**2)/(2*width**2))
           +amplitude2/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation2))**2)/(2*width**2))
           +amplitude3/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation3))**2)/(2*width**2)) 
           +amplitude4/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation4))**2)/(2*width**2))
           +amplitude5/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation5))**2)/(2*width**2))
           +amplitude6/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation6))**2)/(2*width**2))
           +amplitude7/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation7))**2)/(2*width**2))
           +amplitude8/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation8))**2)/(2*width**2))
           +amplitude9/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation9))**2)/(2*width**2)))

def decuple_gaussian_fixwidth(x, center, width,
                              amplitude1,  amplitude2,  amplitude3,  amplitude4,   amplitude5,  amplitude6,  amplitude7,  amplitutde8,  amplitude9, amplitude10,
                              separation2, separation3, separation4, separation5, separation6,  separation7, separation8, separation9,  separation10):
    """ nonuple gaussian with identical width
    amplitutde[1-10]: of the Gaussians
    center: of the first Gaussian
    separation[2-10]: between the second/third/fourth and the first Gaussian
    width: identical for all Gaussians
    """
    return (amplitude1/np.sqrt(2.*np.pi)/width*np.exp(-((x-center)**2)/(2*width**2))
           +amplitude2/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation2))**2)/(2*width**2))
           +amplitude3/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation3))**2)/(2*width**2)) 
           +amplitude4/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation4))**2)/(2*width**2))
           +amplitude5/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation5))**2)/(2*width**2))
           +amplitude6/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation6))**2)/(2*width**2))
           +amplitude7/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation7))**2)/(2*width**2))
           +amplitude8/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation8))**2)/(2*width**2))
           +amplitude9/np.sqrt(2.*np.pi)/width*np.exp(-((x-(center+separation9))**2)/(2*width**2)))


# Multiple Gaussian
# Brute-force implementation for performance
#_tuple = ['double', 'triple', 'quadruple', 'quintuple', 'sextuple', 'septuple', 'octuple', 'nonuple', 'decuple']
_tuple_fixwidth = {'double': double_gaussian_fixwidth,       'triple': triple_gaussian_fixwidth, 
                   'quadruple': quadruple_gaussian_fixwidth, 'quintuple': quintuple_gaussian_fixwidth, 
                   'sextuple': sextuple_gaussian_fixwidth,   'septuple': septuple_gaussian_fixwidth, 
                   'octuple': octuple_gaussian_fixwidth,     'nonuple': nonuple_gaussian_fixwidth, 
                   'decuple': decuple_gaussian_fixwidth} 
#_tuple_fixseparation = {'double': double_gaussian_fixseparation,       'triple': triple_gaussian_fixseparation, 
#                        'quadruple': quadruple_gaussian_fixseparation, 'quintuple': quintuple_gaussian_fixseparation, 
#                        'sextuple': sextuple_gaussian_fixseparation,   'septuple': septuple_gaussian_fixseparation, 
#                        'octuple': octuple_gaussian_fixseparation,     'nonuple': nonuple_gaussian_fixseparation, 
#                        'decuple': decuple_gaussian_fixseparation} 
#_tuple_fix = {'double': double_gaussian_fix,       'triple': triple_gaussian_fix, 
#              'quadruple': quadruple_gaussian_fix, 'quintuple': quintuple_gaussian_fix, 
#              'sextuple': sextuple_gaussian_fix,   'septuple': septuple_gaussian_fix, 
#              'octuple': octuple_gaussian_fix,     'nonuple': nonuple_gaussian_fix, 
#              'decuple': decuple_gaussian_fix} 

def Multiple_Gaussian_Fix(tupleorder='double'):
    return _tuple_fixwidth[tupleorder]

#def Multiple_Gaussian(tupleorder='double', fixseparation=False, fixwidth=True):
#    if fixwidth: outfunc = _tuple_fixwidth[tupleorder]
#    #if fixseparation: outfunc = _tuple_fixseparation[tupleorder]
#    #if (fixseparation and fixwidth): outfunc = _tuple_fixseparation_fixwidth[tupleorder]
#    return outfunc



