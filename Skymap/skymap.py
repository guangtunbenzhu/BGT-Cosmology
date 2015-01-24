
""" 
Useful small tools for spectroscopic analysis
"""

import numpy as np
import datapath
import os.path
from skymap_utils import wcs_getval

def dust_getmaps(maps):
    path = datapath.dustmap_path('dust')
    files = {'EBV': ['SFD_dust_4096_ngp.fits', 'SFD_dust_4096_sgp.fits'],
            'MASK': ['SFD_mask_4096_ngp.fits', 'SFD_mask_4096_sgp.fits'],
               'T': ['SFD_temp_4096_ngp.fits', 'SFD_temp_4096_sgp.fits'],
               'X': ['SFD_xmap_4096_ngp.fits', 'SFD_xmap_4096_sgp.fits'],
            }.get(maps,['SFD_dust_4096_ngp.fits', 'SFD_dust_4096_sgp.fits']) 

    fullnames = []
    for thisfile in files: fullnames.append(os.path.join(path,thisfile))

    return fullnames

def dust_getval(longitude, latitude, maps='EBV', interp=True, noloop=True, verbose=True):
    '''
    Extract values from the SFD dust map and apply 14% correction based on Schlafly & Finkbeiner 2011.
    '''

    infiles = dust_getmaps(maps.upper())
    return wcs_getval(longitude, latitude, infiles, interp=interp, noloop=noloop, verbose=verbose, 
                      transpose=True)/0.86
