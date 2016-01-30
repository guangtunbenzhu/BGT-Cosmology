
""" 

"""

from __future__ import print_function, division

import numpy as np
import fitsio
import datapath
from os.path import isfile, join
from specutils import get_loglam


# allinone utils
def allinone_wave_filename():
    path = datapath.allinone_path()
    filename = 'AIO_CommonWave.fits'
    return join(path, filename)


def allinone_wave_readin():
    infile = allinone_wave_filename()
    if not isfile(infile):
       raise IOError("{0} not found.")
       # print ("{0} does not exist! \n" 
       #       "Check the path or call allinone_wave_make() first.".format(infile))
       # return -1
    return fitsio.read(infile)


def allinone_wave_make(overwrite=False):
    outfile = allinone_wave_filename()
    if (isfile(outfile) and not overwrite):
       print("File {0} exists. Use overwrite to overwrite it.".format(outfile))
       return -1
    master_loglam = get_loglam(minwave=448., maxwave=10402., dloglam=1.E-4)
    master_wave = np.power(10., master_loglam)
    # preferred structured array
    outstr = np.zeros(1, dtype=[('WAVE','f8', (master_loglam.size,))])
    # Using outstr[0]['WAVE'] does not work, only the first element got the value, why? Memory layout conflict?
    # Or shape conflict (1, n) vs. (n,)
    outstr[0]['WAVE'][:] = np.power(10., master_loglam)
    fits = fitsio.FITS(outfile, 'rw', clobber=True)
    fits.write(outstr)
    fits.close()


def allinone_wavebase(band):
    wavebase = {'EUV': [450.,900.],
                'FUV': [900.,1800.],
                'NUV': [1800.,3600.],
                'OPTICAL': [3600.,7200.],
                'NIR': [7200.,10400.]
               }.get(band)
    return wavebase


def allinone_filebase(band):
    filebase = {'EUV': 'Wave00450_00900A',
                'FUV': 'Wave00900_01800A',
                'NUV': 'Wave01800_03600A',
                'OPTICAL': 'Wave03600_07200A',
                'NIR': 'Wave07200_10400A'
               }.get(band)
    return filebase


def allinone_filename(band, prefix=None):
    path = datapath.allinone_path()
    filebase = allinone_filebase(band)
    try: 
        filename = prefix+filebase+'.fits'
    except TypeError:
        filename = filebase+'.fit'
    return join(path, filename)

