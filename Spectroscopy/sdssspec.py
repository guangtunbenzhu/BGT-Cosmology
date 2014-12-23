from __future__ import division, print_function
import os
import numpy as np
import fitsio
import datapath
from glob import glob


_spplate_hdunames = ('flux','invvar','andmask','ormask','disp','plugmap','sky', 'loglam', )

def spplate_filename(platein, mjdin, path):
    """Name for a spPlate file
    """
    plate = np.ravel(platein)
    mjd = np.ravel(mjdin)
    if plate.size != 1 or mjd.size !=1:
       raise ValueError("I can only take one plate and one mjd.")

    return os.path.join(path,"spPlate-{0:4d}-{1:05d}.fits".format(int(plate), int(mjd)))

# http://data.sdss3.org/datamodel/files/SPECTRO_REDUX/RUN2D/PLATE4/spPlate.html
def readspec(platein, mjdin, fiberin, path, output): 
    """Read spectra from a *single* spPlate file
    a simpler, hopefully better version of readspec in idlspec2d/pydl
    requiring mjd and path so that one knows what they are doing
    output should be a tuple, e.g. ('flux', 'ivar',)
    """

    # Check desired output
    if type(output) != tuple:
       raise TypeError("output should be a tuple.")

    for thishdu in output:
        if type(thishdu) != str:
           raise TypeError("hdunames in the output should be strings.") 
        if thishdu not in _spplate_hdunames:
           raise ValueError("{0} is not in the hdu list {1}".format(thishdu, _spplate_hdunames))

    # check input
    plate = np.ravel(platein)
    mjd = np.ravel(mjdin)
    if plate.size != 1 or mjd.size !=1:
       raise ValueError("I can only take one plate and one mjd.")

    # get filename
    filename = spplate_filename(plate, mjd, path)
    if not os.path.isfile(filename):
       raise ValueError("I can't find this file {0}.".format(filename))

    spplate_fits = fitsio.FITS(filename)
    hdr = spplate_fits[0].read_header()

    fiber = np.ravel(fiberin) - 1
    if np.amin(fiber)<0 or np.amax(fiber)>=hdr['naxis2']:
       raise ValueError("Fiber ID cannot be smaller than 1 or larger than {0}.".format(hdr['naxis2']))

    # output, a dictionary
    spplate_data = dict()
    for thishdu in output:
        if thishdu == "loglam":
           c0 = hdr['coeff0']
           c1 = hdr['coeff1']
           npix = hdr['naxis1']
           # loglam vector is the same for a given plate
           spplate_data[thishdu] = c0+c1*np.arange(npix, dtype='d')
        else:
           index = _spplate_hdunames.index(thishdu)
           spplate_data[thishdu] = spplate_fits[index].read(rows=fiber) 

    spplate_fits.close()

    return spplate_data

