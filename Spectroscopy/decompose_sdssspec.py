"""
Utils for working on decomposed SDSS spectra
"""

from __future__ import division, print_function
from os.path import isfile, join
import numpy as np
import fitsio
import datapath
import specutils


def decompose_filename(platein, fiberin, path):
    """Name for a spPlate file
    """
    plate = np.ravel(platein)
    fiber = np.ravel(fiberin)
    if plate.size != 1 or fiber.size !=1:
       raise ValueError("I can only take one plate and one fiber.")

    subpath = '{0:04d}'.format(int(plate))
    filename = 'QSO_spec_decomposed_{0:04d}_{1:03d}.fits'.format(int(plate), int(fiber))

    return join(path, subpath, filename)


# http://data.sdss3.org/datamodel/files/SPECTRO_REDUX/RUN2D/PLATE4/spPlate.html
def read_spec(platein, fiberin, path): 
    """Read in spectra 
    """

    # Check input
    plate = np.ravel(platein)
    fiber = np.ravel(fiberin)
    if plate.size != 1 or fiber.size !=1:
       raise ValueError("I can only take one plate and one fiber.")

    # get filename
    filename = decompose_filename(plate, fiber, path)
    if not isfile(filename):
       raise ValueError("I can't find this file {0}.".format(filename))

    #print(filename)
    return (fitsio.read(filename))[0]

def load_interp_spec(objs, newloglam, path, rest=False):
    """Read in spectra of a list of objects and interpolate them onto the same wavelength grid
    objs : a structured array with 'plate', 'mjd', 'fiber'
    newloglam : desired wavelength grid in logarithmic scale
    path : 
    """

    # check objs, newloglam
    objs = np.ravel(objs)

    # output:
    flux = np.zeros((objs.size, newloglam.size))
    ivar = np.zeros((objs.size, newloglam.size))

    # read in spec
    output = ('loglam', 'flux', 'ivar',)
    for (iobj, thisobj)  in zip(np.arange(objs.size), objs):
        # Note it is in quasar's rest-frame
        thisdata = read_spec(thisobj['PLATE'], thisobj['FIBER'], path)
        # print(thisdata.shape)
        tmpz = thisobj['Z'] if rest else 0.
        inloglam = np.log10(thisdata['WAVE']*(1.+thisdata['Z'])/(1.+tmpz))
        influx = thisdata['RESIDUAL'] # Only want the residual
        inivar = thisdata['IVAR']*np.power(thisdata['NMF_CONTINUUM']*thisdata['MED_CONTINUUM'], 2)
        (flux[iobj, :], ivar[iobj, :]) = specutils.interpol_spec(inloglam, influx, inivar, newloglam)
    if objs.size == 1:
       flux = np.ravel(flux)
       ivar = np.ravel(ivar)

    return (flux, ivar)

