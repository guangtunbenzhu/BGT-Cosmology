from __future__ import division, print_function
from os.path import isfile, join
import numpy as np
import fitsio
import datapath
import specutils
from glob import glob


_spplate_hdunames = ('flux','ivar','andmask','ormask','disp','plugmap','sky', 'loglam', )

def spplate_filename(platein, mjdin, path):
    """Name for a spPlate file
    """
    plate = np.ravel(platein)
    mjd = np.ravel(mjdin)
    if plate.size != 1 or mjd.size !=1:
       raise ValueError("I can only take one plate and one mjd.")

    return join(path,"spPlate-{0:4d}-{1:05d}.fits".format(int(plate), int(mjd)))

# http://data.sdss3.org/datamodel/files/SPECTRO_REDUX/RUN2D/PLATE4/spPlate.html
def read_spec(platein, mjdin, fiberin, path, output): 
    """Read spectra from a *single* spPlate file
    a simpler, hopefully better version of readspec in idlspec2d/pydl
    requiring mjd and path so that one knows what they are doing
    output should be a tuple, e.g. ('flux', 'ivar',)
    """

    # check desired output
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
    if not isfile(filename):
       raise ValueError("I can't find this file {0}.".format(filename))

    #print(filename)
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
           #spplate_data[thishdu] = spplate_fits[index].read(rows=fiber.tolist) 
           # read in the whole image for now instead of a for loop to read in row by row
           spplate_data[thishdu] = (spplate_fits[index].read())[fiber,:] if fiber.size>1 else np.ravel((spplate_fits[index].read())[fiber,:])

    spplate_fits.close()

    return spplate_data

def load_interp_spec(objs, newloglam, path, rest=False):
    """Read spectra of a list of objects and interpolate them onto the same wavelength grid
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
        thisdata = read_spec(thisobj['PLATE'], thisobj['MJD'], thisobj['FIBER'], path, output)
        tmpz = thisobj['Z'] if rest else 0.
        inloglam = np.log10(np.power(10., thisdata['loglam'])/(1.+tmpz))
        influx = thisdata['flux']*(1.+tmpz)
        inivar = thisdata['ivar']/np.power(1.+tmpz, 2)
        (flux[iobj, :], ivar[iobj, :]) = specutils.interpol_spec(inloglam, influx, inivar, newloglam)
    if objs.size == 1:
       flux = np.ravel(flux)
       ivar = np.ravel(ivar)

    return (flux, ivar)

