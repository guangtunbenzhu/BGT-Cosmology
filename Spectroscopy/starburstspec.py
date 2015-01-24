
""" 

"""

from __future__ import division

from os.path import isfile, join
import numpy as np
import fitsio
import datapath
import allinonespec as aio
import specutils
from scipy.stats import nanmean, nanmedian
from progressbar import ProgressBar

# prefixes
_allinone_observer_bands = ['FUV', 'NUV']
_allinone_rest_bands     = ['FUV', 'NUV']
_allinone_observer_fileprefix = 'AIO_SBSF_HSTFOS_ObserverFrame_'
_allinone_rest_fileprefix     = 'AIO_SBSF_HSTFOS_NEDzRestFrame_'

_starburstfile = 'catalog.txt'
_subpath = 'starburst/fos_ghrs_spectra'
_minmaxwave = [3600., 10400.]

#mgii_index = np.array([12, 18, 19, 20, 25, 27, 28, 32, 33, 34, 35, 36, 37, 44, 45])
_mgii_index = np.array([12, 18, 19, 20, 25, 28, 32, 34, 35, 45])
#_mgii_index = np.array([12, 18, 19, 20, 25, 27, 28, 32, 33, 34, 35, 45])
#_mgii_index = np.array([12, 18, 19, 20, 25, 28, 32, 33, 34, 35, 45])
#_mgii_index = np.array([18, 20, 25, 28, 32, 33, 35, 45])

def starburst_filename():
    path = datapath.hstfos_path()
    return join(path, _subpath, _starburstfile)

def starburst_readin():
    infile = starburst_filename()

    if isfile(infile):
       return np.genfromtxt(infile, dtype=[('gal', 'S15')])
    else:
       raise IOError("Can't find file {0}.".format(infile))

def allinone_rest_filename(band):
    return aio.allinone_filename(band, prefix=_allinone_rest_fileprefix)

def allinone_observer_filename(band):
    return aio.allinone_filename(band, prefix=_allinone_observer_fileprefix)

def readspec_rest(obj):
    restpath = join(datapath.hstfos_path(), _subpath, 'corrected')
    specfile = obj['gal']+'.fits'
    restfile = join(restpath, specfile)
    return (fitsio.read(restfile))[0]

def readspec_obs(obj):
    obspath = join(datapath.hstfos_path(), _subpath, 'observed')
    obsfile = join(obspath, obj['gal']+'.fits')
    return (fitsio.read(obsfile))[0]

def rest_allspec(overwrite=False):
    """Load and interpolate *ALL* HST FOS/GHRS starburst spectra
    on to the same rest-frame wavelength grid
    """

    path = join(datapath.hstfos_path(), _subpath, 'corrected')

    # check output files
    bands = _allinone_rest_bands
    for thisband in bands:
        # check outfiles
        outfile = allinone_rest_filename(thisband)
        if isfile(outfile) and not overwrite:
           print "File {0} exists. Use overwrite to overwrite it.".format(outfile)
           return -1
        # print "Will write into these files: {0}".format(outfile)

    # read in the starburst catalog
    objs_ori = starburst_readin()
    nobj = objs_ori.size

    # make a temporary new catalog
    objs_dtype = [('RA', 'f8'),
                  ('DEC', 'f8'),
                  ('Z', 'f8'),
                  ('gal', 'S15')]
    objs = np.zeros(nobj, dtype=objs_dtype)
    objs['RA'] = 0.
    objs['DEC'] = 0.
    objs['Z'] = 0.
    objs['gal'] = objs_ori['gal']

    # read in master wavelength grid
    master_wave = (aio.allinone_wave_readin())[0]['WAVE']
    master_loglam = np.log10(master_wave)
    nwave = master_wave.size

    # initialization, nobj second dimension because of NMF traditions
    rest_allflux = np.zeros((nwave, nobj))
    rest_allivar = np.zeros((nwave, nobj))

    # Wavelength
    wave_pos = np.array([1000., 3300.])
    rest_loc = np.searchsorted(master_wave, wave_pos)
    newloglam = master_loglam[rest_loc[0]:rest_loc[1]]
    flux = np.zeros((objs.size, newloglam.size))
    ivar = np.zeros((objs.size, newloglam.size))

    pbar = ProgressBar(maxval=nobj).start()
    # Progress bar
    for (iobj, thisobj)  in zip(np.arange(objs.size), objs):
        pbar.update(iobj)
        thisdata = readspec_rest(thisobj)
        inloglam = np.log10(thisdata['wave'])
        influx = thisdata['flux']
        inivar = 1./np.power(thisdata['error'], 2)
        (rest_allflux[rest_loc[0]:rest_loc[1], iobj], rest_allivar[rest_loc[0]:rest_loc[1], iobj]) = specutils.interpol_spec(inloglam, influx, inivar, newloglam)

    #Progress bar
    pbar.finish()

    # write out
    print "Now I am writing everything out..."
    allinone_rest_writeout(objs, master_wave, rest_allflux, rest_allivar, overwrite=overwrite)


def allinone_rest_writeout(objs, wave, flux, ivar, overwrite=False):
    """Write out into an AllInOne file in the rest frame
    """

    # check output files
    bands = _allinone_rest_bands
    for thisband in bands:
        # check outfiles
        outfile = allinone_rest_filename(thisband)
        if isfile(outfile) and not overwrite:
           print "File {0} exists. Use overwrite to overwrite it.".format(outfile)
        # print "Will write into these files: {0}".format(outfile)

        # wavelength range
        wavebase = aio.allinone_wavebase(thisband)
        index_wave = np.searchsorted(wave, wavebase)
        nwave = index_wave[1] - index_wave[0]

        # objects with redshift in the covered range
        # index_obj = (np.where(np.logical_and((objs['Z'] > (_minmaxwave[0]/wave[index_wave[1]]-1.-0.001)), (objs['Z'] <= (_minmaxwave[1]/wave[index_wave[0]]-1.+0.001)))))[0]
        # All objects
        index_obj = np.arange(objs.size)
        if index_obj.size>0:
           outstr_dtype = [('INDEX_OBJ', 'i4', (index_obj.size,)), 
                           ('RA', 'f8', (index_obj.size,)), ('DEC', 'f8', (index_obj.size,)), ('Z', 'f4', (index_obj.size,)),
                           ('INDEX_WAVE', 'i4', (2,)), 
                           ('WAVE', 'f4', (nwave, )), 
                           ('FLUX', 'f4', (nwave, index_obj.size)), 
                           ('IVAR', 'f4', (nwave, index_obj.size))]
           outstr = np.array([(index_obj,
                               objs[index_obj]['RA'], objs[index_obj]['DEC'], objs[index_obj]['Z'], 
                               index_wave, 
                               wave[index_wave[0]:index_wave[1]], 
                               flux[index_wave[0]:index_wave[1], index_obj],
                               ivar[index_wave[0]:index_wave[1], index_obj])],
                               dtype=outstr_dtype)
           fits = fitsio.FITS(outfile, 'rw', clobber=overwrite)
           fits.write(outstr)
           fits.close()


def allinone_rest_readin_band(band):
    infile = allinone_rest_filename(band)
    if isfile(infile):
       print "Reading {0}.".format(infile)
       return (fitsio.read(infile))[0]
    else:
       raise IOError("Can't find {0}".format(infile))

def rest_allspec_readin():

    # read in the starburst catalog
    objs_ori = starburst_readin()
    nobj = objs_ori.size
   
    # read in master wavelength grid
    master_wave = (aio.allinone_wave_readin())[0]['WAVE']
    master_loglam = np.log10(master_wave)
    nwave = master_wave.size

    # initialization, nobj second dimension because of NMF traditions
    rest_allflux = np.zeros((nwave, nobj))
    rest_allivar = np.zeros((nwave, nobj))

    bands = _allinone_rest_bands
    for thisband in bands:
        data = allinone_rest_readin_band(thisband)
        index_wave = data['INDEX_WAVE']
        index_obj = data['INDEX_OBJ']
        rest_allflux[index_wave[0]:index_wave[1], index_obj] = data['FLUX']
        rest_allivar[index_wave[0]:index_wave[1], index_obj] = data['IVAR']

    return (master_wave, rest_allflux, rest_allivar)

def mgii_composite():
    (master_wave, rest_allflux, rest_allivar) = rest_allspec_readin()
    master_loglam = np.log10(master_wave)
    # mask out useless wavelength ranges
    # left 2300
    wave_pos = np.array([2300.])
    rest_loc = np.searchsorted(master_wave, wave_pos)
    rest_allivar[0:rest_loc[0],:] = 0.
    # Fe II 2350
    wave_pos = np.array([2330., 2420])
    rest_loc = np.searchsorted(master_wave, wave_pos)
    rest_allivar[rest_loc[0]:rest_loc[1],:] = 0.
    # Fe II 2600
    wave_pos = np.array([2570., 2640])
    rest_loc = np.searchsorted(master_wave, wave_pos)
    rest_allivar[rest_loc[0]:rest_loc[1],:] = 0.
    # Mg II 2800
    wave_pos = np.array([2770., 2820])
    rest_loc = np.searchsorted(master_wave, wave_pos)
    rest_allivar[rest_loc[0]:rest_loc[1],:] = 0.
    # Mg I 2853
    wave_pos = np.array([2843., 2863])
    rest_loc = np.searchsorted(master_wave, wave_pos)
    rest_allivar[rest_loc[0]:rest_loc[1],:] = 0.
    # right 2900
    wave_pos = np.array([2900.])
    rest_loc = np.searchsorted(master_wave, wave_pos)
    rest_allivar[rest_loc[0]:,:] = 0.

    normalized_rest_allflux = rest_allflux
    for i in np.arange((rest_allflux.shape)[1]):
        imask = (np.where(rest_allivar[:,i]>0.))[0]
        if imask.size>0: 
           x = np.log10(master_wave[imask])
           y = rest_allflux[imask, i]
           z = np.polyfit(x, y, 3)
           p = np.poly1d(z)
           continuum = p(master_loglam)
           normalized_rest_allflux[:,i] = rest_allflux[:,i]/continuum
    
    wave_pos = np.array([2300., 2900.])
    rest_loc = np.searchsorted(master_wave, wave_pos)

    outwave = master_wave[rest_loc[0]:rest_loc[1]]
    fluxmean = nanmean(normalized_rest_allflux[rest_loc[0]:rest_loc[1], _mgii_index], 1)
    fluxmedian = nanmedian(normalized_rest_allflux[rest_loc[0]:rest_loc[1], _mgii_index], 1)
    fluxused = normalized_rest_allflux[rest_loc[0]:rest_loc[1], _mgii_index]

    return (outwave, fluxmean, fluxmedian, fluxused)



