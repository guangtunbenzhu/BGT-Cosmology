
""" 

"""

from __future__ import division

from os.path import isfile, join
import numpy as np
import fitsio
import datapath
import allinonespec as aio
import sdssspec as sdssspec
from progressbar import ProgressBar

# prefixes
_allinone_observer_bands = ['OPTICAL']
_allinone_rest_bands     = ['NUV', 'OPTICAL']
_allinone_observer_fileprefix = 'AIO_ELG_eBOSS_ObserverFrame_'
_allinone_rest_fileprefix     = 'AIO_ELG_eBOSS_SDSSRestFrame_'
_elgfile = 'spAll-ELG-v5.4-zQ.fits'
_minmaxwave = [3600., 10400.]

def elg_filename():
    path = datapath.sdss_path()
    return join(path, 'eBOSS', _elgfile)

def elg_readin():
    infile = elg_filename()
    if isfile(infile):
       return fitsio.read(infile, ext=1)
    else:
       raise IOError("Can't find file {0}.".format(infile))

def allinone_rest_filename(band):
    return aio.allinone_filename(band, prefix=_allinone_rest_fileprefix)

def allinone_observer_filename(band):
    return aio.allinone_filename(band, prefix=_allinone_observer_fileprefix)

def rest_allspec(overwrite=False):
    """Load and interpolate *ALL* eBOSS ELG spectra
    on to the same rest-frame wavelength grid
    """

    path1 = join(datapath.sdss_path(), 'v5_7_6')
    path2 = join(datapath.sdss_path(), 'specDR12')

    # check output files
    bands = _allinone_rest_bands
    for thisband in bands:
        # check outfiles
        outfile = allinone_rest_filename(thisband)
        if isfile(outfile) and not overwrite:
           print "File {0} exists. Use overwrite to overwrite it.".format(outfile)
           return -1
        # print "Will write into these files: {0}".format(outfile)

    # read in the elg catalog
    objs_ori = elg_readin()
    nobj = objs_ori.size

    # make a temporary new catalog
    objs_dtype = [('PLATE', 'i4'),
                  ('MJD', 'i4'),
                  ('FIBER', 'i4'),
                  ('RA', 'f8'),
                  ('DEC', 'f8'),
                  ('Z', 'f8')]
    objs = np.zeros(nobj, dtype=objs_dtype)
    objs['PLATE'] = objs_ori['PLATE_1']
    objs['MJD'] = objs_ori['MJD']
    objs['FIBER'] = objs_ori['FIBERID_1']
    objs['RA'] = objs_ori['PLUG_RA']
    objs['DEC'] = objs_ori['PLUG_DEC']
    objs['Z'] = objs_ori['Z']

    # read in master wavelength grid
    master_wave = (aio.allinone_wave_readin())[0]['WAVE']
    master_loglam = np.log10(master_wave)
    nwave = master_wave.size

    # initialization, nobj second dimension because of NMF traditions
    rest_allflux = np.zeros((nwave, nobj))
    rest_allivar = np.zeros((nwave, nobj))
    #rest_allflux = np.zeros((nwave, 10))
    #rest_allivar = np.zeros((nwave, 10))

    # Progress bar
    pbar = ProgressBar(maxval=nobj).start()
    #for i in np.arange(10):
    for i in np.arange(nobj):
        # Progress bar
        pbar.update(i)

        tmpz = objs[i]['Z']

        # Wavelength
        wave_pos = np.array([3600./(1.+tmpz), 10400./(1.+tmpz)])
        rest_loc = np.searchsorted(master_wave, wave_pos)
        tmp_loglam = master_loglam[rest_loc[0]:rest_loc[1]]

        # read and interpolate
        try:
            tmp_outflux, tmp_outivar = sdssspec.load_interp_spec(objs[i], tmp_loglam, path1, rest=True)
            rest_allflux[rest_loc[0]:rest_loc[1],i] = tmp_outflux
            rest_allivar[rest_loc[0]:rest_loc[1],i] = tmp_outivar
        except (IndexError, TypeError, NameError, ValueError):
            try:
                tmp_outflux, tmp_outivar = sdssspec.load_interp_spec(objs[i], tmp_loglam, path2, rest=True)
                rest_allflux[rest_loc[0]:rest_loc[1],i] = tmp_outflux
                rest_allivar[rest_loc[0]:rest_loc[1],i] = tmp_outivar
            except (IndexError, TypeError, NameError, ValueError):
                print("Error reading plate {0} mjd {1} fiber {2}".format(objs[i]['PLATE'], objs[i]['MJD'], objs[i]['FIBER']))

        # output

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

        # U R here.
        # objects with redshift in the covered range
        index_obj = (np.where(np.logical_and((objs['Z'] > (_minmaxwave[0]/wave[index_wave[1]]-1.-0.001)), (objs['Z'] <= (_minmaxwave[1]/wave[index_wave[0]]-1.+0.001)))))[0]
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

    # read in the elg catalog
    objs_ori = elg_readin()
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


