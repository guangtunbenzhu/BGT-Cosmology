
""" 

"""

from __future__ import division

import numpy as np
import fitsio
import datapath
from os.path import isfile, join
from specutils import get_loglam
from progressbar import ProgressBar
from pydl.pydlspec2d.spec2d import combine1fiber


# specutils
def simple_coadd(flux, axis, weight=None):
    if weight is None: 
       weight = np.where(np.isfinite(flux), 1., 0.)
    outweight = np.sum(weight, axis=axis)
    outweight = np.where((np.isfinite(outweight) and outweight>0.), outweight, 0.)
    outflux = np.sum(flux*weight, axis=axis)/(outweight + (outweight==0.))
    return outflux, outweight


# allinone utils
def allinone_wave_filename():
    path = datapath.allinone_path()
    filename = 'AIO_CommonWave.fits'
    return join(path, filename)


def allinone_wave_readin():
    infile = allinone_wave_filename()
    if not isfile(infile):
       print ("{0} does not exist! \n" 
              "Check the path or call allinone_wave_make() first.".format(infile))
       return -1
    return fitsio.read(infile)


def allinone_wave_make(overwrite=False):
    outfile = allinone_wave_filename()
    if (isfile(outfile) and not overwrite):
       print "File exists. Use overwrite to overwrite it."
       return -1
    master_loglam = get_loglam(minwave=448., maxwave=10402., dloglam=1.E-4)
    master_wave = np.power(10., master_loglam)
    # preferred structured array
    outstr = np.zeros(1, dtype=[('wave','f8', (master_loglam.size,))])
    # Using outstr[0]['wave'] does not work, only the first element got the value, why? Memory layout conflict?
    # Or shape conflict (1, n) vs. (n,)
    outstr[0]['wave'][:] = np.power(10., master_loglam)
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


# hstfos utils
hstfos_allinone_observer_bands = ['FUV', 'NUV']
hstfos_allinone_rest_bands     = ['EUV', 'FUV', 'NUV']
hstfos_allinone_observer_fileprefix = 'AIO_QSO_HSTFOS_ObserverFrame_'
hstfos_allinone_rest_fileprefix     = 'AIO_QSO_HSTFOS_NEDzRestFrame_'


def hstfos_allinone_rest_filename(band):
    path = datapath.allinone_path()
    filebase = allinone_rest_filebase(band)
    filename = join(path, hstfos_allinone_rest_fileprefix+filebase+'.fits')
    return join(path, filename)


def hstfos_allinone_observer_filename(band):
    path = datapath.allinone_path()
    filebase = allinone_observer_filebase(band)
    filename = hstfos_allinone_observer_fileprefix+filebase+'.fits'
    return join(path, filename)


def hstfos_readspec(obj, channel, status=None):
    """Read in an HST FOS quasar spectrum
    """
    path = datapath.hstfos_path()
    subpath = ((obj['Name'].strip()).replace('+','p')).replace('-','m')
    filename = channel+'.fits'
    infile = join(path, subpath, filename)
    print infile
    if isfile(infile) is True:
       status=1
       return fitsio.read(infile)
    status=-1
    return status


def hstfos_allinone_rest_writeout(qso, wave, flux, continuum, ivar, overwrite=False):
    """
    Saving continuum as well.
    """

    bands = hstfos_allinone_restframe_bands
    for thisband in bands:
        # check outfiles
        outfile = hstfos_allinone_restframe_filename(thisband)
        if isfile(thisfile) and not overwrite:
           print "File {0} exists. Use overwrite to overwrite it.".format(outfile)
        print "Will write into these files: {0}".format(outfile)
        wavebase = allinone_rest_wavebase(thisband)

        # wavelength range
        index_wave = np.searchsorted(wave, wavebase)
        nwave = index_wave[1] - index_wave[0]
        # quasars with flux in the range
        index_qso = (where((qso.z > (1100./wave[index_wave[1]]-1.-0.001)) and (qso.z <= (3350./wave[index_wave[0]]-1.+0.001))))[0]
        if index_qso.size>0:
           outstr_dtype = [('index_qso', 'i4', (index_qso.size,)), 
                           ('ra', 'f8', (index_qso.size,)), ('dec', 'f8', (index_qso.size,)), ('z', 'f4', (index_qso.size,)),
                           ('wave', 'f4', (nwave,)), 
                           ('flux', 'f4', (nwave,)), 
                           ('continuum', 'f4', (nwave,)), 
                           ('ivar', 'f4', (nwave,))] 
           outstr = np.array([(index_qso,
                               qso[index_qso]['ra'], qso[index_qso]['dec'], qso[index_qso]['z'], 
                               wave[index_wave[0]:index_wave[1]], 
                               flux[index_wave[0]:index_wave[1], index_qso],
                               continuum[index_wave[0]:index_wave[1], index_qso],
                               ivar[index_wave[0]:index_wave[1], index_qso])],
                               dtype=outstr_dtype)
           fits = fitsio.FITS(outfile, 'rw', clobber=overwrite)
           fits.write(outstr)
           fits.close()


def hstfos_qso_filename():
    path = datapath.hstfos_path()
    filename = 'HSTFOS_qso.fits'
    return join(path, filename)


def hstfos_qso_readin():
    infile = hstfos_qso_filename()
    return fitsio.read(infile)


def hstfos_rest_allqsospec(overwrite=False):
    """Load and interpolate *ALL* HST FOS quasar spectra

    on to the same rest-frame wavelength grid
    """
    pbar = ProgressBar()

    # check output files
    
    outfiles = hstfos_allinone_restframe_allfiles()
    for thisfile in outfiles:
        if isfile(thisfile) and not overwrite:
           print "At least one of the files exists. Use overwrite to overwrite it."
           return None
    print "Will write into these files: {0}".format(outfiles)

    # read in quasars
    qso = hstfos_qso_readin() 

    # read in master wavelength grid
    master_wave = (allinone_wave_readin())['wave']
    master_loglam = np.log10(master_wave)
    nwave = master_wave.size()

    rest_allflux = np.zeros((nwave, nqso))
    rest_allcont = np.zeros((nwave, nqso))
    rest_allivar = np.zeros((nwave, nqso))

    channel = ['h130', 'h190', 'h270']
    for i in np.arange(nqso):
        # Progress bar
        pbar.update(i+1)
        tmpz = qso[i].z
        wave_pos = np.array([1100./(1.+tmpz), 3350./(1.+tmpz)])
        rest_loc = np.searchsorted(master_wave, wave_pos)
        for j in np.arange(3):
            tmpspec = hstfos_readspec(qso[i], channel[j], status=status)
            if status==1:
               tmploglam = np.log10(tmpspec['wave']/(1.+tmpz))
               tmpflux = tmpspec['flux']/(1.+tmpz)
               tmpcont = tmpspec['continuum']/(1.+tmpz)
               tmpivar = 1./pow(tmpspec['error'], 2)/(1.+tmpz)
               tmpivar = np.where(np.isfinite(tmpivar), tmpivar, 0.0)

               # Interpolation, Flux
               finterp, iinterp = combine1fiber(tmploglam, tmpflux, 
                   objivar=tmpivar, newloglam=master_loglam[rest_loc[0]:rest_loc[1]])
               tmp_outflux[:,j] = finterp
               tmp_outivar[:,j] = iinterp
               # Interpolation, Continuum
               cinterp, ciinterp = combine1fiber(tmploglam, tmpcont, 
                   objivar=tmpivar, newloglam=master_loglam[rest_loc[0]:rest_loc[1]])
               tmp_outcont[:,j] = cinterp

        # Coadd
        tmp_allflux, tmp_allivar      = simple_coadd(tmp_outflux, 1, weight=tmp_outivar)
        tmp_allcont, tmp_allivar_cont = simple_coadd(tmp_outcont, 1, weight=tmp_outivar)

        # output
        rest_allflux[rest_loc[0]:rest_loc[1],i] = tmp_allflux*10. # 10 for 10^(-16) to 10^(-17)
        rest_allcont[rest_loc[0]:rest_loc[1],i] = tmp_allcont*10. # 10 for 10^(-16) to 10^(-17)
        rest_allivar[rest_loc[0]:rest_loc[1],i] = tmp_allivar/100.

    # write out
    hstfos_allinone_rest_writeout(master_wave, rest_allflux, rest_allcont, rest_allivar)

