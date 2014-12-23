
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
    #outweight = np.where(np.logical_and(np.isfinite(outweight), outweight>0.), outweight, 0.)
    outweight[~(np.logical_and(np.isfinite(outweight), outweight>0.))] = 0.
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
       raise IOError("{0} not found.")
       # print ("{0} does not exist! \n" 
       #       "Check the path or call allinone_wave_make() first.".format(infile))
       # return -1
    return fitsio.read(infile)


def allinone_wave_make(overwrite=False):
    outfile = allinone_wave_filename()
    if (isfile(outfile) and not overwrite):
       print "File exists. Use overwrite to overwrite it."
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


# hstfos utils
hstfos_allinone_observer_bands = ['FUV', 'NUV']
hstfos_allinone_rest_bands     = ['EUV', 'FUV', 'NUV']
hstfos_allinone_observer_fileprefix = 'AIO_QSO_HSTFOS_ObserverFrame_'
hstfos_allinone_rest_fileprefix     = 'AIO_QSO_HSTFOS_NEDzRestFrame_'


def hstfos_allinone_rest_filename(band):
    path = datapath.allinone_path()
    filebase = allinone_filebase(band)
    filename = join(path, hstfos_allinone_rest_fileprefix+filebase+'.fits')
    return join(path, filename)


def hstfos_allinone_observer_filename(band):
    path = datapath.allinone_path()
    filebase = allinone_observer_filebase(band)
    filename = hstfos_allinone_observer_fileprefix+filebase+'.fits'
    return join(path, filename)


def hstfos_readspec(obj, channel):
    """Read in an HST FOS quasar spectrum
    """
    path = datapath.hstfos_path()
    subpath = ((obj['NAME'].strip()).replace('+','p')).replace('-','m')
    filename = channel+'.fits'
    infile = join(path, subpath, filename)
    if not isfile(infile):
       raise IOError("{0} not found.")
    # print "Reading {0}.".format(infile)
    return fitsio.read(infile)


def hstfos_allinone_rest_writeout(qso, wave, flux, continuum, ivar, overwrite=False):
    """
    Saving continuum as well.
    """

    bands = hstfos_allinone_rest_bands
    for thisband in bands:
        # check outfiles
        outfile = hstfos_allinone_rest_filename(thisband)
        if isfile(outfile) and not overwrite:
           print "File {0} exists. Use overwrite to overwrite it.".format(outfile)
        print "Will write into these files: {0}".format(outfile)

        # wavelength range
        wavebase = allinone_wavebase(thisband)
        index_wave = np.searchsorted(wave, wavebase)
        nwave = index_wave[1] - index_wave[0]
        # quasars with redshift in the range
        index_qso = (np.where(np.logical_and((qso['Z'] > (1100./wave[index_wave[1]]-1.-0.001)), (qso['Z'] <= (3350./wave[index_wave[0]]-1.+0.001)))))[0]
        if index_qso.size>0:
           outstr_dtype = [('INDEX_QSO', 'i4', (index_qso.size,)), 
                           ('RA', 'f8', (index_qso.size,)), ('DEC', 'f8', (index_qso.size,)), ('Z', 'f4', (index_qso.size,)),
                           ('WAVE', 'f4', (nwave, )), 
                           ('FLUX', 'f4', (nwave, index_qso.size)), 
                           ('CONTINUUM', 'f4', (nwave, index_qso.size)), 
                           ('IVAR', 'f4', (nwave, index_qso.size))]
           outstr = np.array([(index_qso,
                               qso[index_qso]['RA'], qso[index_qso]['DEC'], qso[index_qso]['Z'], 
                               wave[index_wave[0]:index_wave[1]], 
                               flux[index_wave[0]:index_wave[1], index_qso],
                               continuum[index_wave[0]:index_wave[1], index_qso],
                               ivar[index_wave[0]:index_wave[1], index_qso])],
                               dtype=outstr_dtype)
           fits = fitsio.FITS(outfile, 'rw', clobber=overwrite)
           fits.write(outstr)
           fits.close()


def hstfos_allinone_rest_readin(band):
    infile = hstfos_allinone_rest_filename(band)
    if isfile(infile):
       print "Reading {0}.".format(infile)
       return (fitsio.read(infile))[0]
    else:
       raise IOError("Can't find {0}").format(infile)

def somethingsomething():
    # read in quasars
    qso = hstfos_qso_readin() 
    nqso = qso.size

    # read in master wavelength grid
    master_wave = (allinone_wave_readin())[0]['WAVE']
    master_loglam = np.log10(master_wave)
    nwave = master_wave.size

    bands = hstfos_allinone_rest_bands
    rest_allflux = np.zeros((nwave, nqso))
    rest_allcont = np.zeros((nwave, nqso))
    rest_allivar = np.zeros((nwave, nqso))

    # read out
    for thisband in bands:
        wavebase = allinone_wavebase(thisband)
        index_wave = np.searchsorted(master_wave, wavebase)
        nwave = index_wave[1] - index_wave[0]

        thisstr = hstfos_allinone_rest_readin(thisband)
        index_qso = thisstr['INDEX_QSO']
        rest_allflux[index_wave[0]:index_wave[1], index_qso] = thisstr['FLUX']
        rest_allcont[index_wave[0]:index_wave[1], index_qso] = thisstr['CONTINUUM']
        rest_allivar[index_wave[0]:index_wave[1], index_qso] = thisstr['IVAR']


    # Perform composite or NMF or something or something
        

def hstfos_qso_filename():
    path = datapath.hstfos_path()
    filename = 'hstfos_master_visual.fits'
    return join(path, filename)


def hstfos_qso_readin():
    infile = hstfos_qso_filename()
    return fitsio.read(infile)


def hstfos_rest_allqsospec(overwrite=False):
    """Load and interpolate *ALL* HST FOS quasar spectra

    on to the same rest-frame wavelength grid
    """
    # check output files
    bands = hstfos_allinone_rest_bands
    for thisband in bands:
        # check outfiles
        outfile = hstfos_allinone_rest_filename(thisband)
        if isfile(outfile) and not overwrite:
           print "File {0} exists. Use overwrite to overwrite it.".format(outfile)
        print "Will write into these files: {0}".format(outfile)

    # read in quasars
    qso = hstfos_qso_readin() 
    nqso = qso.size

    # read in master wavelength grid
    master_wave = (allinone_wave_readin())[0]['WAVE']
    master_loglam = np.log10(master_wave)
    nwave = master_wave.size

    rest_allflux = np.zeros((nwave, nqso))
    rest_allcont = np.zeros((nwave, nqso))
    rest_allivar = np.zeros((nwave, nqso))

    channel = ['h130', 'h190', 'h270']
    #question = 'y'

    pbar = ProgressBar(maxval=nqso).start()
    for i in np.arange(nqso):
    # for i in np.arange(10):
        # Progress bar
        pbar.update(i)
        tmpz = qso[i]['Z']
        wave_pos = np.array([1100./(1.+tmpz), 3350./(1.+tmpz)])
        rest_loc = np.searchsorted(master_wave, wave_pos)
        tmp_outflux = np.zeros((rest_loc[1]-rest_loc[0],3))
        tmp_outivar = np.zeros((rest_loc[1]-rest_loc[0],3))
        tmp_outcont = np.zeros((rest_loc[1]-rest_loc[0],3))
        for j in np.arange(3):
            try:
               tmpspec = (hstfos_readspec(qso[i], channel[j]))[0]
               if np.count_nonzero(tmpspec['error'])==0:
                  raise ValueError("All errors are 0. Something's wrong!")
                  continue
            except IOError:
               #print "Reading channel {0} spectrum failed.".format(channel[j])
               continue
            tmploglam = np.log10(tmpspec['wave']/(1.+tmpz))
            tmpflux = np.zeros(tmploglam.size)
            tmpcont = np.zeros(tmploglam.size)
            tmpivar = np.zeros(tmploglam.size)
            inonzero_error = np.nonzero(tmpspec['error'])
            tmpflux[inonzero_error] = tmpspec['flux'][inonzero_error]*(1.+tmpz)
            tmpcont[inonzero_error] = tmpspec['continuum'][inonzero_error]*(1.+tmpz)
            tmpivar[inonzero_error] = 1./pow(tmpspec['error'][inonzero_error], 2)/pow((1.+tmpz),2)

            # Interpolation, Flux
            # print "Interpolation..."
            try: 
                finterp, iinterp = combine1fiber(tmploglam, tmpflux, 
                   objivar=tmpivar, newloglam=master_loglam[rest_loc[0]:rest_loc[1]])
                # print "Interpolation Done..."
                tmp_outflux[:,j] = finterp
                tmp_outivar[:,j] = iinterp
            except (IndexError, TypeError, NameError):
                print "Something went wrong while working on qso {0}".format(i)

            # Interpolation, Continuum
            # print "Interpolation..."
            try: 
                cinterp, ciinterp = combine1fiber(tmploglam, tmpcont, 
                   objivar=tmpivar, newloglam=master_loglam[rest_loc[0]:rest_loc[1]])
                tmp_outcont[:,j] = cinterp
            # print "Interpolation Done..."
            except (IndexError, TypeError, NameError):
                print "Something went wrong while working on qso {0}".format(i)

        #question = raw_input("Do you want to continue? (y/n): ")
        #if question!='y':
           #print "Per your request, I am stopping here."
           #return None

        # Coadd
        # print "Coadd..."
        tmp_allflux, tmp_allivar      = simple_coadd(tmp_outflux, 1, weight=tmp_outivar)
        tmp_allcont, tmp_allivar_cont = simple_coadd(tmp_outcont, 1, weight=tmp_outivar)

        # output
        rest_allflux[rest_loc[0]:rest_loc[1],i] = tmp_allflux*10. # 10 for 10^(-16) to 10^(-17)
        rest_allcont[rest_loc[0]:rest_loc[1],i] = tmp_allcont*10. # 10 for 10^(-16) to 10^(-17)
        rest_allivar[rest_loc[0]:rest_loc[1],i] = tmp_allivar/100.

    pbar.finish()
    # write out
    print "Now I am writing everything out..."
    hstfos_allinone_rest_writeout(qso, master_wave, rest_allflux, rest_allcont, rest_allivar, overwrite=overwrite)

