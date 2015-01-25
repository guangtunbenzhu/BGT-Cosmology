
""" 
Absorption spectroscopy for strong absorbers
"""

from __future__ import division

from os.path import isfile, join
import numpy as np
import fitsio
import datapath
import allinonespec as aio
import decompose_sdssspec as decompose_sdssspec
from progressbar import ProgressBar

# prefixes
_allinone_observer_bands = ['OPTICAL']
_allinone_rest_bands     = ['FUV', 'NUV', 'OPTICAL']
_allinone_observer_fileprefix_dr7 = 'AIO_MgIIAbsorber_SDSS_ObserverFrame_'
_allinone_rest_fileprefix_dr7     = 'AIO_MgIIAbsorber_SDSS_SDSSRestFrame_'
_allinone_observer_fileprefix_dr12 = 'AIO_MgIIAbsorber_BOSS_ObserverFrame_'
_allinone_rest_fileprefix_dr12     = 'AIO_MgIIAbsorber_BOSS_SDSSRestFrame_'
_absfile_dr7 = 'Trimmed_SDSS_DR7_107.fits'
_absfile_dr12 = 'Trimmed_BOSS_DR12_107.fits'
_minwave = 3600.
_maxwave = 10400.

def allinone_rest_filename(band, version='DR7'):
    '''Filename for All_in_One (AIO) file in the rest frame
    '''
    if version == 'DR7': 
       return aio.allinone_filename(band, prefix=_allinone_rest_fileprefix_dr7)
    elif version == 'DR12':
       return aio.allinone_filename(band, prefix=_allinone_rest_fileprefix_dr12)
    else:
       raise IOError("Please check the data release version.")


def allinone_observer_filename(band):
    '''Filename for All_in_One (AIO) file in the observer frame
    '''
    if version == 'DR7': 
       return aio.allinone_filename(band, prefix=_allinone_observer_fileprefix_dr7)
    elif version == 'DR12':
       return aio.allinone_filename(band, prefix=_allinone_observer_fileprefix_dr12)
    else:
       raise IOError("Please check the data release version.")


def absorber_filename(version='DR7'):
    '''Filename for the catalog
    '''
    path = datapath.absorber_path()
    if version == 'DR7':
       filename = _absfile_dr7
    elif version == 'DR12':
       filename = _absfile_dr12
    else:
       raise IOError("Please check the data release version.")
    return join(path, filename)

def absorber_readin(version='DR7'):
    ''' Read in the catalog
    '''
    infile = absorber_filename(version)
    if isfile(infile):
       return fitsio.read(infile, ext=1)
    else:
       raise IOError("Can't find file {0}.".format(infile))

def rest_allspec(version='DR7', overwrite=False):
    """Make the All_in_One (AIO) file in the rest frame
    Load and interpolate *ALL* Absorber spectra on to the same rest-frame wavelength grid
    """

    # Spectra directory
    path0 = join(datapath.nmf_path(), '107')
    if version == 'DR7':
       path = join(path0, 'Decompose')
    elif version == 'DR12':
       path = join(path0, 'Decompose_DR12')
    else:
       raise IOError("Please check the data release version and the path to the decomposed spectroscopic data.")

    # Check All_in_One (AIO) output files
    bands = _allinone_rest_bands
    for thisband in bands:
        # Check outfiles
        outfile = allinone_rest_filename(thisband, version)
        if isfile(outfile) and not overwrite:
           print("File {0} exists. Use overwrite to overwrite it.".format(outfile))
           return -1

    # Read in the catalog
    objs_ori = absorber_readin(version)
    nobj = objs_ori.size

    # Make a temporary new catalog, universal
    objs_dtype = [('PLATE', 'i4'),
                  ('MJD', 'i4'),
                  ('FIBER', 'i4'),
                  ('RA', 'f8'),
                  ('DEC', 'f8'),
                  ('Z', 'f8')]
    objs = np.zeros(nobj, dtype=objs_dtype)

    # Assign the right values
    objs['PLATE'] = objs_ori['PLATE']
    objs['MJD'] = objs_ori['MJD']
    objs['FIBER'] = objs_ori['FIBER']
    objs['RA'] = objs_ori['RA']
    objs['DEC'] = objs_ori['DEC']
    objs['Z'] = objs_ori['ZABS'] # This is very important 

    # Read in master wavelength grid
    master_wave = (aio.allinone_wave_readin())[0]['WAVE']
    master_loglam = np.log10(master_wave)
    nwave = master_wave.size

    # Initialization, nobj second dimension because of NMF convention
    rest_allflux = np.zeros((nwave, nobj))
    rest_allivar = np.zeros((nwave, nobj))
    # rest_allflux = np.zeros((nwave, 10)) # This is for testing
    # rest_allivar = np.zeros((nwave, 10))

    # Initialization of the progress bar
    pbar = ProgressBar(maxval=nobj).start()
    # for i in np.arange(10): # This is for testing
    for i in np.arange(nobj):
        # Progress bar
        pbar.update(i)
        tmpz = objs[i]['Z']

        # Useful wavelength range in the observer frame
        wave_pos = np.array([_minwave/(1.+tmpz), _maxwave/(1.+tmpz)])
        rest_loc = np.searchsorted(master_wave, wave_pos)
        tmp_loglam = master_loglam[rest_loc[0]:rest_loc[1]]

        # Read in and interpolate
        tmp_outflux, tmp_outivar = decompose_sdssspec.load_interp_spec(objs[i], tmp_loglam, path, rest=True)
        rest_allflux[rest_loc[0]:rest_loc[1],i] = tmp_outflux
        rest_allivar[rest_loc[0]:rest_loc[1],i] = tmp_outivar

    # Closing progress bar
    pbar.finish()
    # raise ValueError("Stop here")
    # Write out
    print "Now I am writing everything out..."
    allinone_rest_writeout(objs, master_wave, rest_allflux, rest_allivar, version, overwrite=overwrite)

# Write out the All_in_One (AIO) array in the rest frame into the AIO file
# The reason to have different bands and only save those objects with redshift in the covered range is disk space
def allinone_rest_writeout(objs, wave, flux, ivar, version='DR7', overwrite=False):
    """Write out into an AllInOne file in the rest frame
    """

    # Check output files
    bands = _allinone_rest_bands
    for thisband in bands:
        # Check outfiles
        outfile = allinone_rest_filename(thisband, version)
        if isfile(outfile) and not overwrite:
           print "File {0} exists. Use overwrite to overwrite it.".format(outfile)

        # Wavelength range
        wavebase = aio.allinone_wavebase(thisband)
        index_wave = np.searchsorted(wave, wavebase)
        nwave = index_wave[1] - index_wave[0]

        # Objects with redshift in the covered range
        index_obj = (np.where(np.logical_and((objs['Z'] > (_minwave/wave[index_wave[1]]-1.-0.001)), (objs['Z'] <= (_maxwave/wave[index_wave[0]]-1.+0.001)))))[0]
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

def allinone_rest_readin_band(band, version='DR7'):
    ''' Read in All_in_One (AIO) files given a band
    '''
    infile = allinone_rest_filename(band, version)
    if isfile(infile):
       print "Reading {0}.".format(infile)
       return (fitsio.read(infile))[0]
    else:
       raise IOError("Can't find {0}".format(infile))

def rest_allspec_readin(version='DR7'):
    ''' Read in all All_in_One (AIO) files and make a gigantic AIO array
    '''

    # Read in the catalog
    objs_ori = absorber_readin(version)
    nobj = objs_ori.size
   
    # Read in master wavelength grid
    master_wave = (aio.allinone_wave_readin())[0]['WAVE']
    master_loglam = np.log10(master_wave)
    nwave = master_wave.size

    # Initialization, nobj second dimension because of NMF convention 
    rest_allflux = np.zeros((nwave, nobj))
    rest_allivar = np.zeros((nwave, nobj))

    bands = _allinone_rest_bands
    for thisband in bands:
        data = allinone_rest_readin_band(thisband, version)
        index_wave = data['INDEX_WAVE']
        index_obj = data['INDEX_OBJ']
        rest_allflux[index_wave[0]:index_wave[1], index_obj] = data['FLUX']
        rest_allivar[index_wave[0]:index_wave[1], index_obj] = data['IVAR']

    return (master_wave, rest_allflux, rest_allivar)


