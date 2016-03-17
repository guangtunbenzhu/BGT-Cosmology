
""" 
__license__ = "MIT"
__author__ = "Guangtun Ben Zhu (BGT) @ Johns Hopkins University"
__startdate__ = "2016.03.11"
__name__ = "archespec"
__module__ = "archetype"

__lastdate__ = "2016.03.11"
__version__ = "0.01"
"""

from os.path import isfile, join
import numpy as np
from scipy.stats import nanmean, nanmedian

import fitsio
import lmfit

import datapath
import allinonespec as aio
import sdssspec as sdssspec
import cosmology as cosmo

_EPS = 1E-5
# prefixes
_allinone_observer_bands = ['OPTICAL']
_allinone_rest_bands     = ['OPTICAL']
_allinone_observer_fileprefix = 'AIO_Archetype_ObserverFrame_'
_allinone_rest_fileprefix     = 'AIO_Archetype_SDSSRestFrame_'
_archefile = 'Archetype_sample.fits'

_nbootstrap = 100

_minmaxwave = [3600., 10400.]
_contmask = np.array([[2200., 2249.88-7.],
                      [2260.78+6., 2297.58-10.],
                      [2297.58+6., 2324.21-7.],
                      [2344.21+6., 2365.36-7.],
                      [2396.36+6., 2422.56-7.],
                      [2425.14+6., 2470.97-7.],
                      [2471.09+6., 2510.00-7.],
                      [2511.00+6., 2576.88-7.],
                      [2626.45+6., 2796.35-7.],
                      [2803.53+6., 2852.96-7.],
                      [2852.96+6., 2900.]])
_oiimask = np.array([[3100., 3189.67-7.],
                     [3189.67+7., 3700.]])
_o3mask = np.array([[4750., 4863.-13.],
## _o3mask = np.array([[4920, 4959.-7.],
                    [4959.+6., 5007.-7.],
                    [5007.+7., 5040.]])

_zmin = 0.0
_zmax = 0.3
_zcorr = 0. # 10./3.E5 # redshift correction, 10 km/s

# 2/3/4/5 bins
def make_oiiewbins(zmin=_zmin, zmax=_zmax):
    """
    """
    nbin = 2+3+4+5
    oiiewmin = np.zeros(nbin)
    oiiewmax = np.zeros(nbin)
    oiiewbin = np.zeros(nbin)
    oiiewmin[0:2] = [_EPS, 50.0]
    oiiewmax[0:2] = [50.0, 200.]
    oiiewmin[2:2+3] = [_EPS, 40.0, 70.0]
    oiiewmax[2:2+3] = [40.0, 70.0, 200.]
    oiiewmin[5:5+4] = [_EPS, 30.0, 50.0, 80.0]
    oiiewmax[5:5+4] = [30.0, 50.0, 80.0, 200.]
    oiiewmin[9:9+5] = [_EPS, 25.0, 45.0, 60.0, 90.0]
    oiiewmax[9:9+5] = [25.0, 45.0, 60.0, 90.0, 200.]

    oiilummin = np.zeros(nbin)
    oiilummax = np.zeros(nbin)
    oiilumbin = np.zeros(nbin)
    oiilummin[0:2] = [40.0, 41.6]
    oiilummax[0:2] = [41.6, 43.5]
    oiilummin[2:2+3] = [40.0, 41.4, 41.8]
    oiilummax[2:2+3] = [41.4, 41.8, 43.5]
    oiilummin[5:5+4] = [40.0, 41.3, 41.6, 41.9]
    oiilummax[5:5+4] = [41.3, 41.6, 41.9, 43.5]
    oiilummin[9:9+5] = [40.0, 41.2, 41.5, 41.7, 42.0]
    oiilummax[9:9+5] = [41.2, 41.5, 41.7, 42.0, 43.5]

    # Calculate the medians
    objs_ori = elg_readin()
    vac_objs = elg_readin(vac=True)
    nobj = objs_ori.size
    zindex = (np.where(np.logical_and(np.logical_and(np.logical_and(
                 objs_ori['zGOOD']==1, objs_ori['Z']>zmin), objs_ori['Z']<zmax), objs_ori['CLASS']=='GALAXY')))[0]
    oiiew = vac_objs['OIIEW'][zindex]
    logoiilum = np.log10(vac_objs['OIILUM'][zindex])

    for i in np.arange(nbin):
        oiiewbin[i] = nanmedian(oiiew[((oiiew>oiiewmin[i]) & (oiiew<oiiewmax[i]))])
        oiilumbin[i] = nanmedian(logoiilum[((logoiilum>oiilummin[i]) & (logoiilum<oiilummax[i]))])

    return (oiiewmin, oiiewmax, oiiewbin, oiilummin, oiilummax, oiilumbin)

def arche_filename(vac=False):
    path = datapath.garching_path()
    if (not vac):
        return join(path, _archefile)
    else:
        return join(path, 'VAGC_'+_archefile)

def arche_readin(vac=False):
    infile = arche_filename(vac=vac)
    if isfile(infile):
        if (not vac):
            return fitsio.read(infile, ext=1)
        else:
            return (fitsio.read(infile, ext=1))[0]
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

    path1 = join(datapath.sdss_path(), 'specDR7')

    # check output files
    bands = _allinone_rest_bands
    for thisband in bands:
        # check outfiles
        outfile = allinone_rest_filename(thisband)
        if isfile(outfile) and not overwrite:
           print("File {0} exists. Use overwrite to overwrite it.".format(outfile))
           return -1
        # print "Will write into these files: {0}".format(outfile)

    # read in the elg catalog
    objs_ori = arche_readin()
    nobj = objs_ori.size

    # make a temporary new catalog
    objs_dtype = [('PLATE', 'i4'),
                  ('MJD', 'i4'),
                  ('FIBER', 'i4'),
                  ('RA', 'f8'),
                  ('DEC', 'f8'),
                  ('Z', 'f8')]
    objs = np.zeros(nobj, dtype=objs_dtype)
    objs['PLATE'] = objs_ori['PLATE']
    objs['MJD'] = objs_ori['MJD']
    objs['FIBER'] = objs_ori['FIBER']
    objs['RA'] = objs_ori['RA']
    objs['DEC'] = objs_ori['DEC']
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
    # pbar = ProgressBar(maxval=nobj).start()
    #for i in np.arange(10):
    for i in np.arange(nobj):
        # Progress bar
        # pbar.update(i)

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
    # pbar.finish()

    # write out
    print("Now I am writing everything out...")
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
           print("File {0} exists. Use overwrite to overwrite it.".format(outfile))
        # print "Will write into these files: {0}".format(outfile)

        # wavelength range
        wavebase = aio.allinone_wavebase(thisband)
        index_wave = np.searchsorted(wave, wavebase)
        nwave = index_wave[1] - index_wave[0]

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
       print("Reading {0}.".format(infile))
       return (fitsio.read(infile))[0]
    else:
       raise IOError("Can't find {0}".format(infile))

def rest_allspec_readin():

    # read in the archetype catalog
    objs_ori = arche_readin()
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

    master_wave = master_wave*(1.+_zcorr)
    return (master_wave, rest_allflux, rest_allivar)

def make_mask(wave, oii=False, o3=False):
    """
    """
    mask = np.zeros(wave.size, dtype='bool')
    if oii:
       for i in np.arange((_oiimask.shape)[0]):
           mask[(wave>_oiimask[i,0]) & (wave<_oiimask[i,1])] = True
    elif o3:
       for i in np.arange((_o3mask.shape)[0]):
           mask[(wave>_o3mask[i,0]) & (wave<_o3mask[i,1])] = True
    else:
       for i in np.arange((_contmask.shape)[0]):
           mask[(wave>_contmask[i,0]) & (wave<_contmask[i,1])] = True

    return mask

def calculate_continuum(loglam, flux, ivar, mask, polyorder=2):
    """
    """
    x = loglam[(mask) & (ivar>0)]
    y = flux[(mask) & (ivar>0)]
    if (x.size>0):
        z = np.polyfit(x, y, polyorder)
        p = np.poly1d(z)
        cont = p(loglam)
    else:
        cont = np.ones(loglam.shape)
    return cont

def calculate_continuum_powerlaw(loglam, flux, ivar, mask):
    """
    """
    x = loglam[(mask) & (ivar>0)]
    y = flux[(mask) & (ivar>0)]
    z = np.polyfit(x, y, polyorder)
    p = np.poly1d(z)
    cont = p(loglam)
    return cont

def value_add_elg(overwrite=False):
    """
    """
    # Check output file
    outfile = elg_filename(vac=True)
    if isfile(outfile) and not overwrite:
        print("File {0} exists. Set overwrite=True to overwrite it.".format(outfile))
        return -1
 
    Mpc_cm = 3.08568025E24
    objs_ori = elg_readin()
    nobj = objs_ori.size
    z = objs_ori['Z']

    (master_wave, rest_allflux, rest_allivar) = rest_allspec_readin()

    # OII luminosity and equivalent width
    oiilum = np.zeros(nobj)
    oiiew = np.zeros(nobj)

    index_oii  = np.searchsorted(master_wave, 3728.48)
    dnoiiwave = 10
    dwave = np.median(master_wave[index_oii-dnoiiwave:index_oii+dnoiiwave]-master_wave[index_oii-dnoiiwave-1:index_oii+dnoiiwave-1])
    print("dwave: {0}".format(dwave))
    oiisum = np.sum(rest_allflux[index_oii-dnoiiwave:index_oii+dnoiiwave, :]*(rest_allivar[index_oii-dnoiiwave:index_oii+dnoiiwave, :]>0), axis=0)*dwave 
    print("allfinite: {0}".format(np.count_nonzero(np.isfinite(oiisum))))
    oii_left = np.sum(rest_allflux[index_oii-25:index_oii-15, :]*(rest_allivar[index_oii-25:index_oii-15, :]>0), axis=0)/(25.-15.)
    oii_right = np.sum(rest_allflux[index_oii+15:index_oii+25, :]*(rest_allivar[index_oii+15:index_oii+25, :]>0), axis=0)/(25.-15.)
    oii_cont = (oii_left+oii_right)/2.
    oiiew = (oiisum-oii_cont*dwave)/oii_cont
    oiilum = (oiisum-oii_cont*dwave)*np.power(cosmo.luminosity_distance(z), 2)*4.*np.pi*np.power(Mpc_cm,2)*1E-17

    index_oiii  = np.searchsorted(master_wave, 5008.24)
    dnoiiiwave = 10
    dwave = np.median(master_wave[index_oiii-dnoiiiwave:index_oiii+dnoiiiwave]-master_wave[index_oiii-dnoiiiwave-1:index_oiii+dnoiiiwave-1])
    print("dwave: {0}".format(dwave))
    oiiisum = np.sum(rest_allflux[index_oiii-dnoiiiwave:index_oiii+dnoiiiwave, :]*(rest_allivar[index_oiii-dnoiiiwave:index_oiii+dnoiiiwave, :]>0), axis=0)*dwave 
    print("allfinite: {0}".format(np.count_nonzero(np.isfinite(oiiisum))))
    oiii_left = np.sum(rest_allflux[index_oiii-25:index_oiii-15, :]*(rest_allivar[index_oiii-25:index_oiii-15, :]>0), axis=0)/(25.-15.)
    oiii_right = np.sum(rest_allflux[index_oiii+15:index_oiii+25, :]*(rest_allivar[index_oiii+15:index_oiii+25, :]>0), axis=0)/(25.-15.)
    oiii_cont = (oiii_left+oiii_right)/2.
    oiiiew = (oiiisum-oiii_cont*dwave)/oiii_cont
    oiiilum = (oiiisum-oiii_cont*dwave)*np.power(cosmo.luminosity_distance(z), 2)*4.*np.pi*np.power(Mpc_cm,2)*1E-17

    index_hbeta  = np.searchsorted(master_wave, 4862.64)
    dnhbetawave = 10
    dwave = np.median(master_wave[index_hbeta-dnhbetawave:index_hbeta+dnhbetawave]-master_wave[index_hbeta-dnhbetawave-1:index_hbeta+dnhbetawave-1])
    print("dwave: {0}".format(dwave))
    hbetasum = np.sum(rest_allflux[index_hbeta-dnhbetawave:index_hbeta+dnhbetawave, :]*(rest_allivar[index_hbeta-dnhbetawave:index_hbeta+dnhbetawave, :]>0), axis=0)*dwave 
    print("allfinite: {0}".format(np.count_nonzero(np.isfinite(hbetasum))))
    hbeta_left = np.sum(rest_allflux[index_hbeta-25:index_hbeta-15, :]*(rest_allivar[index_hbeta-25:index_hbeta-15, :]>0), axis=0)/(25.-15.)
    hbeta_right = np.sum(rest_allflux[index_hbeta+15:index_hbeta+25, :]*(rest_allivar[index_hbeta+15:index_hbeta+25, :]>0), axis=0)/(25.-15.)
    hbeta_cont = (hbeta_left+hbeta_right)/2.
    hbetaew = (hbetasum-hbeta_cont*dwave)/hbeta_cont
    hbetalum = (hbetasum-hbeta_cont*dwave)*np.power(cosmo.luminosity_distance(z), 2)*4.*np.pi*np.power(Mpc_cm,2)*1E-17


    outstr_dtype = [('Z', 'f4', z.shape), 
                    ('OIILUM', 'f8', oiilum.shape), 
                    ('OIIEW', 'f8', oiiew.shape),
                    ('OIIILUM', 'f8', oiiilum.shape), 
                    ('OIIIEW', 'f8', oiiiew.shape),
                    ('HBETALUM', 'f8', hbetalum.shape), 
                    ('HBETAEW', 'f8', hbetaew.shape),
                    ]

    outstr  = np.array([(z, oiilum, oiiew, oiiilum, oiiiew, hbetalum, hbetaew)],
                         dtype=outstr_dtype)

    print("Write into file: {0}.".format(outfile))
    fits = fitsio.FITS(outfile, 'rw', clobber=overwrite)
    fits.write(outstr)
    fits.close()

def new_composite_engine(wave, flux, ivar, polyorder=2, oii=False, o3=False, bootstrap=False, nbootstrap=_nbootstrap):
    """All the composites should be made with this engine.
    - mean doesn't work for noisy data yet
    - mask is given by _contmask
    """
    loglam = np.log10(wave)
    nwave = wave.size
    nobj = flux.size/wave.size

    mask = make_mask(wave, oii=oii, o3=o3)
    masksize = np.count_nonzero(mask)
    if masksize>10: 
       x = loglam[mask]
       # Median, not entirely necessary
       obj_median = nanmedian(flux[mask, :], axis=0)
       y_median = flux/obj_median.reshape(1, nobj)
       norm_median = np.zeros(y_median.shape)
       for iobj in np.arange(nobj):
           continuum = calculate_continuum(loglam, flux[:,iobj], ivar[:,iobj], mask, polyorder)
           norm_median[:,iobj] = y_median[:, iobj]/continuum

       norm_median[ivar<=0] = np.nan

       # Bootstrapping:
       if bootstrap:
           median_norm_median = np.zeros((nwave, nbootstrap))
           mean_norm_median = np.zeros((nwave, nbootstrap))

           # Composite
           # pbar = ProgressBar(maxval=nbootstrap).start()
           for iboot in np.arange(nbootstrap):
               # pbar.update(iboot)
               index_boot = np.random.randint(0, nobj, size=nobj)
               median_norm_median_tmp = nanmedian(norm_median[:, index_boot], axis=1)
               mean_norm_median_tmp = nanmean(norm_median[:, index_boot], axis=1)
               # Median
               y = median_norm_median_tmp[mask]
               z = np.polyfit(x, y, polyorder)
               p = np.poly1d(z)
               continuum = p(loglam)
               median_norm_median[:, iboot] = median_norm_median_tmp/continuum

               # Mean
               y = mean_norm_median_tmp[mask]
               z = np.polyfit(x, y, polyorder)
               p = np.poly1d(z)
               continuum = p(loglam)
               mean_norm_median[:, iboot] = mean_norm_median_tmp/continuum
           # pbar.finish()

       # Regular
       else: 
           # Composite
           median_norm_median = nanmedian(norm_median, axis=1)
           mean_norm_median = nanmean(norm_median, axis=1)

           # Median
           y = median_norm_median[mask]
           z = np.polyfit(x, y, polyorder)
           p = np.poly1d(z)
           continuum = p(loglam)
           median_norm_median = median_norm_median/continuum

           # Mean
           y = mean_norm_median[mask]
           z = np.polyfit(x, y, polyorder)
           p = np.poly1d(z)
           continuum = p(loglam)
           mean_norm_median = mean_norm_median/continuum

    return (median_norm_median, mean_norm_median)

def new_feiimgii_composite(zmin=_zmin, zmax=_zmax, polyorder=3, bootstrap=False, nbootstrap=_nbootstrap):

    # Read in
    objs_ori = elg_readin()
    (master_wave, rest_allflux, rest_allivar) = rest_allspec_readin()
    master_loglam = np.log10(master_wave)

    #wave_pos = np.array([2200., 4050.])
    # Extended to 5200 to include [OIII] 5007 -- Guangtun, 06/08/2015
    wave_pos = np.array([2200., 5200.])
    # zmin<z<zmax; zGOOD==1; CLASS='GALAXY'
    zindex = (np.where(np.logical_and(np.logical_and(np.logical_and(
                 objs_ori['zGOOD']==1, objs_ori['Z']>zmin), objs_ori['Z']<zmax), objs_ori['CLASS']=='GALAXY')))[0]
    print(zindex.shape)

    rest_loc = np.searchsorted(master_wave, wave_pos)
    outwave = master_wave[rest_loc[0]:rest_loc[1]]
    outloglam = np.log10(outwave)

    tmpflux = rest_allflux[rest_loc[0]:rest_loc[1],zindex]
    tmpivar = rest_allivar[rest_loc[0]:rest_loc[1],zindex]

    (fluxmedian, fluxmean) = new_composite_engine(outwave, tmpflux, tmpivar, polyorder, bootstrap=bootstrap, nbootstrap=nbootstrap)
    (oiifluxmedian, oiifluxmean) = new_composite_engine(outwave, tmpflux, tmpivar, polyorder=2, oii=True, bootstrap=bootstrap, nbootstrap=nbootstrap)
    (oiiifluxmedian, oiiifluxmean) = new_composite_engine(outwave, tmpflux, tmpivar, polyorder=2, o3=True, bootstrap=bootstrap, nbootstrap=nbootstrap)
    
    return (outwave, fluxmedian, fluxmean, oiifluxmedian, oiifluxmean, oiiifluxmedian, oiiifluxmean)

def save_feiimgii_composite(bootstrap=False, nbootstrap=_nbootstrap, overwrite=False):
    """
    """

    outfile = feiimgii_composite_filename(bootstrap=bootstrap)
    if isfile(outfile) and not overwrite:
        print("File {0} exists. Set overwrite=True to overwrite it.".format(outfile))
        return -1
 
    (outwave, fluxmedian, fluxmean, oiifluxmedian, oiifluxmean, oiiifluxmedian, oiiifluxmean) = new_feiimgii_composite(bootstrap=bootstrap, nbootstrap=nbootstrap)
    nwave = outwave.size
    outstr_dtype = [('WAVE', 'f4', outwave.shape), 
                    ('FLUXMEDIAN', 'f4', fluxmedian.shape), 
                    ('FLUXMEAN', 'f4', fluxmean.shape),
                    ('OII_FLUXMEDIAN', 'f4', oiifluxmedian.shape), 
                    ('OII_FLUXMEAN', 'f4', oiifluxmean.shape),
                    ('OIII_FLUXMEDIAN', 'f4', oiiifluxmedian.shape), 
                    ('OIII_FLUXMEAN', 'f4', oiiifluxmean.shape)]

    outstr  = np.array([(outwave, fluxmedian, fluxmean, oiifluxmedian, oiifluxmean, oiiifluxmedian, oiiifluxmean)],
                         dtype=outstr_dtype)

    print("Write into file: {0}.".format(outfile))
    fits = fitsio.FITS(outfile, 'rw', clobber=overwrite)
    fits.write(outstr)
    fits.close()

# For OII dependence, let's duplicate these two routines but remember to double check if the original two routines change
def new_feiimgii_composite_binoii(zmin=_zmin, zmax=_zmax, polyorder=3, bootstrap=False, nbootstrap=_nbootstrap):

    # Read in
    objs_ori = elg_readin()
    vac_objs = elg_readin(vac=True)
    (master_wave, rest_allflux, rest_allivar) = rest_allspec_readin()
    master_loglam = np.log10(master_wave)

    #wave_pos = np.array([2200., 4050.])
    # Extended to 5200 to include [OIII] 5007 -- Guangtun, 06/08/2015
    wave_pos = np.array([2200., 5200.])
    # zmin<z<zmax; zGOOD==1; CLASS='GALAXY'
    zindex = (np.where(np.logical_and(np.logical_and(np.logical_and(
                 objs_ori['zGOOD']==1, objs_ori['Z']>zmin), objs_ori['Z']<zmax), objs_ori['CLASS']=='GALAXY')))[0]
    print(zindex.shape)

    rest_loc = np.searchsorted(master_wave, wave_pos)
    outwave = master_wave[rest_loc[0]:rest_loc[1]]
    outloglam = np.log10(outwave)

    tmpflux = rest_allflux[rest_loc[0]:rest_loc[1],zindex]
    tmpivar = rest_allivar[rest_loc[0]:rest_loc[1],zindex]

    oiiew = vac_objs['OIIEW'][zindex]
    logoiilum = np.log10(vac_objs['OIILUM'][zindex])
    oiiewmin, oiiewmax, oiiewbin, oiilummin, oiilummax, oiilumbin = make_oiiewbins()

    for i in np.arange(oiiewmin.size):
        ewbin = (np.where(np.logical_and(oiiew>oiiewmin[i], oiiew<oiiewmax[i])))[0]
        oii_tmpflux = tmpflux[:,ewbin]
        oii_tmpivar = tmpivar[:,ewbin]
        (ewtmp_fluxmedian, ewtmp_fluxmean) = new_composite_engine(outwave, oii_tmpflux, oii_tmpivar, polyorder, bootstrap=bootstrap, nbootstrap=nbootstrap)
        (ewtmp_oiifluxmedian, ewtmp_oiifluxmean) = new_composite_engine(outwave, oii_tmpflux, oii_tmpivar, polyorder=2, oii=True, bootstrap=bootstrap, nbootstrap=nbootstrap)
        (ewtmp_oiiifluxmedian, ewtmp_oiiifluxmean) = new_composite_engine(outwave, oii_tmpflux, oii_tmpivar, polyorder=2, o3=True, bootstrap=bootstrap, nbootstrap=nbootstrap)
 
        lumbin = (np.where(np.logical_and(logoiilum>oiilummin[i], logoiilum<oiilummax[i])))[0]
        oii_tmpflux = tmpflux[:,lumbin]
        oii_tmpivar = tmpivar[:,lumbin]
        (lumtmp_fluxmedian, lumtmp_fluxmean) = new_composite_engine(outwave, oii_tmpflux, oii_tmpivar, polyorder, bootstrap=bootstrap, nbootstrap=nbootstrap)
        (lumtmp_oiifluxmedian, lumtmp_oiifluxmean) = new_composite_engine(outwave, oii_tmpflux, oii_tmpivar, polyorder=2, oii=True, bootstrap=bootstrap, nbootstrap=nbootstrap)
        (lumtmp_oiiifluxmedian, lumtmp_oiiifluxmean) = new_composite_engine(outwave, oii_tmpflux, oii_tmpivar, polyorder=2, o3=True, bootstrap=bootstrap, nbootstrap=nbootstrap)
 
        if (i == 0):
            outshape = ewtmp_fluxmedian.shape+oiiewmin.shape
            ew_fluxmedian = np.zeros(outshape)
            ew_fluxmean = np.zeros(outshape)
            ew_oiifluxmedian = np.zeros(outshape)
            ew_oiifluxmean = np.zeros(outshape)
            ew_oiiifluxmedian = np.zeros(outshape)
            ew_oiiifluxmean = np.zeros(outshape)
            lum_fluxmedian = np.zeros(outshape)
            lum_fluxmean = np.zeros(outshape)
            lum_oiifluxmedian = np.zeros(outshape)
            lum_oiifluxmean = np.zeros(outshape)
            lum_oiiifluxmedian = np.zeros(outshape)
            lum_oiiifluxmean = np.zeros(outshape)
        print("outshape: {0}".format(ew_fluxmedian.shape))
        if (not bootstrap):
            ew_fluxmedian[:,i] = ewtmp_fluxmedian
            ew_fluxmean[:,i] = ewtmp_fluxmean
            ew_oiifluxmedian[:,i] = ewtmp_oiifluxmedian
            ew_oiifluxmean[:,i] = ewtmp_oiifluxmean
            ew_oiiifluxmedian[:,i] = ewtmp_oiiifluxmedian
            ew_oiiifluxmean[:,i] = ewtmp_oiiifluxmean
            lum_fluxmedian[:,i] = lumtmp_fluxmedian
            lum_fluxmean[:,i] = lumtmp_fluxmean
            lum_oiifluxmedian[:,i] = lumtmp_oiifluxmedian
            lum_oiifluxmean[:,i] = lumtmp_oiifluxmean
            lum_oiiifluxmedian[:,i] = lumtmp_oiiifluxmedian
            lum_oiiifluxmean[:,i] = lumtmp_oiiifluxmean
        else:
            ew_fluxmedian[:,:,i] = ewtmp_fluxmedian
            ew_fluxmean[:,:,i] = ewtmp_fluxmean
            ew_oiifluxmedian[:,:,i] = ewtmp_oiifluxmedian
            ew_oiifluxmean[:,:,i] = ewtmp_oiifluxmean
            ew_oiiifluxmedian[:,:,i] = ewtmp_oiiifluxmedian
            ew_oiiifluxmean[:,:,i] = ewtmp_oiiifluxmean
            lum_fluxmedian[:,:,i] = lumtmp_fluxmedian
            lum_fluxmean[:,:,i] = lumtmp_fluxmean
            lum_oiifluxmedian[:,:,i] = lumtmp_oiifluxmedian
            lum_oiifluxmean[:,:,i] = lumtmp_oiifluxmean
            lum_oiiifluxmedian[:,:,i] = lumtmp_oiiifluxmedian
            lum_oiiifluxmean[:,:,i] = lumtmp_oiiifluxmean

    return (outwave, ew_fluxmedian, ew_fluxmean, ew_oiifluxmedian, ew_oiifluxmean, ew_oiiifluxmedian, ew_oiiifluxmean, 
                    lum_fluxmedian,lum_fluxmean,lum_oiifluxmedian,lum_oiifluxmean,lum_oiiifluxmedian,lum_oiiifluxmean)

def save_feiimgii_composite_binoii(bootstrap=False, nbootstrap=_nbootstrap, overwrite=False):
    """
    """

    outfile = feiimgii_composite_filename(bootstrap=bootstrap, binoii=True)
    if ((isfile(outfile)) and (not overwrite)):
        print("File {0} exists. Set overwrite=True to overwrite it.".format(outfile))
        return -1
 
    oiiewmin, oiiewmax, oiiewbin, oiilummin, oiilummax, oiilumbin = make_oiiewbins()

    outwave, ew_fluxmedian, ew_fluxmean, ew_oiifluxmedian, ew_oiifluxmean, ew_oiiifluxmedian, ew_oiiifluxmean, \
             lum_fluxmedian,lum_fluxmean,lum_oiifluxmedian,lum_oiifluxmean,lum_oiiifluxmedian,lum_oiiifluxmean = \
             new_feiimgii_composite_binoii(bootstrap=bootstrap, nbootstrap=nbootstrap)

    nwave = outwave.size
    outstr_dtype = [('WAVE', 'f4', outwave.shape), 
                    ('EWFLUXMEDIAN', 'f4', ew_fluxmedian.shape), 
                    #('EWFLUXMEAN', 'f4', ew_fluxmean.shape),
                    ('EWOII_FLUXMEDIAN', 'f4', ew_oiifluxmedian.shape), 
                    #('EWOII_FLUXMEAN', 'f4', ew_oiifluxmean.shape),
                    ('EWOIII_FLUXMEDIAN', 'f4', ew_oiiifluxmedian.shape), 
                    #('EWOIII_FLUXMEAN', 'f4', ew_oiiifluxmean.shape),
                    ('LUMFLUXMEDIAN', 'f4', lum_fluxmedian.shape), 
                    #('LUMFLUXMEAN', 'f4', lum_fluxmean.shape),
                    ('LUMOII_FLUXMEDIAN', 'f4', lum_oiifluxmedian.shape), 
                    #('LUMOII_FLUXMEAN', 'f4', lum_oiifluxmean.shape),
                    ('LUMOIII_FLUXMEDIAN', 'f4', lum_oiiifluxmedian.shape), 
                    #('LUMOIII_FLUXMEAN', 'f4', lum_oiiifluxmean.shape),
                    ('OIIEWMIN', 'f4', oiiewmin.shape),
                    ('OIIEWMAX', 'f4', oiiewmax.shape),
                    ('OIIEWBIN', 'f4', oiiewbin.shape),
                    ('OIILUMMIN', 'f4', oiilummin.shape),
                    ('OIILUMMAX', 'f4', oiilummax.shape),
                    ('OIILUMBIN', 'f4', oiilumbin.shape)]

    outstr  = np.array([(outwave, ew_fluxmedian, ew_oiifluxmedian, ew_oiiifluxmedian, lum_fluxmedian, lum_oiifluxmedian, lum_oiiifluxmedian, 
                         oiiewmin, oiiewmax, oiiewbin, oiilummin, oiilummax, oiilumbin)],
                         dtype=outstr_dtype)

    print("Write into file: {0}.".format(outfile))
    fits = fitsio.FITS(outfile, 'rw', clobber=overwrite)
    fits.write(outstr)
    fits.close()


#def make_model(lines):
#    """Make a model for a normalized spectrum 
#    In logarithmic space
#    """
#
#    dloglam = 1E-4 # or 69./3E5/np.log(10.)
#    left_bound = 10.*dloglam # pixels
#    right_bound = 5.*dloglam # pixels
#    width = 200./3E5/np.log(10.) # Delta_v/c in unit of log_10(lambda), 200 km/s
#    min_width = 50./3E5/np.log(10.) # 
#    max_width = 2000./3E5/np.log(10.) #
#    namp = 10 # maximum amplitude
#
#    full_model = {}
#
#    # Underlying quadratic model
#    tmp_prefix = 'Quadratic_'
#    full_model[0] = lmfit.models.QuadraticModel(prefix=tmp_prefix)
#
#    pars = full_model[0].make_params()
#    pars[tmp_prefix+'a'].set(0., min=-0.1, max=0.1)
#    pars[tmp_prefix+'b'].set(0., min=-0.5, max=0.5)
#    pars[tmp_prefix+'c'].set(1., min=0.9,  max=1.1)
#
#    # Line Gaussian model
#    # Line: 'ELEMENT', 'WAVE', 'EW', 'SIGN'
#    nlines = lines.size
#    if nlines==0: return (full_model[0], pars)
#
#    for (iline, this_line) in zip(np.arange(nlines)+len(full_model), lines):
#         tmp_prefix = this_line['ELEMENT']+'_'+'{0:02d}'.format(iline)+'_'
#         full_model[iline] = lmfit.models.GaussianModel(prefix=tmp_prefix)
# 
#         pars.update(full_model[iline].make_params())
#         tmp_wave = this_line['WAVE']-1.
#         tmp_loglam = np.log10(this_line['WAVE']-1.)
#
#         tmp_left = np.log10(this_line['WAVE']-left_bound)
#         tmp_right = np.log10(this_line['WAVE']-right_bound)
#         pars[tmp_prefix+'center'].set(tmp_loglam, min=tmp_left, max=tmp_right)
#         pars[tmp_prefix+'sigma'].set(width, min=min_width, max=max_width)
#
#         tmp_sign = this_line['SIGN']
#         tmp_amp = tmp_sign*this_line['EW']/tmp_wave/np.log(10.)
#         if tmp_sign>0:
#            pars[tmp_prefix+'amplitude'].set(tmp_amp, min=0, max=tmp_amp*namp)
#         else:
#            pars[tmp_prefix+'amplitude'].set(tmp_amp, min=tmp_amp*namp, max=0)
#
#    model = full_model[0]
#    for imod in np.arange(len(full_model)-1)+1:
#        model = model+full_model[imod]
#
#    return (model, pars)
#
#
# All stuff below must be obsolete
# The new one is new_feiimgii_composite
#def feiimgii_composite(zmin=0.6, zmax=1.2):

    # Read in
#    objs_ori = elg_readin()
#    (master_wave, rest_allflux, rest_allivar) = rest_allspec_readin()
#    master_loglam = np.log10(master_wave)
#
#    wave_pos = np.array([2200., 4050.])
#    #zmin = _minmaxwave[0]/wave_pos[0]-1.
#    #zmax = _minmaxwave[1]/wave_pos[1]-1.
#    # zmin<z<zmax; zGOOD==1; CLASS='GALAXY'
#    zindex = (np.where(np.logical_and(np.logical_and(np.logical_and(
#                 objs_ori['zGOOD']==1, objs_ori['Z']>zmin), objs_ori['Z']<zmax), objs_ori['CLASS']=='GALAXY')))[0]
#
#    rest_loc = np.searchsorted(master_wave, wave_pos)
#    outwave = master_wave[rest_loc[0]:rest_loc[1]]
#    outloglam = np.log10(outwave)
#
#    tmpflux = rest_allflux[rest_loc[0]:rest_loc[1],zindex]
#    tmpivar = rest_allivar[rest_loc[0]:rest_loc[1],zindex]
#    fluxmean = np.zeros((tmpflux.shape)[0])
#    #fluxmean = np.average(tmpflux, axis=1, weights=tmpivar.astype(bool))
#    fluxmedian = np.zeros((tmpflux.shape)[0])
#    fluxflag = np.ones(fluxmedian.size)
#    for i in np.arange((tmpflux.shape)[0]):
#        iuse = (np.where(tmpivar[i,:]>0))[0]
#        fluxmedian[i] = np.median(tmpflux[i,iuse])
#        fluxmean[i] = np.mean(tmpflux[i,iuse])
#
#    # Mask out useless wavelength ranges
#    # left 2300
#    wave_pos = np.array([2200.])
#    rest_loc = np.searchsorted(outwave, wave_pos)
#    fluxflag[0:rest_loc[0]] = 0
#    # Fe II 2350
#    wave_pos = np.array([2330., 2420])
#    rest_loc = np.searchsorted(outwave, wave_pos)
#    fluxflag[rest_loc[0]:rest_loc[1]] = 0.
#    # Fe II 2600
#    wave_pos = np.array([2570., 2640])
#    rest_loc = np.searchsorted(outwave, wave_pos)
#    fluxflag[rest_loc[0]:rest_loc[1]] = 0.
#    # Mg II 2800
#    wave_pos = np.array([2770., 2820])
#    rest_loc = np.searchsorted(outwave, wave_pos)
#    fluxflag[rest_loc[0]:rest_loc[1]] = 0.
#    # Mg I 2853
#    wave_pos = np.array([2843., 2863])
#    rest_loc = np.searchsorted(outwave, wave_pos)
#    fluxflag[rest_loc[0]:rest_loc[1]] = 0.
#    # right 2900
#    wave_pos = np.array([2900.])
#    rest_loc = np.searchsorted(outwave, wave_pos)
#    fluxflag[rest_loc[0]:] = 0.
#
#    imask = (np.where(fluxflag>0.))[0]
#    if imask.size>10: 
#       x = outloglam[imask]
#       # Mean
#       y = fluxmean[imask]
#       z = np.polyfit(x, y, 3)
#       p = np.poly1d(z)
#       continuum = p(outloglam)
#       norm_fluxmean = fluxmean/continuum
#       # Median
#       y = fluxmedian[imask]
#       z = np.polyfit(x, y, 3)
#       p = np.poly1d(z)
#       continuum = p(outloglam)
#       norm_fluxmedian = fluxmedian/continuum
#   
#    return (outwave, fluxmean, fluxmedian, norm_fluxmean, norm_fluxmedian)
#
#def make_model(lines):
#    """Make a model for a normalized spectrum 
#    In logarithmic space
#    """
#
#    dloglam = 1E-4 # or 69./3E5/np.log(10.)
#    left_bound = 10.*dloglam # pixels
#    right_bound = 5.*dloglam # pixels
#    width = 200./3E5/np.log(10.) # Delta_v/c in unit of log_10(lambda), 200 km/s
#    min_width = 50./3E5/np.log(10.) # 
#    max_width = 2000./3E5/np.log(10.) #
#    namp = 10 # maximum amplitude
#
#    full_model = {}
#
#    # Underlying quadratic model
#    tmp_prefix = 'Quadratic_'
#    full_model[0] = lmfit.models.QuadraticModel(prefix=tmp_prefix)
#
#    pars = full_model[0].make_params()
#    pars[tmp_prefix+'a'].set(0., min=-0.1, max=0.1)
#    pars[tmp_prefix+'b'].set(0., min=-0.5, max=0.5)
#    pars[tmp_prefix+'c'].set(1., min=0.9,  max=1.1)
#
#    # Line Gaussian model
#    # Line: 'ELEMENT', 'WAVE', 'EW', 'SIGN'
#    nlines = lines.size
#    if nlines==0: return (full_model[0], pars)
#
#    for (iline, this_line) in zip(np.arange(nlines)+len(full_model), lines):
#         tmp_prefix = this_line['ELEMENT']+'_'+'{0:02d}'.format(iline)+'_'
#         full_model[iline] = lmfit.models.GaussianModel(prefix=tmp_prefix)
# 
#         pars.update(full_model[iline].make_params())
#         tmp_wave = this_line['WAVE']-1.
#         tmp_loglam = np.log10(this_line['WAVE']-1.)
#
#         tmp_left = np.log10(this_line['WAVE']-left_bound)
#         tmp_right = np.log10(this_line['WAVE']-right_bound)
#         pars[tmp_prefix+'center'].set(tmp_loglam, min=tmp_left, max=tmp_right)
#         pars[tmp_prefix+'sigma'].set(width, min=min_width, max=max_width)
#
#         tmp_sign = this_line['SIGN']
#         tmp_amp = tmp_sign*this_line['EW']/tmp_wave/np.log(10.)
#         if tmp_sign>0:
#            pars[tmp_prefix+'amplitude'].set(tmp_amp, min=0, max=tmp_amp*namp)
#         else:
#            pars[tmp_prefix+'amplitude'].set(tmp_amp, min=tmp_amp*namp, max=0)
#
#    model = full_model[0]
#    for imod in np.arange(len(full_model)-1)+1:
#        model = model+full_model[imod]
#
#    return (model, pars)
#
#def line_property(loglam, flux, lines, npixels=15):
#    """Measure line properties in a normalized spectrum in the rest frame:
#    Total equivalent width: REW
#    Velocity profile: REW(velocity)/REW(total)
#    """
#    
#    nlines = lines.size
#    ew_profile = np.zeros(nlines, dtype=[('WAVE', '({0},)f4'.format(npixels)), ('VEL', '({0},)f4'.format(npixels)), ('EW', '({0},)f4'.format(npixels))])
#    for (iline, this_line) in zip(np.arange(nlines), lines):
#        tmp_loglam0 = np.log10(this_line['WAVE'])
#        tmp_left = np.log10(this_line['WAVELEFT'])
#        rest_loc = np.searchsorted(loglam, tmp_left)
#        #print(rest_loc)
#        #print(np.cumsum(flux[rest_loc:(rest_loc+npixels)]))
#        ew_profile[iline]['EW'][:] = np.cumsum(flux[rest_loc:(rest_loc+npixels)])
#        #print(ew_profile[iline]['EW'])
#        ew_profile[iline]['VEL'][:] = (loglam[rest_loc:(rest_loc+npixels)]-tmp_loglam0)*np.log(10.)*3E5
#        ew_profile[iline]['WAVE'][:] = np.power(10, loglam[rest_loc:(rest_loc+npixels)])
#
#    return ew_profile
#
#def speclines(region='2800'):
#    if region == '2800':
#       nlines = 2
#       lines = zeros(nlines, dtype=[('SIGN', 'i'),('ELEMENT','S20'),('WAVE','f4'),('EW','f4'), ('WAVELEFT', 'f4')])
#       lines[0] = (-1, 'MgII', 2796.35, 2., 2789.)
#       lines[1] = (-1, 'MgII', 2803.53, 2., 2798.)
