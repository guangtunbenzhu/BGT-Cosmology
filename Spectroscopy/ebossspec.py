
""" 

"""

from __future__ import division

from os.path import isfile, join
import numpy as np
import fitsio
import datapath
import allinonespec as aio
import sdssspec as sdssspec
from scipy.stats import nanmean, nanmedian
from progressbar import ProgressBar
import lmfit


# prefixes
_allinone_observer_bands = ['OPTICAL']
_allinone_rest_bands     = ['NUV', 'OPTICAL']
_allinone_observer_fileprefix = 'AIO_ELG_eBOSS_ObserverFrame_'
_allinone_rest_fileprefix     = 'AIO_ELG_eBOSS_SDSSRestFrame_'
_elgfile = 'spAll-ELG-v5.4-zQ.fits'
_minmaxwave = [3600., 10400.]
_contmask = np.array([[2200., 2249.88-7.],
                      [2260.78+7., 2297.58-7.],
                      [2297.58+7., 2324.21-7.],
                      [2396.36+7., 2422.56-7.],
                      [2425.14+7., 2470.97-7.],
                      [2471.09+7., 2510.00-7.],
                      [2511.00+7., 2576.88-7.],
                      [2626.45+7., 2796.35-7.],
                      [2803.53+7., 2852.96-7.],
                      [2852.96+7., 2900.]])
_oiimask = np.array([[3100., 3189.67-7.],
                     [3189.67+7., 3700.]])

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

def make_mask(wave, oii=False):
    """
    """
    mask = np.zeros(wave.size, dtype='bool')
    if oii:
       for i in np.arange((_oiimask.shape)[0]):
           mask[(wave>_oiimask[i,0]) & (wave<_oiimask[i,1])] = True
    else:
       for i in np.arange((_contmask.shape)[0]):
           mask[(wave>_contmask[i,0]) & (wave<_contmask[i,1])] = True

    return mask

def calculate_continuum(loglam, flux, ivar, mask, polyorder=2):
    """
    """
    x = loglam[(mask) & (ivar>0)]
    y = flux[(mask) & (ivar>0)]
    z = np.polyfit(x, y, polyorder)
    p = np.poly1d(z)
    cont = p(loglam)
    return cont

def new_composite_engine(wave, flux, ivar, polyorder=2, oii=False):
    """All the composites should be made with this engine.
    - mean doesn't work for noisy data yet
    - mask is given by _contmask
    """
    loglam = np.log10(wave)
    nwave = wave.size
    nobj = flux.size/wave.size

    mask = make_mask(wave, oii=oii)
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


def composite_engine(loglam, flux, ivar, mask, polyorder=2):
    """All the composites should be made with this engine.
    - ivar is not being used for now.
    - mask is given by _contmask
    """

    nwave = loglam.size
    nobj = flux.size/loglam.size

    # mask = make_mask(loglam, ivar)
    imask = (np.where(mask==1))[0]
    if imask.size>10: 
       
       x = loglam[imask]
       # Mean
       obj_median = nanmedian(flux[imask, :], axis=0)
       y_median = flux/obj_median.reshape(1, nobj)
       norm_median = np.zeros(y_median.shape)
       for iobj in np.arange(nobj):
           y = y_median[imask, iobj]
           z = np.polyfit(x, y, polyorder)
           p = np.poly1d(z)
           continuum = p(loglam)
           norm_median[:,iobj] = y_median[:, iobj]/continuum

       # Composite
       median_norm_median = nanmedian(norm_median, axis=1)
       mean_norm_median = nanmean(norm_median, axis=1)

       # Median
       y = median_norm_median[imask]
       z = np.polyfit(x, y, polyorder)
       p = np.poly1d(z)
       continuum = p(loglam)
       median_norm_median = median_norm_median/continuum

       # Mean
       y = mean_norm_median[imask]
       z = np.polyfit(x, y, polyorder)
       p = np.poly1d(z)
       continuum = p(loglam)
       mean_norm_median = mean_norm_median/continuum

    return (median_norm_median, mean_norm_median)

def new_feiimgii_composite(zmin=0.6, zmax=1.2, polyorder=3):

    # Read in
    objs_ori = elg_readin()
    (master_wave, rest_allflux, rest_allivar) = rest_allspec_readin()
    master_loglam = np.log10(master_wave)

    wave_pos = np.array([2200., 4050.])
    # zmin<z<zmax; zGOOD==1; CLASS='GALAXY'
    zindex = (np.where(np.logical_and(np.logical_and(np.logical_and(
                 objs_ori['zGOOD']==1, objs_ori['Z']>zmin), objs_ori['Z']<zmax), objs_ori['CLASS']=='GALAXY')))[0]
    print zindex.shape

    rest_loc = np.searchsorted(master_wave, wave_pos)
    outwave = master_wave[rest_loc[0]:rest_loc[1]]
    outloglam = np.log10(outwave)

    tmpflux = rest_allflux[rest_loc[0]:rest_loc[1],zindex]
    tmpivar = rest_allivar[rest_loc[0]:rest_loc[1],zindex]

    (fluxmedian, fluxmean) = new_composite_engine(outwave, tmpflux, tmpivar, polyorder)
    (oiifluxmedian, oiifluxmean) = new_composite_engine(outwave, tmpflux, tmpivar, polyorder=2, oii=True)
    
    return (outwave, fluxmedian, fluxmean, oiifluxmedian, oiifluxmean)

def make_model(lines):
    """Make a model for a normalized spectrum 
    In logarithmic space
    """

    dloglam = 1E-4 # or 69./3E5/np.log(10.)
    left_bound = 10.*dloglam # pixels
    right_bound = 5.*dloglam # pixels
    width = 200./3E5/np.log(10.) # Delta_v/c in unit of log_10(lambda), 200 km/s
    min_width = 50./3E5/np.log(10.) # 
    max_width = 2000./3E5/np.log(10.) #
    namp = 10 # maximum amplitude

    full_model = {}

    # Underlying quadratic model
    tmp_prefix = 'Quadratic_'
    full_model[0] = lmfit.models.QuadraticModel(prefix=tmp_prefix)

    pars = full_model[0].make_params()
    pars[tmp_prefix+'a'].set(0., min=-0.1, max=0.1)
    pars[tmp_prefix+'b'].set(0., min=-0.5, max=0.5)
    pars[tmp_prefix+'c'].set(1., min=0.9,  max=1.1)

    # Line Gaussian model
    # Line: 'ELEMENT', 'WAVE', 'EW', 'SIGN'
    nlines = lines.size
    if nlines==0: return (full_model[0], pars)

    for (iline, this_line) in zip(np.arange(nlines)+len(full_model), lines):
         tmp_prefix = this_line['ELEMENT']+'_'+'{0:02d}'.format(iline)+'_'
         full_model[iline] = lmfit.models.GaussianModel(prefix=tmp_prefix)
 
         pars.update(full_model[iline].make_params())
         tmp_wave = this_line['WAVE']-1.
         tmp_loglam = np.log10(this_line['WAVE']-1.)

         tmp_left = np.log10(this_line['WAVE']-left_bound)
         tmp_right = np.log10(this_line['WAVE']-right_bound)
         pars[tmp_prefix+'center'].set(tmp_loglam, min=tmp_left, max=tmp_right)
         pars[tmp_prefix+'sigma'].set(width, min=min_width, max=max_width)

         tmp_sign = this_line['SIGN']
         tmp_amp = tmp_sign*this_line['EW']/tmp_wave/np.log(10.)
         if tmp_sign>0:
            pars[tmp_prefix+'amplitude'].set(tmp_amp, min=0, max=tmp_amp*namp)
         else:
            pars[tmp_prefix+'amplitude'].set(tmp_amp, min=tmp_amp*namp, max=0)

    model = full_model[0]
    for imod in np.arange(len(full_model)-1)+1:
        model = model+full_model[imod]

    return (model, pars)


def feiimgii_composite(zmin=0.6, zmax=1.2):

    # Read in
    objs_ori = elg_readin()
    (master_wave, rest_allflux, rest_allivar) = rest_allspec_readin()
    master_loglam = np.log10(master_wave)

    wave_pos = np.array([2200., 4050.])
    #zmin = _minmaxwave[0]/wave_pos[0]-1.
    #zmax = _minmaxwave[1]/wave_pos[1]-1.
    # zmin<z<zmax; zGOOD==1; CLASS='GALAXY'
    zindex = (np.where(np.logical_and(np.logical_and(np.logical_and(
                 objs_ori['zGOOD']==1, objs_ori['Z']>zmin), objs_ori['Z']<zmax), objs_ori['CLASS']=='GALAXY')))[0]

    rest_loc = np.searchsorted(master_wave, wave_pos)
    outwave = master_wave[rest_loc[0]:rest_loc[1]]
    outloglam = np.log10(outwave)

    tmpflux = rest_allflux[rest_loc[0]:rest_loc[1],zindex]
    tmpivar = rest_allivar[rest_loc[0]:rest_loc[1],zindex]
    fluxmean = np.zeros((tmpflux.shape)[0])
    #fluxmean = np.average(tmpflux, axis=1, weights=tmpivar.astype(bool))
    fluxmedian = np.zeros((tmpflux.shape)[0])
    fluxflag = np.ones(fluxmedian.size)
    for i in np.arange((tmpflux.shape)[0]):
        iuse = (np.where(tmpivar[i,:]>0))[0]
        fluxmedian[i] = np.median(tmpflux[i,iuse])
        fluxmean[i] = np.mean(tmpflux[i,iuse])

    # Mask out useless wavelength ranges
    # left 2300
    wave_pos = np.array([2200.])
    rest_loc = np.searchsorted(outwave, wave_pos)
    fluxflag[0:rest_loc[0]] = 0
    # Fe II 2350
    wave_pos = np.array([2330., 2420])
    rest_loc = np.searchsorted(outwave, wave_pos)
    fluxflag[rest_loc[0]:rest_loc[1]] = 0.
    # Fe II 2600
    wave_pos = np.array([2570., 2640])
    rest_loc = np.searchsorted(outwave, wave_pos)
    fluxflag[rest_loc[0]:rest_loc[1]] = 0.
    # Mg II 2800
    wave_pos = np.array([2770., 2820])
    rest_loc = np.searchsorted(outwave, wave_pos)
    fluxflag[rest_loc[0]:rest_loc[1]] = 0.
    # Mg I 2853
    wave_pos = np.array([2843., 2863])
    rest_loc = np.searchsorted(outwave, wave_pos)
    fluxflag[rest_loc[0]:rest_loc[1]] = 0.
    # right 2900
    wave_pos = np.array([2900.])
    rest_loc = np.searchsorted(outwave, wave_pos)
    fluxflag[rest_loc[0]:] = 0.

    imask = (np.where(fluxflag>0.))[0]
    if imask.size>10: 
       x = outloglam[imask]
       # Mean
       y = fluxmean[imask]
       z = np.polyfit(x, y, 3)
       p = np.poly1d(z)
       continuum = p(outloglam)
       norm_fluxmean = fluxmean/continuum
       # Median
       y = fluxmedian[imask]
       z = np.polyfit(x, y, 3)
       p = np.poly1d(z)
       continuum = p(outloglam)
       norm_fluxmedian = fluxmedian/continuum
   
    return (outwave, fluxmean, fluxmedian, norm_fluxmean, norm_fluxmedian)

def make_model(lines):
    """Make a model for a normalized spectrum 
    In logarithmic space
    """

    dloglam = 1E-4 # or 69./3E5/np.log(10.)
    left_bound = 10.*dloglam # pixels
    right_bound = 5.*dloglam # pixels
    width = 200./3E5/np.log(10.) # Delta_v/c in unit of log_10(lambda), 200 km/s
    min_width = 50./3E5/np.log(10.) # 
    max_width = 2000./3E5/np.log(10.) #
    namp = 10 # maximum amplitude

    full_model = {}

    # Underlying quadratic model
    tmp_prefix = 'Quadratic_'
    full_model[0] = lmfit.models.QuadraticModel(prefix=tmp_prefix)

    pars = full_model[0].make_params()
    pars[tmp_prefix+'a'].set(0., min=-0.1, max=0.1)
    pars[tmp_prefix+'b'].set(0., min=-0.5, max=0.5)
    pars[tmp_prefix+'c'].set(1., min=0.9,  max=1.1)

    # Line Gaussian model
    # Line: 'ELEMENT', 'WAVE', 'EW', 'SIGN'
    nlines = lines.size
    if nlines==0: return (full_model[0], pars)

    for (iline, this_line) in zip(np.arange(nlines)+len(full_model), lines):
         tmp_prefix = this_line['ELEMENT']+'_'+'{0:02d}'.format(iline)+'_'
         full_model[iline] = lmfit.models.GaussianModel(prefix=tmp_prefix)
 
         pars.update(full_model[iline].make_params())
         tmp_wave = this_line['WAVE']-1.
         tmp_loglam = np.log10(this_line['WAVE']-1.)

         tmp_left = np.log10(this_line['WAVE']-left_bound)
         tmp_right = np.log10(this_line['WAVE']-right_bound)
         pars[tmp_prefix+'center'].set(tmp_loglam, min=tmp_left, max=tmp_right)
         pars[tmp_prefix+'sigma'].set(width, min=min_width, max=max_width)

         tmp_sign = this_line['SIGN']
         tmp_amp = tmp_sign*this_line['EW']/tmp_wave/np.log(10.)
         if tmp_sign>0:
            pars[tmp_prefix+'amplitude'].set(tmp_amp, min=0, max=tmp_amp*namp)
         else:
            pars[tmp_prefix+'amplitude'].set(tmp_amp, min=tmp_amp*namp, max=0)

    model = full_model[0]
    for imod in np.arange(len(full_model)-1)+1:
        model = model+full_model[imod]

    return (model, pars)

def line_property(loglam, flux, lines, npixels=15):
    """Measure line properties in a normalized spectrum in the rest frame:
    Total equivalent width: REW
    Velocity profile: REW(velocity)/REW(total)
    """
    
    nlines = lines.size
    ew_profile = np.zeros(nlines, dtype=[('WAVE', '({0},)f4'.format(npixels)), ('VEL', '({0},)f4'.format(npixels)), ('EW', '({0},)f4'.format(npixels))])
    for (iline, this_line) in zip(np.arange(nlines), lines):
        tmp_loglam0 = np.log10(this_line['WAVE'])
        tmp_left = np.log10(this_line['WAVELEFT'])
        rest_loc = np.searchsorted(loglam, tmp_left)
        #print(rest_loc)
        #print(np.cumsum(flux[rest_loc:(rest_loc+npixels)]))
        ew_profile[iline]['EW'][:] = np.cumsum(flux[rest_loc:(rest_loc+npixels)])
        #print(ew_profile[iline]['EW'])
        ew_profile[iline]['VEL'][:] = (loglam[rest_loc:(rest_loc+npixels)]-tmp_loglam0)*np.log(10.)*3E5
        ew_profile[iline]['WAVE'][:] = np.power(10, loglam[rest_loc:(rest_loc+npixels)])

    return ew_profile

def speclines(region='2800'):
    if region == '2800':
       nlines = 2
       lines = zeros(nlines, dtype=[('SIGN', 'i'),('ELEMENT','S20'),('WAVE','f4'),('EW','f4'), ('WAVELEFT', 'f4')])
       lines[0] = (-1, 'MgII', 2796.35, 2., 2789.)
       lines[1] = (-1, 'MgII', 2803.53, 2., 2798.)
