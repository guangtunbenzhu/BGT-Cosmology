
""" 
Analysis tools for eboss (composite) spectra, for science
"""

# Python 3 vs. Python 2
from __future__ import division

# Standard library modules
from os.path import isfile, join
import numpy as np
from scipy.stats import nanmean, nanmedian
from scipy.interpolate import interp1d

# Third-party modules
import fitsio
from progressbar import ProgressBar
import lmfit
from sklearn import linear_model

# Your own modules
import datapath
import speclines
import specutils
import ebossspec

#####################
# Code starts here  #
#####################

_velspace_flux_file = 'Velspace_flux.fits'
_bootstrap_velspace_flux_file = 'Bootstrap_velspace_flux.fits'
_unify_emissionline_profile_file = 'Unify_emissionline_profile_flux.fits'
_bootstrap_unify_emissionline_profile_file = 'Bootstrap_unify_emissionline_profile_flux.fits'
_unify_absorptionline_profile_file = 'Unify_absorptionline_profile_flux.fits'
_bootstrap_unify_absorptionline_profile_file = 'Bootstrap_unify_absorptionline_profile_flux.fits'

# Normalization
_npix_left = 5 
_npix_right = 3+1
# Fitting
_npix_left_fit = 7
_npix_right_fit = 4+1
_noffset = 100
def velspace_flux_filename(bootstrap=False):
    """
    """
    path = datapath.sdss_path()
    if bootstrap:
        return join(path, 'eBOSS/lines', _bootstrap_velspace_flux_file)
    else:
        return join(path, 'eBOSS/lines', _velspace_flux_file)

def unify_emissionline_profile_filename(bootstrap=False):
    """
    """
    path = datapath.sdss_path()
    if bootstrap:
        return join(path, 'eBOSS/lines', _bootstrap_unify_emissionline_profile_file)
    else:
        return join(path, 'eBOSS/lines', _unify_emissionline_profile_file)

def unify_absorptionline_profile_filename(bootstrap=False):
    """
    """
    path = datapath.sdss_path()
    if bootstrap:
        return join(path, 'eBOSS/lines', _bootstrap_unify_absorptionline_profile_file)
    else:
        return join(path, 'eBOSS/lines', _unify_absorptionline_profile_file)


def velspace_flux(overwrite=False):
    """
    \pm 2000 km/s
    """

    # outfile
    outfile = velspace_flux_filename()
    if isfile(outfile) and not overwrite:
        print "File {0} exists. Use overwrite to overwrite it.".format(outfile)
        return -1

    # input data, might need to be moved out of the function
    infile = ebossspec.feiimgii_composite_filename()
    data = fitsio.read(infile)
    inloglam = np.log10(data['WAVE'][0])
    influx = data['FLUXMEDIAN'][0]
    inivar = np.ones(influx.size)
    
    # 4 Lines WITHOUT non-resonant channels
    linewave_nofluores = np.array([speclines.FeII2383.wave, speclines.MgII2796.wave,
        speclines.MgII2803.wave, speclines.MgI2853.wave])
    # 4 Lines WITH non-resonant channels
    linewave_yesfluores = np.array([speclines.FeII2344.wave, speclines.FeII2374.wave,
        speclines.FeII2587.wave, speclines.FeII2600.wave])
    # Non-resonant transitions
    linewave_nonres = np.array([speclines.FeII2366.wave, speclines.FeII2396.wave,
        speclines.FeII2613.wave, speclines.FeII2626.wave])
    linewave_all = np.r_[linewave_nofluores, linewave_yesfluores, linewave_nonres]

    # To velocity space centered on the rest-frame wavelength of the lines
    outvel, outflux = flux_wave2velspace(inloglam, influx, inivar, 
                               linewave_all)
    # Save the data into files
    print "Write into file: {0}".format(outfile)
    out_dtype = [('LINES', 'f8', linewave_all.shape),
                 ('VEL', 'f8', outvel.shape), 
                 ('FLUX', 'f8', outflux.shape)]
    outstr = np.array([(linewave_all, outvel, outflux)], dtype=out_dtype)
    fits = fitsio.FITS(outfile, 'rw', clobber=overwrite)
    fits.write(outstr)
    fits.close()

    return True

def flux_wave2velspace(inloglam, influx, inivar, pivot):
    """
    wavelength to velocityspace, with interpolation
    velocity = 0 at pivotal wavelength
    """
    outflux = np.zeros((np.ravel(np.array([pivot])).size, _noffset*2))
    for i, linewave in enumerate(np.ravel(np.array([pivot]))):
        outloglam = specutils.get_loglam(pivot=linewave)
        rest_loc = np.searchsorted(outloglam, np.log10(linewave))
        if (np.fabs(outloglam[rest_loc]-np.log10(linewave)) > \
            np.fabs(np.log10(linewave)-outloglam[rest_loc-1])):
            rest_loc -= 1
        outloglam = outloglam[rest_loc-_noffset:rest_loc+_noffset]
        (outflux[i, :], tmpivar) = specutils.interpol_spec(inloglam, influx, inivar, outloglam)

    outvel = specutils.get_velgrid(noffset=_noffset)

    return (outvel, outflux)


def velspace_flux_readin():
    """
    """
    infile = velspace_flux_filename()
    return (fitsio.read(infile))[0]

def unify_emissionline_profile(overwrite=False):
    """
    Create a common emission line profile
    """

    # Check outfile
    outfile = unify_emissionline_profile_filename()
    if isfile(outfile) and not overwrite:
        print "File {0} exists. Use overwrite to overwrite it.".format(outfile)
        return -1

    # Read in individual lines
    data = velspace_flux_readin()
    # Non-resonant emission lines; See velspace_flux() for the indices
    line_indices = np.array([8, 9, 10, 11])
    lines = data['LINES'][line_indices]
    vel = data['VEL']
    flux = data['FLUX'][line_indices]

    # Normalization
    flux_norm = np.sum(flux[:, _noffset-_npix_left:_noffset+_npix_right]-1., axis=1)
    normalized_flux = (flux-1.)/flux_norm.reshape(lines.size,1)+1.

    # Unified profile; use 2366, 2396 and 2626 only; 2613 is contaminated by MnII2606
    use_indices = np.array([0,1,2,3])
    unified_flux = np.sum(normalized_flux[use_indices,:]-1., axis=0)/use_indices.size+1.
    #unified_flux = np.median(normalized_flux[use_indices,:]-1., axis=0)+1.
    # blue side not using 2613
    use_indices = np.array([0,1,3])
    unified_flux[:_noffset-_npix_left] = np.sum(normalized_flux[use_indices,:_noffset-_npix_left]-1., axis=0)/use_indices.size+1.
    #unified_flux[:noffset-npix_left] = np.median(normalized_flux[use_indices,:noffset-npix_left]-1., axis=0)+1.

    # Set +-1000 km/s to 0
    unified_flux[:_noffset-15] = 1.
    unified_flux[_noffset+14:] = 1.

    # Write out
    out_dtype = [('LINES', 'f8', lines.shape),
                 ('VEL', 'f8', vel.shape), 
                 ('FLUX', 'f8', flux.shape),
                 ('NORMFLUX', 'f8', normalized_flux.shape),
                 ('FNORM', 'f8', flux_norm.shape),
                 ('INDEX', 'i4', use_indices.shape),
                 ('UNIFIEDFLUX', 'f8', unified_flux.shape)]
    outstr = np.array([(lines, vel, flux, normalized_flux, flux_norm, use_indices, unified_flux)], 
                      dtype=out_dtype)
    fits = fitsio.FITS(outfile, 'rw', clobber=overwrite)
    fits.write(outstr)
    fits.close()
   
    return True

def unify_emissionline_profile_readin():
    """
    """
    infile = unify_emissionline_profile_filename()
    return (fitsio.read(infile))[0]

def unify_absorptionline_profile(overwrite=False):
    """
    Create a common absorption line profile
    using 2374/2396 as an anchor
    """

    # Check outfile
    outfile = unify_absorptionline_profile_filename()
    if isfile(outfile) and not overwrite:
        print "File {0} exists. Use overwrite to overwrite it.".format(outfile)
        return -1

    # Read in individual lines. See velspace_flux()
    data = velspace_flux_readin()
    # Non-resonant emission lines; See velspace_flux() for the indices
    lines = data['LINES']
    vel = data['VEL']
    flux = data['FLUX']
    index_2396 = np.argmin(np.fabs(lines-2396.36))

    # Read in unified emission profile. See unify_emission_profile()
    emission = unify_emissionline_profile_readin()
    tmp_lines = emission['LINES']
    tmp_index_2396 = np.argmin(np.fabs(tmp_lines-2396.36))
    fnorm_2396 = emission['FNORM'][tmp_index_2396]
    # Velocity grid should be the same
    unified_vel = emission['VEL']
    assert np.allclose(vel, unified_vel), "Velocity grids are not the same."
    unified_emission = emission['UNIFIEDFLUX'] 
    # emission should be positive
    tmpunified_emission = unified_emission-1
    tmpunified_emission[tmpunified_emission<0.] = 0.
    unified_emission = 1.+tmpunified_emission

    # Use 2374 as the anchor; 2396 is the dominant fluorescent channel for 2374
    index_2374 = np.argmin(np.fabs(lines-2374.46))
    # resonant emission fraction 
    fresonant_2374 = speclines.FeII2374.EinsteinA/speclines.FeII2396.EinsteinA
    #absorption_2374 = flux[index_2374]-fresonant_2374*(fnorm_2396*(unified_emission[index_2396]-1.)) # Be careful with the sign here
    absorption_2374 = flux[index_2374]#-fresonant_2374*(fnorm_2396*(unified_emission[index_2396]-1.)) # Be careful with the sign here
    # Normalization
    fabs_norm_2374 = np.sum(1.-absorption_2374[_noffset-_npix_left:_noffset+_npix_right])
    # This must be almost the same as the true absorption profile
    absorption_norm_2374 = 1.-(1.-absorption_2374)/fabs_norm_2374
    # Set +-1000 km/s to be 0
    absorption_norm_2374[:_noffset-14] = 1.
    absorption_norm_2374[_noffset+14:] = 1.
    unified_old = absorption_norm_2374

    use_indices = np.arange(8)
    XX = np.zeros((_npix_right_fit+_npix_left_fit, 2))
    # Emission component
    xemission = unified_emission[_noffset-_npix_left_fit:_noffset+_npix_right_fit]-1.
    XX[:,1] = xemission 
    flux_absorption = np.zeros((8, vel.size))

    lr = linear_model.LinearRegression(fit_intercept=False)
    coeff_abs = np.zeros((8,2))
    niter_max = 5
    niter = 0
    while niter < niter_max:
        niter += 1
        # Now do all the 8 (7+2374) absorption lines

        # General linear model
        # Absorption component
        XX[:,0] = 1.-unified_old[_noffset-_npix_left_fit:_noffset+_npix_right_fit]
        for i, index in enumerate(use_indices):
            # Linear Regression only
            YY = 1.-flux[index, _noffset-_npix_left_fit:_noffset+_npix_right_fit]
            lr.fit(XX, YY)
            coeff_abs[i,:] = lr.coef_
            #if niter < (niter_max-1): 
            coeff_abs[i,1] = lr.coef_[1] if lr.coef_[1]<0 else 0.
            flux_absorption[i,:] = flux[index,:]+coeff_abs[i,1]*(unified_emission-1.) # Be careful with the sign here

        # Normalization
        fabs_norm = np.sum(1.-flux_absorption[:,_noffset-_npix_left:_noffset+_npix_right], axis=1)
        # tau>>1 approximation AFTER Normalization Factor
        flux_absorption[flux_absorption<0] = 0.
        normalized_flux_absorption = 1.-(1.-flux_absorption)/fabs_norm.reshape(use_indices.size,1)

        # Composite (Unified) Absorption Profile; Only use unsaturated lines
        unified_absorption = 1.-np.sum(1.-normalized_flux_absorption[3:,:], axis=0)/use_indices[3:].size
        # Set +-1000 km/s to be 0
        unified_absorption[:_noffset-14] = 1.
        unified_absorption[_noffset+14:] = 1.
        unified_old = unified_absorption

    # Refit Mg II (Indices 1/2)
    for i in np.arange(2)+1:
        coeff_abs[i,0] = fabs_norm[i]
        coeff_abs[i,1] = np.sum(flux_absorption[i,_noffset-_npix_left_fit:_noffset+_npix_right_fit]-flux[i,_noffset-_npix_left_fit:_noffset+_npix_right_fit])\
                        /np.sum(unified_emission[_noffset-_npix_left_fit:_noffset+_npix_right_fit]-1.)
    # Refit Mg II (with two lines)

    # Here we need another iteration to add *all* emission to the original spectrum and create the final 'true' absorption profile.



    # Write out
    out_dtype = [('LINES', 'f8', use_indices.shape),
                 ('VEL', 'f8', vel.shape), 
                 ('FLUX', 'f8', flux_absorption.shape), # Observed absorption
                 ('FABS', 'f8', flux_absorption.shape), # "True" absorption
                 ('NORMFABS', 'f8', normalized_flux_absorption.shape), # Normalized "True" absorption
                 ('FNORM', 'f8', fabs_norm.shape), # Normalization
                 ('INDEX', 'i4', use_indices.shape), 
                 ('UNIFIEDABSORPTION', 'f8', unified_absorption.shape), 
                 ('UNIFIEDEMISSION', 'f8', unified_emission.shape),
                 ('COEFF', 'f8', coeff_abs.shape),
                 ('FABS_2374', 'f8', absorption_norm_2374.shape)]
    outstr = np.array([(lines[use_indices], vel, flux[use_indices,:], flux_absorption, \
                       normalized_flux_absorption, fabs_norm, use_indices, unified_absorption, \
                       unified_emission, coeff_abs, absorption_norm_2374)], 
                       dtype=out_dtype)
    fits = fitsio.FITS(outfile, 'rw', clobber=overwrite)
    fits.write(outstr)
    fits.close()
   
    return True

def unify_absorptionline_profile_readin():
    """
    """
    infile = unify_absorptionline_profile_filename()
    return (fitsio.read(infile))[0]

def velocity_nonparametric():
    """
    """
    data = unify_absorptionline_profile_readin()
    vel = data['VEL']
    flux = data['FLUX']
    flux_abs = data['FABS']
    lines = data['LINES']
    
    percent = np.linspace(0.1, 0.9, 17)
    vpercent = np.zeros((lines.size, percent.size))
    vpercent_abs = np.zeros((lines.size, percent.size))
    unified_vpercent = np.zeros(percent.shape)

    vel_npix_left = 12. # different from _npix_left
    vel_npix_right = 6.+1.
    velspace = vel[_noffset-vel_npix_left:_noffset+vel_npix_right]
    #velspace = np.arange(npix_left+npix_right)*69.-npix_left*69. # Needs to double check
    for i in np.arange(lines.size):
        tmpflux = 1.-flux[i, _noffset-vel_npix_left:_noffset+vel_npix_right]
        tmpfabs = 1.-flux_abs[i, _noffset-vel_npix_left:_noffset+vel_npix_right]
        tmpflux[tmpflux<0.] = 1E-5
        tmpfabs[tmpfabs<0.] = 1E-5
        tmpflux_cumsum = (np.cumsum(tmpflux[::-1]))[::-1]
        tmpfabs_cumsum = (np.cumsum(tmpfabs[::-1]))[::-1]
        tmpflux_percent = tmpflux_cumsum/np.max(tmpflux_cumsum)
        tmpfabs_percent = tmpfabs_cumsum/np.max(tmpfabs_cumsum)
        #print(velspace.shape, tmpflux_percent.shape)
        finterp = interp1d(tmpflux_percent, velspace, kind='linear')
        finterp_abs = interp1d(tmpfabs_percent, velspace, kind='linear')
        vpercent[i,:] = finterp(percent)
        vpercent_abs[i,:] = finterp_abs(percent)

    unified_tmpfabs = 1.-data['UNIFIEDABSORPTION'][_noffset-vel_npix_left:_noffset+vel_npix_right]
    unified_tmpfabs[unified_tmpfabs<0.] = 1E-5
    unified_tmpfabs_cumsum = (np.cumsum(unified_tmpfabs[::-1]))[::-1]
    unified_tmpfabs_percent = unified_tmpfabs_cumsum/np.max(unified_tmpfabs_cumsum)
    finterp_abs = interp1d(unified_tmpfabs_percent, velspace)
    unified_vpercent = finterp_abs(percent)

    # Write out
    out_dtype = [('LINES', 'f8', lines.shape),
                 ('VELSPACE', 'f8', velspace.shape),
                 ('PERCENT', 'f8', percent.shape),
                 ('FLUX_PERCENT', 'f8', vpercent.shape), 
                 ('FABS_PERCENT', 'f8', vpercent_abs.shape), 
                 ('UNIFIEDPERCENT', 'f8', unified_vpercent.shape)]
    outstr = np.array([(lines, velspace, percent, vpercent, vpercent_abs, unified_vpercent)],
                       dtype=out_dtype)
    #fits = fitsio.FITS(outfile, 'rw', clobber=overwrite)
    #fits.write(outstr)
    #fits.close()

    return outstr
