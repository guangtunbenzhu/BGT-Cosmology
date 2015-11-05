
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
import starburstspec

# Some small number
_EPS = 1E-5

#####################
# Code starts here  #
#####################

_velspace_flux_file = 'Velspace_flux.fits'
_bootstrap_velspace_flux_file = 'Bootstrap_velspace_flux.fits'
_unify_emissionline_profile_file = 'Unify_emissionline_profile_flux.fits'
_bootstrap_unify_emissionline_profile_file = 'Bootstrap_unify_emissionline_profile_flux.fits'
_unify_absorptionline_profile_file = 'Unify_absorptionline_profile_flux.fits'
_bootstrap_unify_absorptionline_profile_file = 'Bootstrap_unify_absorptionline_profile_flux.fits'

# Velocity pixel number = 2*_noffset; each pixel 69 km/s
_noffset = 100
# Normalization
_npix_left = 5 
_npix_right = 3+1
# Fitting
_npix_left_fit = 7
_npix_right_fit = 4+1
# For nonparametric velocity measurement
_vel_npix_left = 9 
_vel_npix_right = 5+1
_percentlist = np.linspace(0.1, 0.9, 17)


def qsostack_absorber_filename(rew=False, mgiirewmin=2.0, mgiirewmax=8.0):
    path = datapath.absorber_path()
    filename = 'Absorbers_Composite_Allabs_{0:3.1f}_{1:3.1f}AA_z0.6_1.2.fits'.format(mgiirewmin, mgiirewmax)
    if (rew): filename = filename.replace('.fits', '_REW.fits')
    return join(path, filename)

def velspace_flux_filename(bootstrap=False, binoii=False):
    """
    """
    path = datapath.sdss_path()
    if bootstrap:
        if (not binoii):
            return join(path, 'eBOSS/lines', _bootstrap_velspace_flux_file)
        else:
            return join(path, 'eBOSS/lines', 'OII_'+_bootstrap_velspace_flux_file)
    else:
        if (not binoii):
            return join(path, 'eBOSS/lines', _velspace_flux_file)
        else:
            return join(path, 'eBOSS/lines', 'OII_'+_velspace_flux_file)

def oiii_velspace_flux_filename(bootstrap=False, binoii=False):
    """
    """
    path = datapath.sdss_path()
    if bootstrap:
        if (not binoii):
            return join(path, 'eBOSS/lines', 'OIII_'+_bootstrap_velspace_flux_file)
        else:
            return join(path, 'eBOSS/lines', 'OII_'+'OIII_'+_bootstrap_velspace_flux_file)
    else:
        if (not binoii):
            return join(path, 'eBOSS/lines', 'OIII_'+_velspace_flux_file)
        else:
            return join(path, 'eBOSS/lines', 'OII_'+'OIII_'+_velspace_flux_file)


def corrected_velspace_flux_filename(bootstrap=False, binoii=False):
    """
    """
    path = datapath.sdss_path()
    if bootstrap:
        if (not binoii):
            return join(path, 'eBOSS/lines', 'Corrected_'+_bootstrap_velspace_flux_file)
        else:
            return join(path, 'eBOSS/lines', 'OII_Corrected_'+_bootstrap_velspace_flux_file)
    else:
        if (not binoii):
            return join(path, 'eBOSS/lines', 'Corrected_'+_velspace_flux_file)
        else:
            return join(path, 'eBOSS/lines', 'OII_Corrected_'+_velspace_flux_file)

def unify_emissionline_profile_filename(bootstrap=False, binoii=False):
    """
    """
    path = datapath.sdss_path()
    if bootstrap:
        if (not binoii):
            return join(path, 'eBOSS/lines', _bootstrap_unify_emissionline_profile_file)
        else:
            return join(path, 'eBOSS/lines', 'OII_'+_bootstrap_unify_emissionline_profile_file)
    else:
        if (not binoii):
            return join(path, 'eBOSS/lines', _unify_emissionline_profile_file)
        else:
            return join(path, 'eBOSS/lines', 'OII_'+_unify_emissionline_profile_file)

def unify_absorptionline_profile_filename(bootstrap=False, binoii=False):
    """
    """
    path = datapath.sdss_path()
    if bootstrap:
        if (not binoii):
            return join(path, 'eBOSS/lines', _bootstrap_unify_absorptionline_profile_file)
        else:
            return join(path, 'eBOSS/lines', 'OII_'+_bootstrap_unify_absorptionline_profile_file)
    else:
        if (not binoii):
            return join(path, 'eBOSS/lines', _unify_absorptionline_profile_file)
        else:
            return join(path, 'eBOSS/lines', 'OII_'+_unify_absorptionline_profile_file)

def do_velspace_flux(overwrite=False, bootstrap=False, binoii=False):
    """
    """
    # outfile
    outfile = velspace_flux_filename(bootstrap=bootstrap, binoii=binoii)
    if (isfile(outfile) and (not overwrite)):
        print "File {0} exists. Use overwrite to overwrite it.".format(outfile)
        return -1

    # input data, might need to be moved out of the function
    data = ebossspec.feiimgii_composite_readin(bootstrap=bootstrap, binoii=binoii)
    inloglam = np.log10(data['WAVE'])
    if (not binoii): 
        influx = data['FLUXMEDIAN']
        inivar = np.ones(influx.shape)
        outstr = velspace_flux(inloglam, influx, inivar)
    else:
        ewinflux = data['EWFLUXMEDIAN']
        luminflux = data['LUMFLUXMEDIAN']
        nbin = data['OIIEWMIN'].size
        for i in np.arange(nbin):
            if (not bootstrap):
                tmp_ewinflux = ewinflux[:,i]
                tmp_luminflux = luminflux[:,i]
            else:
                tmp_ewinflux = ewinflux[:,:,i]
                tmp_luminflux = luminflux[:,:,i]
            inivar = np.ones(tmp_ewinflux.shape)
            tmp_ewoutstr = velspace_flux(inloglam, tmp_ewinflux, inivar)
            tmp_lumoutstr = velspace_flux(inloglam, tmp_luminflux, inivar)
            if (i == 0):
                ewoutstr = np.zeros(nbin, dtype=tmp_ewoutstr.dtype)
                lumoutstr = np.zeros(nbin, dtype=tmp_lumoutstr.dtype)
            ewoutstr[i] = tmp_ewoutstr
            lumoutstr[i] = tmp_lumoutstr
 
    # Save the data into files
    print "Write into file: {0}".format(outfile)
    
    fits = fitsio.FITS(outfile, 'rw', clobber=overwrite)
    if (not binoii):
        fits.write(outstr)
    else:
        fits.write(ewoutstr)
        fits.write(lumoutstr)
    fits.close()

    return True

def velspace_flux(inloglam, influx, inivar):
    """
    \pm 2000 km/s
    """

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

    # Some necessary quality control
    outflux[outflux<0] = 0.
    out_dtype = [('LINES', 'f8', linewave_all.shape),
                 ('VEL', 'f8', outvel.shape), 
                 ('FLUX', 'f8', outflux.shape)]
    outstr = np.array([(linewave_all, outvel, outflux)], dtype=out_dtype)

    return outstr

def flux_wave2velspace(inloglam, influx, inivar, pivot):
    """
    wavelength to velocityspace, with interpolation
    velocity = 0 at pivotal wavelength
    """
    if inloglam.ndim != 1:
        raise ValueError("There should be only one wavelength array!")

    # Initialize output, depending on the number of input spectra
    ndim_flux = influx.ndim
    if (ndim_flux == 1): 
        outflux = np.zeros((np.ravel(np.array([pivot])).size, _noffset*2))
    else:
        # If the dimension of input spectra is larger than one (ndim > 1), 
        #     then the spectra should be formatted as influx.shape = (nwave, nspec)
        nspec = (influx.shape)[1]
        outflux = np.zeros((np.ravel(np.array([pivot])).size, _noffset*2, nspec))

    # Loop over all the lines
    for i, linewave in enumerate(np.ravel(np.array([pivot]))):
        outloglam = specutils.get_loglam(pivot=linewave)
        rest_loc = np.searchsorted(outloglam, np.log10(linewave))
        if (np.fabs(outloglam[rest_loc]-np.log10(linewave)) > \
            np.fabs(np.log10(linewave)-outloglam[rest_loc-1])):
            rest_loc -= 1
        outloglam = outloglam[rest_loc-_noffset:rest_loc+_noffset]
        if (ndim_flux == 1):
           (outflux[i,:], tmpivar) = specutils.interpol_spec(inloglam, influx, inivar, outloglam)
        else:
           # Loop over all the spectra, could use vectorize/lambda functions
           #print "Looping over all the spectra"
           for j in np.arange(nspec):
               (outflux[i,:,j], tmpivar) = specutils.interpol_spec(inloglam, influx[:,j], inivar[:,j], outloglam)

    outvel = specutils.get_velgrid(noffset=_noffset)

    return (outvel, outflux)

def oiii_do_velspace_flux(overwrite=False, bootstrap=False, binoii=False):
    """
    """
    outfile = oiii_velspace_flux_filename()
    if (isfile(outfile) and (not overwrite)):
        print "File {0} exists. Use overwrite to overwrite it.".format(outfile)
        return -1

    # input data, might need to be moved out of the function
    data = ebossspec.feiimgii_composite_readin(bootstrap=bootstrap, binoii=binoii)
    inloglam = np.log10(data['WAVE'])
    if (not binoii):
        influx = data['OIII_FLUXMEDIAN']
        inivar = np.ones(influx.shape)
        outstr = oiii_velspace_flux(inloglam, influx, inivar)
    else:
        ewinflux = data['EWOIII_FLUXMEDIAN']
        luminflux = data['LUMOIII_FLUXMEDIAN']
        nbin = data['OIIEWMIN'].size
        for i in np.arange(nbin):
            if (not bootstrap):
                tmp_ewinflux = ewinflux[:,i]
                tmp_luminflux = luminflux[:,i]
            else:
                tmp_ewinflux = ewinflux[:,:,i]
                tmp_luminflux = luminflux[:,:,i]
            inivar = np.ones(tmp_ewinflux.shape)
            tmp_ewoutstr = oiii_velspace_flux(inloglam, tmp_ewinflux, inivar)
            tmp_lumoutstr = oiii_velspace_flux(inloglam, tmp_luminflux, inivar)
            if (i == 0):
                ewoutstr = np.zeros(nbin, dtype=tmp_ewoutstr.dtype)
                lumoutstr = np.zeros(nbin, dtype=tmp_lumoutstr.dtype)
            ewoutstr[i] = tmp_ewoutstr
            lumoutstr[i] = tmp_lumoutstr

    # Save the data into files
    print "Write into file: {0}".format(outfile)

    fits = fitsio.FITS(outfile, 'rw', clobber=overwrite)
    if (not binoii):
        fits.write(outstr)
    else:
        fits.write(ewoutstr)
        fits.write(lumoutstr)
    fits.close()

    return True

def oiii_velspace_flux(inloglam, influx, inivar):
    """
    \pm 2000 km/s
    """

    # 4 Lines WITHOUT non-resonant channels
    linewave_all = np.array([speclines.OIII5008.wave, speclines.OIII4960.wave])

    # To velocity space centered on the rest-frame wavelength of the lines
    outvel, outflux = flux_wave2velspace(inloglam, influx, inivar, 
                               linewave_all)

    # Some necessary quality control
    outflux[outflux<0] = 0.
    out_dtype = [('LINES', 'f8', linewave_all.shape),
                 ('VEL', 'f8', outvel.shape), 
                 ('FLUX', 'f8', outflux.shape)]
    outstr = np.array([(linewave_all, outvel, outflux)], dtype=out_dtype)

    return outstr

def oiii_velspace_flux_readin(bootstrap=False, binoii=False):
    """
    """
    infile = oiii_velspace_flux_filename(bootstrap=bootstrap, binoii=binoii)
    if (not binoii):
        return (fitsio.read(infile))[0]
    else:
        return (fitsio.read(infile, 1), fitsio.read(infile, 2)) 


def vel2wavespace(pivot, invel, influx, inivar, outloglam):
    """
    velocity to wavelength space, with interpolation
    velocity = 0 at pivotal wavelength
    """
    if (influx.ndim > 1):
        raise ValueError("I can only take one spectrum for now.")

    outflux = np.zeros((np.ravel(np.array([pivot])).size, outloglam.size))
    for i, linewave in enumerate(np.ravel(np.array([pivot]))):
        inloglam = specutils.vel2loglam(linewave, invel)
        rest_loc = np.searchsorted(outloglam, np.log10(linewave))
        if (np.fabs(outloglam[rest_loc]-np.log10(linewave)) > \
            np.fabs(np.log10(linewave)-outloglam[rest_loc-1])):
            rest_loc -= 1
        tmploglam = outloglam[rest_loc-_noffset:rest_loc+_noffset]
        (tmpflux, tmpivar) = specutils.interpol_spec(inloglam, influx, inivar, tmploglam)
        outflux[i, rest_loc-_noffset:rest_loc+_noffset] = tmpflux

    return outflux

def velspace_flux_readin(bootstrap=False, binoii=False):
    """
    """
    infile = velspace_flux_filename(bootstrap=bootstrap, binoii=binoii)
    if (not binoii):
        return (fitsio.read(infile))[0]
    else:
        return (fitsio.read(infile, 1), fitsio.read(infile, 2)) 

def corrected_velspace_flux_readin(bootstrap=False, binoii=False):
    """
    """
    infile = corrected_velspace_flux_filename(bootstrap=bootstrap, binoii=binoii)
    if (not binoii):
        return (fitsio.read(infile))[0]
    else:
        return (fitsio.read(infile, 1), fitsio.read(infile, 2)) 

def unify_emissionline_profile(overwrite=False, bootstrap=False):
    """
    Create a common emission line profile
    """

    # Check outfile
    outfile = unify_emissionline_profile_filename(bootstrap=bootstrap)
    if (isfile(outfile) and (not overwrite)):
        print "File {0} exists. Use overwrite to overwrite it.".format(outfile)
        return -1

    # Read in individual lines
    data = velspace_flux_readin(bootstrap=bootstrap)
    # Non-resonant emission lines; See velspace_flux() for the indices
    line_indices = np.array([8, 9, 10, 11])
    lines = data['LINES'][line_indices]
    vel = data['VEL']
    flux = data['FLUX'][line_indices]

    # Normalization
    if (not bootstrap):
        flux_norm = np.sum(flux[:, _noffset-_npix_left:_noffset+_npix_right]-1., axis=1)
        normalized_flux = (flux-1.)/flux_norm.reshape(lines.size,1)+1.

        # Unified profile; use 2366, 2396 and 2626 only; 2613 is contaminated by MnII2606
        use_indices = np.array([0,1,2,3])
        unified_flux = np.sum(normalized_flux[use_indices,:]-1., axis=0)/use_indices.size+1.

        # Dispersion
        unified_disp = np.std(normalized_flux[use_indices,:]-1., axis=0)

        #unified_flux = np.median(normalized_flux[use_indices,:]-1., axis=0)+1.
        # blue side not using 2613
        use_indices = np.array([0,1,3])
        unified_flux[:_noffset-_npix_left] = np.sum(normalized_flux[use_indices,:_noffset-_npix_left]-1., axis=0)/use_indices.size+1.
        #unified_disp[:_noffset-_npix_left] = np.std(normalized_flux[use_indices,:]-1., axis=0)
        #unified_flux[:noffset-npix_left] = np.median(normalized_flux[use_indices,:noffset-npix_left]-1., axis=0)+1.

        # Set +-1000 km/s to 0
        unified_flux[:_noffset-15] = 1.
        unified_flux[_noffset+14:] = 1.
    else: 
        # print "flux.shape: {0}".format(flux.shape)
        nspec = (flux.shape)[2]
        flux_norm = np.sum(flux[:, _noffset-_npix_left:_noffset+_npix_right, :]-1., axis=1)
        normalized_flux = (flux-1.)/flux_norm.reshape(lines.size,1,nspec)+1.

        # Unified profile; use 2366, 2396 and 2626 only; 2613 is contaminated by MnII2606
        use_indices = np.array([0,1,2,3])
        unified_flux = np.sum(normalized_flux[use_indices,:,:]-1., axis=0)/use_indices.size+1.

        #unified_flux = np.median(normalized_flux[use_indices,:]-1., axis=0)+1.
        # blue side not using 2613
        use_indices = np.array([0,1,3])
        unified_flux[:_noffset-_npix_left,:] = np.sum(normalized_flux[use_indices,:_noffset-_npix_left,:]-1., axis=0)/use_indices.size+1.
        #unified_flux[:noffset-npix_left] = np.median(normalized_flux[use_indices,:noffset-npix_left]-1., axis=0)+1.

        # Set +-1000 km/s to 0
        unified_flux[:_noffset-15,:] = 1.
        unified_flux[_noffset+14:,:] = 1.

        # Dispersion
        tmp_flux = np.zeros((nspec*use_indices.size, vel.size))
        for i in np.arange(nspec):
            for j in np.arange(use_indices.size):
                tmp_flux[i*use_indices.size+j,:] = normalized_flux[use_indices[j],:,i]
        unified_disp = np.std(tmp_flux-1., axis=0)

    # Write out
    out_dtype = [('LINES', 'f8', lines.shape),
                 ('VEL', 'f8', vel.shape), 
                 ('FLUX', 'f8', flux.shape),
                 ('NORMFLUX', 'f8', normalized_flux.shape),
                 ('FNORM', 'f8', flux_norm.shape),
                 ('INDEX', 'i4', use_indices.shape),
                 ('UNIFIEDFLUX', 'f8', unified_flux.shape),
                 ('UNIFIEDDISP', 'f8', unified_disp.shape)]
    outstr = np.array([(lines, vel, flux, normalized_flux, flux_norm, use_indices, unified_flux, unified_disp)], 
                      dtype=out_dtype)
    fits = fitsio.FITS(outfile, 'rw', clobber=overwrite)
    fits.write(outstr)
    fits.close()
   
    return True

def unify_emissionline_profile_binoii(overwrite=False, bootstrap=False):
    """
    Create a common emission line profile
    """

    # Check outfile
    outfile = unify_emissionline_profile_filename(bootstrap=bootstrap, binoii=True)
    if (isfile(outfile) and (not overwrite)):
        print "File {0} exists. Use overwrite to overwrite it.".format(outfile)
        return -1

    # Read in individual lines
    (ewdata, lumdata) = velspace_flux_readin(bootstrap=bootstrap, binoii=True)
    # Non-resonant emission lines; See velspace_flux() for the indices
    line_indices = np.array([8, 9, 10, 11])

    lines = ewdata[0]['LINES'][line_indices]
    vel = ewdata[0]['VEL']
    nbin = ewdata.size
    for iewlum, thisdata in enumerate([ewdata, lumdata]):
        for ibin in np.arange(nbin):
            flux = thisdata[ibin]['FLUX'][line_indices]

            # Normalization
            if (not bootstrap):
                flux_norm = np.sum(flux[:, _noffset-_npix_left:_noffset+_npix_right]-1., axis=1)
                normalized_flux = (flux-1.)/flux_norm.reshape(lines.size,1)+1.

                # Unified profile; use 2366, 2396 and 2626 only; 2613 is contaminated by MnII2606
                use_indices = np.array([0,1,2,3])
                unified_flux = np.sum(normalized_flux[use_indices,:]-1., axis=0)/use_indices.size+1.
        
                # Dispersion
                unified_disp = np.std(normalized_flux[use_indices,:]-1., axis=0)
    
                #unified_flux = np.median(normalized_flux[use_indices,:]-1., axis=0)+1.
                # blue side not using 2613
                use_indices = np.array([0,1,3])
                unified_flux[:_noffset-_npix_left] = np.sum(normalized_flux[use_indices,:_noffset-_npix_left]-1., axis=0)/use_indices.size+1.
                #unified_disp[:_noffset-_npix_left] = np.std(normalized_flux[use_indices,:]-1., axis=0)
                #unified_flux[:noffset-npix_left] = np.median(normalized_flux[use_indices,:noffset-npix_left]-1., axis=0)+1.

                # Set +-1000 km/s to 0
                unified_flux[:_noffset-15] = 1.
                unified_flux[_noffset+14:] = 1.
            else: 
                # print "flux.shape: {0}".format(flux.shape)
                nspec = (flux.shape)[2]
                flux_norm = np.sum(flux[:, _noffset-_npix_left:_noffset+_npix_right, :]-1., axis=1)
                normalized_flux = (flux-1.)/flux_norm.reshape(lines.size,1,nspec)+1.

                # Unified profile; use 2366, 2396 and 2626 only; 2613 is contaminated by MnII2606
                use_indices = np.array([0,1,2,3])
                unified_flux = np.sum(normalized_flux[use_indices,:,:]-1., axis=0)/use_indices.size+1.

                #unified_flux = np.median(normalized_flux[use_indices,:]-1., axis=0)+1.
                # blue side not using 2613
                use_indices = np.array([0,1,3])
                unified_flux[:_noffset-_npix_left,:] = np.sum(normalized_flux[use_indices,:_noffset-_npix_left,:]-1., axis=0)/use_indices.size+1.
                #unified_flux[:noffset-npix_left] = np.median(normalized_flux[use_indices,:noffset-npix_left]-1., axis=0)+1.
    
                # Set +-1000 km/s to 0
                unified_flux[:_noffset-15,:] = 1.
                unified_flux[_noffset+14:,:] = 1.
    
                # Dispersion
                tmp_flux = np.zeros((nspec*use_indices.size, vel.size))
                for i in np.arange(nspec):
                    for j in np.arange(use_indices.size):
                        tmp_flux[i*use_indices.size+j,:] = normalized_flux[use_indices[j],:,i]
                unified_disp = np.std(tmp_flux-1., axis=0)

            if ((iewlum == 0) and (ibin == 0)):
                out_dtype = [('LINES', 'f8', lines.shape),
                              ('VEL', 'f8', vel.shape), 
                              ('FLUX', 'f8', flux.shape),
                              ('NORMFLUX', 'f8', normalized_flux.shape),
                              ('FNORM', 'f8', flux_norm.shape),
                              ('INDEX', 'i4', use_indices.shape),
                              ('UNIFIEDFLUX', 'f8', unified_flux.shape),
                              ('UNIFIEDDISP', 'f8', unified_disp.shape)]
                ew_outstr = np.zeros(nbin, dtype=out_dtype)
                lum_outstr = np.zeros(nbin, dtype=out_dtype)

            tmp_outstr = np.array([(lines, vel, flux, normalized_flux, flux_norm, use_indices, unified_flux, unified_disp)],
                                  dtype=out_dtype)
            if (iewlum == 0):
                ew_outstr[ibin] = tmp_outstr
            else:
                lum_outstr[ibin] = tmp_outstr

    fits = fitsio.FITS(outfile, 'rw', clobber=overwrite)
    fits.write(ew_outstr)
    fits.write(lum_outstr)
    fits.close()
   
    return True


def unify_emissionline_profile_readin(bootstrap=False, binoii=False):
    """
    """
    infile = unify_emissionline_profile_filename(bootstrap=bootstrap, binoii=binoii)
    if (not binoii):
        return (fitsio.read(infile))[0]
    else:
        return (fitsio.read(infile, 1), fitsio.read(infile, 2)) 

def unify_absorptionline_profile(overwrite=False, bootstrap=False):
    """
    Create a common absorption line profile
    using 2374/2396 as an anchor
    """

    # Check outfile
    outfile = unify_absorptionline_profile_filename(bootstrap=bootstrap)
    if (isfile(outfile) and (not overwrite)):
        print "File {0} exists. Use overwrite to overwrite it.".format(outfile)
        return -1

    # Read in individual lines. See velspace_flux()
    data = velspace_flux_readin(bootstrap=bootstrap)
    # Non-resonant emission lines; See velspace_flux() for the indices
    lines = data['LINES']
    vel = data['VEL']
    flux = data['FLUX']
    index_2396 = np.argmin(np.fabs(lines-2396.36))

    # Read in unified emission profile. See unify_emission_profile()
    emission = unify_emissionline_profile_readin(bootstrap=bootstrap)
    tmp_lines = emission['LINES']
    tmp_index_2396 = np.argmin(np.fabs(tmp_lines-2396.36))
    fnorm_2396 = emission['FNORM'][tmp_index_2396]
    # Velocity grid should be the same
    unified_vel = emission['VEL']
    assert np.allclose(vel, unified_vel), "Velocity grids are not the same."
    unified_emission = emission['UNIFIEDFLUX'] 
    unified_emission_disp = emission['UNIFIEDDISP'] 
    # emission should be positive
    tmpunified_emission = unified_emission-1
    tmpunified_emission[tmpunified_emission<0.] = _EPS
    unified_emission = 1.+tmpunified_emission

    # Use 2374 as the anchor; 2396 is the dominant fluorescent channel for 2374
    index_2374 = np.argmin(np.fabs(lines-2374.46))
    # resonant emission fraction 
    fresonant_2374 = speclines.FeII2374.EinsteinA/speclines.FeII2396.EinsteinA
    absorption_2374 = flux[index_2374]#-fresonant_2374*(fnorm_2396*(unified_emission[index_2396]-1.)) # Be careful with the sign here

    use_indices = np.arange(8)
    nuse = use_indices.size
    # This is not ideal (repetition ...)
    if (not bootstrap):
        # Normalization
        fabs_norm_2374 = np.sum(1.-absorption_2374[_noffset-_npix_left:_noffset+_npix_right])
        # This must be almost the same as the true absorption profile
        absorption_norm_2374 = 1.-(1.-absorption_2374)/fabs_norm_2374
        # Set +-1000 km/s to be 0
        absorption_norm_2374[:_noffset-14] = 1.
        absorption_norm_2374[_noffset+14:] = 1.
        unified_old = absorption_norm_2374

        XX = np.zeros((_npix_right_fit+_npix_left_fit, 2))
        # Emission component
        xemission = unified_emission[_noffset-_npix_left_fit:_noffset+_npix_right_fit]-1.
        XX[:,1] = xemission 
        flux_absorption = np.zeros((nuse, vel.size))

        lr = linear_model.LinearRegression(fit_intercept=False)
        coeff_abs = np.zeros((nuse,2))
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
            flux_absorption[flux_absorption<0] = _EPS
            normalized_flux_absorption = 1.-(1.-flux_absorption)/fabs_norm.reshape(nuse,1)

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

        # Dispersion
        unified_disp = np.std(1.-normalized_flux_absorption[3:,:], axis=0)
    else: 
        nspec = (flux.shape)[2]
        # Normalization
        fabs_norm_2374 = np.sum(1.-absorption_2374[_noffset-_npix_left:_noffset+_npix_right,:], axis=0)
        # This must be almost the same as the true absorption profile
        absorption_norm_2374 = 1.-(1.-absorption_2374)/fabs_norm_2374.reshape(1,nspec)
        # Set +-1000 km/s to be 0
        absorption_norm_2374[:_noffset-14,:] = 1.
        absorption_norm_2374[_noffset+14:,:] = 1.


        XX = np.zeros((_npix_right_fit+_npix_left_fit, 2))
        lr = linear_model.LinearRegression(fit_intercept=False)
        coeff_abs = np.zeros((nuse,2,nspec))
        flux_absorption = np.zeros((nuse, vel.size, nspec))
        normalized_flux_absorption = np.zeros((nuse, vel.size, nspec))
        unified_absorption = np.zeros((vel.size,nspec))
        niter_max = 5

        # Emission component
        #print "Looping over all spectra..."
        for j in np.arange(nspec):
            unified_old = absorption_norm_2374[:,j]
            xemission = unified_emission[_noffset-_npix_left_fit:_noffset+_npix_right_fit,j]-1.
            XX[:,1] = xemission 

            niter = 0
            while niter < niter_max:
                niter += 1
                # Now do all the 8 (7+2374) absorption lines
    
                # General linear model
                # Absorption component
                XX[:,0] = 1.-unified_old[_noffset-_npix_left_fit:_noffset+_npix_right_fit]
                for i, index in enumerate(use_indices):
                    # Linear Regression only
                    YY = 1.-flux[index, _noffset-_npix_left_fit:_noffset+_npix_right_fit,j]
                    lr.fit(XX, YY)
                    coeff_abs[i,:,j] = lr.coef_
                    #if niter < (niter_max-1): 
                    coeff_abs[i,1,j] = lr.coef_[1] if lr.coef_[1]<0 else 0.
                    flux_absorption[i,:,j] = flux[index,:,j]+coeff_abs[i,1,j]*(unified_emission[:,j]-1.) # Be careful with the sign here

                # Normalization
                fabs_norm = np.sum(1.-flux_absorption[:,_noffset-_npix_left:_noffset+_npix_right,j], axis=1)
                # tau>>1 approximation AFTER Normalization Factor
                flux_absorption[flux_absorption<0] = _EPS
                normalized_flux_absorption[:,:,j] = 1.-(1.-flux_absorption[:,:,j])/fabs_norm.reshape(nuse,1)

                # Composite (Unified) Absorption Profile; Only use unsaturated lines
                unified_absorption[:,j] = 1.-np.sum(1.-normalized_flux_absorption[3:,:,j], axis=0)/use_indices[3:].size
                # Set +-1000 km/s to be 0
                unified_absorption[:_noffset-14,j] = 1.
                unified_absorption[_noffset+14:,j] = 1.
                unified_old = unified_absorption[:,j]

            # Refit Mg II (Indices 1/2)
            for i in np.arange(2)+1:
                coeff_abs[i,0,j] = fabs_norm[i]
                coeff_abs[i,1,j] = np.sum(flux_absorption[i,_noffset-_npix_left_fit:_noffset+_npix_right_fit,j]-flux[i,_noffset-_npix_left_fit:_noffset+_npix_right_fit,j], axis=0)\
                                /np.sum(unified_emission[_noffset-_npix_left_fit:_noffset+_npix_right_fit,j]-1., axis=0)
            # Refit Mg II (with two lines)

            # Here we need another iteration to add *all* emission to the original spectrum and create the final 'true' absorption profile.

        # Dispersion
        tmp_flux = np.zeros((nspec*use_indices[3:].size, vel.size))
        for i in np.arange(nspec):
            for j in np.arange(use_indices[3:].size):
                tmp_flux[i*use_indices[3:].size+j,:] = normalized_flux_absorption[use_indices[3+j],:,i]
        unified_disp = np.std(tmp_flux-1., axis=0)

    # Write out
    out_dtype = [('LINES', 'f8', use_indices.shape),
                 ('VEL', 'f8', vel.shape), 
                 ('FLUX', 'f8', flux_absorption.shape), # Observed absorption
                 ('FABS', 'f8', flux_absorption.shape), # "True" absorption
                 ('NORMFABS', 'f8', normalized_flux_absorption.shape), # Normalized "True" absorption
                 ('FNORM', 'f8', fabs_norm.shape), # Normalization
                 ('INDEX', 'i4', use_indices.shape), 
                 ('UNIFIEDABSORPTION', 'f8', unified_absorption.shape), 
                 ('UNIFIEDABSORPTION_DISP', 'f8', unified_disp.shape), 
                 ('UNIFIEDEMISSION', 'f8', unified_emission.shape),
                 ('UNIFIEDEMISSION_DISP', 'f8', unified_emission_disp.shape),
                 ('COEFF', 'f8', coeff_abs.shape),
                 ('FABS_2374', 'f8', absorption_norm_2374.shape)]
    outstr = np.array([(lines[use_indices], vel, flux[use_indices,:], flux_absorption, \
                       normalized_flux_absorption, fabs_norm, use_indices, unified_absorption, unified_disp, \
                       unified_emission, unified_emission_disp, coeff_abs, absorption_norm_2374)], 
                       dtype=out_dtype)
    fits = fitsio.FITS(outfile, 'rw', clobber=overwrite)
    fits.write(outstr)
    fits.close()

    # Correct the original spectrum
    print "Correting the original spectra..."
    temp = correct_emission_infill(overwrite=overwrite, bootstrap=bootstrap)
   
    #return (loglam, corrected)
    return True

def unify_absorptionline_profile_readin(bootstrap=False, binoii=False):
    """
    """
    infile = unify_absorptionline_profile_filename(bootstrap=bootstrap, binoii=binoii)
    if (not binoii):
        return (fitsio.read(infile))[0]
    else:
        return (fitsio.read(infile, 1), fitsio.read(infile, 2)) 

def correct_emission_infill(overwrite=False, bootstrap=False):
    """
    """

    outfile = corrected_velspace_flux_filename(bootstrap=bootstrap)
    if (isfile(outfile) and (not overwrite)):
        print "File {0} exists. Use overwrite to overwrite it.".format(outfile)
        return -1

    # Original composite
    composite = ebossspec.feiimgii_composite_readin(bootstrap=bootstrap)
    absorption = unify_absorptionline_profile_readin(bootstrap=bootstrap)

    inloglam = np.log10(composite['WAVE'])
    influx = composite['FLUXMEDIAN']
    inivar = np.ones(influx.shape)
    outstr = single_correct_emission_infill(inloglam, influx, inivar, absorption, bootstrap=bootstrap)
    
    # Save the data into files
    print "Write into file: {0}".format(outfile)
    
    fits = fitsio.FITS(outfile, 'rw', clobber=overwrite)
    fits.write(outstr)
    fits.close()

    return True

def correct_emission_infill_binoii(overwrite=False, bootstrap=False):
    """
    """

    outfile = corrected_velspace_flux_filename(bootstrap=bootstrap, binoii=True)
    if (isfile(outfile) and (not overwrite)):
        print "File {0} exists. Use overwrite to overwrite it.".format(outfile)
        return -1

    # Original composite
    composite0 = ebossspec.feiimgii_composite_readin(bootstrap=bootstrap, binoii=True)
    (ew_absorption,lum_absorption) = unify_absorptionline_profile_readin(bootstrap=bootstrap, binoii=True)

    nbin = ew_absorption.size
    inloglam = np.log10(composite0['WAVE'])
    for iewlum in np.arange(2):
        if (iewlum == 0):
            thisflux = composite0['EWFLUXMEDIAN']
            thisabsorption = ew_absorption
        else:
            thisflux = composite0['LUMFLUXMEDIAN']
            thisabsorption = lum_absorption

        for ibin in np.arange(nbin):
            if (not bootstrap):
                influx = thisflux[:,ibin]
            else:
                influx = thisflux[:,:,ibin]

            inivar = np.ones(influx.shape)
            absorption = thisabsorption[ibin]
            tmp_outstr = single_correct_emission_infill(inloglam, influx, inivar, absorption, bootstrap=bootstrap)
            if ((iewlum == 0) and (ibin == 0)):
                ew_outstr = np.zeros(nbin, dtype=tmp_outstr.dtype)
                lum_outstr = np.zeros(nbin, dtype=tmp_outstr.dtype)
            if (iewlum == 0):
                ew_outstr[ibin] = tmp_outstr[0]
            else:
                lum_outstr[ibin] = tmp_outstr[0]
    
    # Save the data into files
    print "Write into file: {0}".format(outfile)
    
    fits = fitsio.FITS(outfile, 'rw', clobber=overwrite)
    fits.write(ew_outstr)
    fits.write(lum_outstr)
    fits.close()

    return True

def single_correct_emission_infill(inloglam, influx, inivar, absorption, bootstrap=False):
    """
    """

    # Emission infill data
    lines = absorption['LINES']
    vel = absorption['VEL']
    coeff = absorption['COEFF']
    emission_infill = np.zeros(influx.shape)
    unifiedemission = absorption['UNIFIEDEMISSION']
    tmpivar = np.ones(vel.shape)
    for i, thisline in enumerate(lines):
        if (not bootstrap):
            tmpemission = (unifiedemission-1.)*coeff[i,1]
            outemission = vel2wavespace(thisline, vel, tmpemission, tmpivar, inloglam)
            emission_infill += np.ravel(outemission)
        else:
            nspec = (influx.shape)[1]
            outemission = np.zeros(influx.shape)
            for j in np.arange(nspec):
                tmpemission = (unifiedemission[:,j]-1.)*coeff[i,1,j]
                outemission[:,j] = vel2wavespace(thisline, vel, tmpemission, tmpivar, inloglam)
                emission_infill[:,j] += outemission[:,j]

    outflux = influx+emission_infill
    outflux[outflux<0] = _EPS
    outstr = velspace_flux(inloglam, outflux, inivar)

    return outstr

def velocity_nonparametric(vel, flux, percent=_percentlist):
    """
    """
    vpercent = np.zeros(percent.shape)
    velspace = vel[_noffset-_vel_npix_left:_noffset+_vel_npix_right]

    tmpflux = 1.-flux[_noffset-_vel_npix_left:_noffset+_vel_npix_right]
    tmpflux[tmpflux<0.] = _EPS
    tmpflux_cumsum = (np.cumsum(tmpflux[::-1]))[::-1]
    tmpflux_percent = tmpflux_cumsum/np.max(tmpflux_cumsum)
    finterp = interp1d(tmpflux_percent, velspace, kind='linear')
    try:
        vpercent = finterp(percent)
    except ValueError:
        print "This interpolation has some issues... Set to -9999."
        vpercent[:] = -9999.

    return (vpercent, np.max(tmpflux_cumsum))

def do_velocity_nonparametric(bootstrap=False):
    """
    """
    absorption = unify_absorptionline_profile_readin(bootstrap=bootstrap)
    corrected = corrected_velspace_flux_readin(bootstrap=bootstrap)

    outstr = single_velocity_nonparametric(absorption, corrected, bootstrap=bootstrap)

    return outstr[0]

def do_velocity_nonparametric_binoii(bootstrap=False):
    """
    """
    (ew_absorption0, lum_absorption0) = unify_absorptionline_profile_readin(bootstrap=bootstrap, binoii=True)
    (ew_corrected0, lum_corrected0) = corrected_velspace_flux_readin(bootstrap=bootstrap, binoii=True)
    nbin = ew_absorption0.size

    for iewlum in np.arange(2):
        if (iewlum == 0):
            absorption0 = ew_absorption0
            corrected0 = ew_corrected0
        else:
            absorption0 = lum_absorption0
            corrected0 = lum_corrected0

        for ibin in np.arange(nbin):
            absorption = absorption0[ibin]
            corrected = corrected0[ibin]
            tmp_outstr = single_velocity_nonparametric(absorption, corrected, bootstrap=bootstrap)
            if ((iewlum == 0) and (ibin == 0)):
                ew_outstr = np.zeros(nbin, dtype=tmp_outstr.dtype)
                lum_outstr = np.zeros(nbin, dtype=tmp_outstr.dtype)
            if (iewlum == 0):
                ew_outstr[ibin] = tmp_outstr[0]
            else:
                lum_outstr[ibin] = tmp_outstr[0]

    return (ew_outstr, lum_outstr)

def single_velocity_nonparametric(absorption, corrected, bootstrap=False):
    """
    """

    dloglam = 1E-4
    lines = absorption['LINES']
    vel = absorption['VEL']
    flux = absorption['FLUX']
    flux_abs = corrected['FLUX']
    unified_flux = absorption['UNIFIEDABSORPTION']

    if (not bootstrap):
        vpercent = np.zeros((_percentlist.size, lines.size))
        vpercent_abs = np.zeros((_percentlist.size, lines.size))
        tflux = np.zeros(lines.shape)
        tfabs = np.zeros(lines.shape)

        for i, thisline in enumerate(lines):
            vpercent[:,i], tflux[i] = velocity_nonparametric(vel, flux[i,:])
            vpercent_abs[:,i], tfabs[i] = velocity_nonparametric(vel, flux_abs[i,:])
        tflux = tflux*dloglam*np.log(10.)*lines
        tfabs = tfabs*dloglam*np.log(10.)*lines
        unified_vpercent, tmp = velocity_nonparametric(vel, unified_flux)
    else:
        nspec = (unified_flux.shape)[1]
        vpercent = np.zeros((_percentlist.size,lines.size, nspec))
        vpercent_abs = np.zeros((_percentlist.size,lines.size, nspec))
        tflux = np.zeros((lines.size, nspec))
        tfabs = np.zeros((lines.size, nspec))
        unified_vpercent = np.zeros((_percentlist.size, nspec))

        for j in np.arange(nspec):
            for i, thisline in enumerate(lines):
                vpercent[:,i,j], tflux[i,j] = velocity_nonparametric(vel, flux[i,:,j])
                vpercent_abs[:,i,j], tfabs[i,j] = velocity_nonparametric(vel, flux_abs[i,:,j])
            tflux[:,j] = tflux[:,j]*dloglam*np.log(10.)*lines
            tfabs[:,j] = tfabs[:,j]*dloglam*np.log(10.)*lines
            unified_vpercent[:,j], tmp = velocity_nonparametric(vel, unified_flux[:,j])

    # Write out
    out_dtype = [('LINES', 'f8', lines.shape),
                 ('PERCENT', 'f8', _percentlist.shape),
                 ('FLUX_PERCENT', 'f8', vpercent.shape), 
                 ('FABS_PERCENT', 'f8', vpercent_abs.shape), 
                 ('UNIFIEDPERCENT', 'f8', unified_vpercent.shape),
                 ('TFLUX', 'f8', tflux.shape),
                 ('TFABS', 'f8', tfabs.shape)]

    outstr = np.array([(lines, _percentlist, vpercent, vpercent_abs, unified_vpercent, tflux, tfabs)],
                       dtype=out_dtype)

    return outstr

def absorber_measure(overwrite=False, mgiirewmin=2.0, mgiirewmax=8.0):
    """
    A stand-alone routine for absorbers
    """
    infile = qsostack_absorber_filename(rew=False, mgiirewmin=mgiirewmin, mgiirewmax=mgiirewmax)
    outfile = infile.replace('.fits', '_REW.fits')
    absorberstack = (fitsio.read(infile))[0]
    flux = absorberstack['FLUXMEDIAN']
    wave = absorberstack['WAVE']
    loglam = np.log10(wave)

    # 4 Lines WITHOUT resonant channels
    linewave_nofluores = np.array([speclines.FeII2383.wave, speclines.MgII2796.wave,
        speclines.MgII2803.wave, speclines.MgI2853.wave])
    # 4 Lines WITH non-resonant channels
    linewave_yesfluores = np.array([speclines.FeII2344.wave, speclines.FeII2374.wave,
        speclines.FeII2587.wave, speclines.FeII2600.wave])
    linewave_all = np.r_[linewave_nofluores, linewave_yesfluores]

    tflux = np.zeros(linewave_all.size)
    for i, thiswave in enumerate(linewave_all):
        rest_loc = np.searchsorted(loglam, np.log10(thiswave))
        dwave = np.median(wave[rest_loc-7:rest_loc+7] - wave[rest_loc-8:rest_loc+6])
        tflux[i] = np.sum(1.-flux[rest_loc-7:rest_loc+7])*dwave

    out_dtype = [('LINES', 'f8', linewave_all.shape),
                 ('TFLUX', 'f8', tflux.shape)]

    outstr = np.array([(linewave_all, tflux)],
                       dtype=out_dtype)

    # Write out
    fits = fitsio.FITS(outfile, 'rw', clobber=overwrite)
    fits.write(outstr)
    fits.close()

def stack_absorber_readin(rew=False, mgiirewmin=2.0, mgiirewmax=8.0):
    infile = qsostack_absorber_filename(rew=rew, mgiirewmin=mgiirewmin, mgiirewmax=mgiirewmax) 
    return (fitsio.read(infile))[0]

def starburst_measure(overwrite=False):
    """
    A stand-alone routine for local star-forming regions
    """
    infile = starburstspec.mgii_composite_filename()
    outfile = infile.replace('.fits', '_REW.fits')
    starburststack = starburstspec.mgii_composite_readin()
    flux = starburststack['FLUXMEDIAN']
    wave = starburststack['WAVE']
    loglam = np.log10(wave)

    # 4 Lines WITHOUT resonant channels
    linewave_nofluores = np.array([speclines.FeII2383.wave, speclines.MgII2796.wave,
        speclines.MgII2803.wave, speclines.MgI2853.wave])
    # 4 Lines WITH non-resonant channels
    linewave_yesfluores = np.array([speclines.FeII2344.wave, speclines.FeII2374.wave,
        speclines.FeII2587.wave, speclines.FeII2600.wave])
    linewave_all = np.r_[linewave_nofluores, linewave_yesfluores]

    tflux = np.zeros(linewave_all.size)
    for i, thiswave in enumerate(linewave_all):
        rest_loc = np.searchsorted(loglam, np.log10(thiswave))
        dwave = np.median(wave[rest_loc-7:rest_loc+7] - wave[rest_loc-8:rest_loc+6])
        tflux[i] = np.sum(1.-flux[rest_loc-7:rest_loc+7])*dwave

    out_dtype = [('LINES', 'f8', linewave_all.shape),
                 ('TFLUX', 'f8', tflux.shape)]

    outstr = np.array([(linewave_all, tflux)],
                       dtype=out_dtype)

    # Write out
    fits = fitsio.FITS(outfile, 'rw', clobber=overwrite)
    fits.write(outstr)
    fits.close()

def starburst_measure_readin():
    infile0 = starburstspec.mgii_composite_filename()
    infile = infile0.replace('.fits', '_REW.fits')
    return (fitsio.read(infile))[0]

def starburst_measure_jackknife(overwrite=False):
    """
    A stand-alone routine for local star-forming regions
    """
    infile = starburstspec.mgii_composite_filename()
    infile = infile.replace('.fits', '_jackknife.fits')
    outfile = infile.replace('.fits', '_REW.fits')
    starburststack = starburstspec.mgii_composite_jackknife_readin()
    flux = starburststack['FLUXMEDIAN']
    wave = starburststack['WAVE']
    loglam = np.log10(wave)

    # 4 Lines WITHOUT resonant channels
    linewave_nofluores = np.array([speclines.FeII2383.wave, speclines.MgII2796.wave,
        speclines.MgII2803.wave, speclines.MgI2853.wave])
    # 4 Lines WITH non-resonant channels
    linewave_yesfluores = np.array([speclines.FeII2344.wave, speclines.FeII2374.wave,
        speclines.FeII2587.wave, speclines.FeII2600.wave])
    linewave_all = np.r_[linewave_nofluores, linewave_yesfluores]

    njack = (flux.shape)[1]
    tflux = np.zeros((linewave_all.size, njack))
    for ijack in np.arange(njack):
        for i, thiswave in enumerate(linewave_all):
            rest_loc = np.searchsorted(loglam, np.log10(thiswave))
            dwave = np.median(wave[rest_loc-7:rest_loc+7] - wave[rest_loc-8:rest_loc+6])
            tflux[i, ijack] = np.sum(1.-flux[rest_loc-7:rest_loc+7, ijack])*dwave

    out_dtype = [('LINES', 'f8', linewave_all.shape),
                 ('TFLUX', 'f8', tflux.shape)]

    outstr = np.array([(linewave_all, tflux)],
                       dtype=out_dtype)

    # Write out
    fits = fitsio.FITS(outfile, 'rw', clobber=overwrite)
    fits.write(outstr)
    fits.close()

def starburst_measure_readin_jackknife():
    infile0 = starburstspec.mgii_composite_filename()
    infile0 = infile0.replace('.fits', '_jackknife.fits')
    infile = infile0.replace('.fits', '_REW.fits')
    return (fitsio.read(infile))[0]


def unify_absorptionline_profile_binoii(overwrite=False, bootstrap=False):
    """
    Create a common absorption line profile
    using 2374/2396 as an anchor
    """

    # Check outfile
    outfile = unify_absorptionline_profile_filename(bootstrap=bootstrap, binoii=True)
    if (isfile(outfile) and (not overwrite)):
        print "File {0} exists. Use overwrite to overwrite it.".format(outfile)
        return -1

    # Read in individual lines. See velspace_flux()
    (ewdata, lumdata) = velspace_flux_readin(bootstrap=bootstrap, binoii=True)
    # Non-resonant emission lines; See velspace_flux() for the indices
    lines = ewdata[0]['LINES']
    vel = ewdata[0]['VEL']
    nbin = ewdata.size

    # Read in unified emission profile. See unify_emission_profile()
    # Use the emission profile from *ALL* spectra
    emission = unify_emissionline_profile_readin(bootstrap=bootstrap)
    tmp_lines = emission['LINES']
    # Velocity grid should be the same
    unified_vel = emission['VEL']
    assert np.allclose(vel, unified_vel), "Velocity grids are not the same."

    unified_emission = emission['UNIFIEDFLUX'] 
    unified_emission_disp = emission['UNIFIEDDISP'] 

    # emission should be positive
    tmpunified_emission = unified_emission-1
    tmpunified_emission[tmpunified_emission<0.] = _EPS
    unified_emission = 1.+tmpunified_emission

    # Use 2374 as the anchor; 2396 is the dominant fluorescent channel for 2374
    # Forget about 2396
    index_2374 = np.argmin(np.fabs(lines-2374.46))

    use_indices = np.arange(8)
    nuse = use_indices.size

    for iewlum, thisdata in enumerate([ewdata, lumdata]):
        for ibin in np.arange(nbin):
            flux = thisdata[ibin]['FLUX']
            # resonant emission fraction 
            absorption_2374 = flux[index_2374]

            # This is not ideal (repetition ...)
            if (not bootstrap):
                # Normalization
                fabs_norm_2374 = np.sum(1.-absorption_2374[_noffset-_npix_left:_noffset+_npix_right])
                # This must be almost the same as the true absorption profile
                absorption_norm_2374 = 1.-(1.-absorption_2374)/fabs_norm_2374
                # Set +-1000 km/s to be 0
                absorption_norm_2374[:_noffset-14] = 1.
                absorption_norm_2374[_noffset+14:] = 1.
                unified_old = absorption_norm_2374

                XX = np.zeros((_npix_right_fit+_npix_left_fit, 2))
                # Emission component
                xemission = unified_emission[_noffset-_npix_left_fit:_noffset+_npix_right_fit]-1.
                XX[:,1] = xemission 
                flux_absorption = np.zeros((nuse, vel.size))

                lr = linear_model.LinearRegression(fit_intercept=False)
                coeff_abs = np.zeros((nuse,2))
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
                    flux_absorption[flux_absorption<0] = _EPS
                    normalized_flux_absorption = 1.-(1.-flux_absorption)/fabs_norm.reshape(nuse,1)
    
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

                # Dispersion
                unified_disp = np.std(1.-normalized_flux_absorption[3:,:], axis=0)
            else: 
                nspec = (flux.shape)[2]
                # Normalization
                fabs_norm_2374 = np.sum(1.-absorption_2374[_noffset-_npix_left:_noffset+_npix_right,:], axis=0)
                # This must be almost the same as the true absorption profile
                absorption_norm_2374 = 1.-(1.-absorption_2374)/fabs_norm_2374.reshape(1,nspec)
                # Set +-1000 km/s to be 0
                absorption_norm_2374[:_noffset-14,:] = 1.
                absorption_norm_2374[_noffset+14:,:] = 1.
        

                XX = np.zeros((_npix_right_fit+_npix_left_fit, 2))
                lr = linear_model.LinearRegression(fit_intercept=False)
                coeff_abs = np.zeros((nuse,2,nspec))
                flux_absorption = np.zeros((nuse, vel.size, nspec))
                normalized_flux_absorption = np.zeros((nuse, vel.size, nspec))
                unified_absorption = np.zeros((vel.size,nspec))
                niter_max = 5

                # Emission component
                #print "Looping over all spectra..."
                for j in np.arange(nspec):
                    unified_old = absorption_norm_2374[:,j]
                    xemission = unified_emission[_noffset-_npix_left_fit:_noffset+_npix_right_fit,j]-1.
                    XX[:,1] = xemission 

                    niter = 0
                    while niter < niter_max:
                        niter += 1
                        # Now do all the 8 (7+2374) absorption lines
    
                        # General linear model
                        # Absorption component
                        XX[:,0] = 1.-unified_old[_noffset-_npix_left_fit:_noffset+_npix_right_fit]
                        for i, index in enumerate(use_indices):
                            # Linear Regression only
                            YY = 1.-flux[index, _noffset-_npix_left_fit:_noffset+_npix_right_fit,j]
                            lr.fit(XX, YY)
                            coeff_abs[i,:,j] = lr.coef_
                            #if niter < (niter_max-1): 
                            coeff_abs[i,1,j] = lr.coef_[1] if lr.coef_[1]<0 else 0.
                            flux_absorption[i,:,j] = flux[index,:,j]+coeff_abs[i,1,j]*(unified_emission[:,j]-1.) # Be careful with the sign here

                        # Normalization
                        fabs_norm = np.sum(1.-flux_absorption[:,_noffset-_npix_left:_noffset+_npix_right,j], axis=1)
                        # tau>>1 approximation AFTER Normalization Factor
                        flux_absorption[flux_absorption<0] = _EPS
                        normalized_flux_absorption[:,:,j] = 1.-(1.-flux_absorption[:,:,j])/fabs_norm.reshape(nuse,1)

                        # Composite (Unified) Absorption Profile; Only use unsaturated lines
                        unified_absorption[:,j] = 1.-np.sum(1.-normalized_flux_absorption[3:,:,j], axis=0)/use_indices[3:].size
                        # Set +-1000 km/s to be 0
                        unified_absorption[:_noffset-14,j] = 1.
                        unified_absorption[_noffset+14:,j] = 1.
                        unified_old = unified_absorption[:,j]

                    # Refit Mg II (Indices 1/2)
                    for i in np.arange(2)+1:
                        coeff_abs[i,0,j] = fabs_norm[i]
                        coeff_abs[i,1,j] = np.sum(flux_absorption[i,_noffset-_npix_left_fit:_noffset+_npix_right_fit,j]-flux[i,_noffset-_npix_left_fit:_noffset+_npix_right_fit,j], axis=0)\
                                        /np.sum(unified_emission[_noffset-_npix_left_fit:_noffset+_npix_right_fit,j]-1., axis=0)
                    # Refit Mg II (with two lines)
        
                    # Here we need another iteration to add *all* emission to the original spectrum and create the final 'true' absorption profile.

                # Dispersion
                tmp_flux = np.zeros((nspec*use_indices[3:].size, vel.size))
                for i in np.arange(nspec):
                    for j in np.arange(use_indices[3:].size):
                        tmp_flux[i*use_indices[3:].size+j,:] = normalized_flux_absorption[use_indices[3+j],:,i]
                unified_disp = np.std(tmp_flux-1., axis=0)

            if ((iewlum == 0) and (ibin == 0)):
                out_dtype = [('LINES', 'f8', use_indices.shape),
                             ('VEL', 'f8', vel.shape), 
                             ('FLUX', 'f8', flux_absorption.shape), # Observed absorption
                             ('FABS', 'f8', flux_absorption.shape), # "True" absorption
                             ('NORMFABS', 'f8', normalized_flux_absorption.shape), # Normalized "True" absorption
                             ('FNORM', 'f8', fabs_norm.shape), # Normalization
                             ('INDEX', 'i4', use_indices.shape), 
                             ('UNIFIEDABSORPTION', 'f8', unified_absorption.shape), 
                             ('UNIFIEDABSORPTION_DISP', 'f8', unified_disp.shape), 
                             ('UNIFIEDEMISSION', 'f8', unified_emission.shape),
                             ('UNIFIEDEMISSION_DISP', 'f8', unified_emission_disp.shape),
                             ('COEFF', 'f8', coeff_abs.shape),
                             ('FABS_2374', 'f8', absorption_norm_2374.shape)]
                ew_outstr = np.zeros(nbin, dtype=out_dtype)
                lum_outstr = np.zeros(nbin, dtype=out_dtype)

            tmp_outstr = np.array([(lines[use_indices], vel, flux[use_indices,:], flux_absorption, \
                                    normalized_flux_absorption, fabs_norm, use_indices, unified_absorption, unified_disp, \
                                    unified_emission, unified_emission_disp, coeff_abs, absorption_norm_2374)], 
                                    dtype=out_dtype)
            if (iewlum == 0):
                ew_outstr[ibin] = tmp_outstr
            else:
                lum_outstr[ibin] = tmp_outstr


    fits = fitsio.FITS(outfile, 'rw', clobber=overwrite)
    fits.write(ew_outstr)
    fits.write(lum_outstr)
    fits.close()

    # Correct the original spectrum
    #print "Correting the original spectra..."
    temp = correct_emission_infill_binoii(overwrite=overwrite, bootstrap=bootstrap)
   
    #return (loglam, corrected)
    return True

#def temp_do_velocity_nonparametric(bootstrap=False):
#    """
#    """
#    dloglam = 1E-4
#    absorption = unify_absorptionline_profile_readin(bootstrap=bootstrap)
#    corrected = corrected_velspace_flux_readin(bootstrap=bootstrap)
#
#    lines = absorption['LINES']
#    vel = absorption['VEL']
#    flux = absorption['FLUX']
#    flux_abs = corrected['FLUX']
#    unified_flux = absorption['UNIFIEDABSORPTION']
#
#    if (not bootstrap):
#        vpercent = np.zeros((_percentlist.size, lines.size))
#        vpercent_abs = np.zeros((_percentlist.size, lines.size))
#        tflux = np.zeros(lines.shape)
#        tfabs = np.zeros(lines.shape)
#
#        for i, thisline in enumerate(lines):
#            vpercent[:,i], tflux[i] = velocity_nonparametric(vel, flux[i,:])
#            vpercent_abs[:,i], tfabs[i] = velocity_nonparametric(vel, flux_abs[i,:])
#        tflux = tflux*dloglam*np.log(10.)*lines
#        tfabs = tfabs*dloglam*np.log(10.)*lines
#        unified_vpercent, tmp = velocity_nonparametric(vel, unified_flux)
#    else:
#        nspec = (unified_flux.shape)[1]
#        vpercent = np.zeros((_percentlist.size,lines.size, nspec))
#        vpercent_abs = np.zeros((_percentlist.size,lines.size, nspec))
#        tflux = np.zeros((lines.size, nspec))
#        tfabs = np.zeros((lines.size, nspec))
#        unified_vpercent = np.zeros((_percentlist.size, nspec))
#
#        for j in np.arange(nspec):
#            for i, thisline in enumerate(lines):
#                vpercent[:,i,j], tflux[i,j] = velocity_nonparametric(vel, flux[i,:,j])
#                vpercent_abs[:,i,j], tfabs[i,j] = velocity_nonparametric(vel, flux_abs[i,:,j])
#            tflux[:,j] = tflux[:,j]*dloglam*np.log(10.)*lines
#            tfabs[:,j] = tfabs[:,j]*dloglam*np.log(10.)*lines
#            unified_vpercent[:,j], tmp = velocity_nonparametric(vel, unified_flux[:,j])
#
#    # Write out
#    out_dtype = [('LINES', 'f8', lines.shape),
#                 ('PERCENT', 'f8', _percentlist.shape),
#                 ('FLUX_PERCENT', 'f8', vpercent.shape), 
#                 ('FABS_PERCENT', 'f8', vpercent_abs.shape), 
#                 ('UNIFIEDPERCENT', 'f8', unified_vpercent.shape),
#                 ('TFLUX', 'f8', tflux.shape),
#                 ('TFABS', 'f8', tfabs.shape)]
#    outstr = np.array([(lines, _percentlist, vpercent, vpercent_abs, unified_vpercent, tflux, tfabs)],
#                       dtype=out_dtype)
#    #fits = fitsio.FITS(outfile, 'rw', clobber=overwrite)
#    #fits.write(outstr)
#    #fits.close()
#
#    return outstr[0]
#
