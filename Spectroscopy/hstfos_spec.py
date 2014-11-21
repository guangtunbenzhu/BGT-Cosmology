
""" 
Useful small tools for spectroscopic analysis
"""

import numpy as np
import spec_utils
import spectrograph
import fitsio
import os
from progressbar import ProgressBar
import numpy.lib.recfunctions as recfunctions
from pydl.pydlspec2d.spec2d import combine1fiber

pbar = ProgressBar()

astrodata_path = {'HSTFOS': os.path.join(os.environ['HOME'], 'AstroData/Quasars/HSTFOS'),
                  'HSTCOS': os.path.join(os.environ['HOME'], 'AstroData/Quasars/HSTCOS')}

savewave = False
# Read in the master catalog
basepath = astrodata_path['HSTFOS']
infile = os.path.join(basepath, 'table1_master.fits')
qso = fitsio.read(infile)

# The gratings
FOS_gratings = (spectrograph.FOSG130H, spectrograph.FOSG190H, spectrograph.FOSG270H)

# Wavelength bins
# Master bin (from 900. to 14400.)
master_loglam = spec_utils.get_loglambda()
# G130H wavelength bin
FOSG130H_blue_index = np.searchsorted(master_loglam, np.log10(FOS_gratings[0].minwave))
FOSG130H_red_index = np.searchsorted(master_loglam, np.log10(FOS_gratings[0].maxwave+FOS_gratings[0].width), side='Right')
#FOSG130H_loglam = master_loglam[FOSG130H_blue_index:FOSG130H_red_index]
# G190H wavelength bin
FOSG190H_blue_index = np.searchsorted(master_loglam, np.log10(FOS_gratings[1].minwave))
FOSG190H_red_index = np.searchsorted(master_loglam, np.log10(FOS_gratings[1].maxwave+FOS_gratings[1].width), side='Right')
#FOSG190H_loglam = master_loglam[FOSG190H_blue_index:FOSG190H_red_index]
# G270H wavelength bin
FOSG270H_blue_index = np.searchsorted(master_loglam, np.log10(FOS_gratings[2].minwave))
FOSG270H_red_index = np.searchsorted(master_loglam, np.log10(FOS_gratings[2].maxwave+FOS_gratings[2].width), side='Right')
#FOSG270H_loglam = master_loglam[FOSG270H_blue_index:FOSG270H_red_index]

# Use slicing to create views
new_index = ((FOSG130H_blue_index, FOSG130H_red_index), 
                    (FOSG190H_blue_index, FOSG190H_red_index),
                    (FOSG270H_blue_index, FOSG270H_red_index))
new_size = FOSG270H_red_index - FOSG130H_blue_index

basefile = ['h130.fits', 'h190.fits', 'h270.fits']
out_basefile = 'HSTFOS_commonwave_spec.fits'

# set up outputs
tmp_formats = "f8, f8"
out_formats = tmp_formats.replace('f', '('+str(new_size)+',)f')
out_names = ['flux', 'ivar']
out_dtype = np.dtype({'names':out_names, 'formats':out_formats.split(', ')})
tmp_outstr = np.zeros(1, dtype=out_dtype)

# Save the common wavelength grid information
if savewave: 
   outwave = np.zeros(1, dtype=[('wave','f8', (master_loglam.size,)),('min_index','i4'),('max_index','i4')])
   outwave['wave'] = np.power(10., master_loglam)
   outwave['min_index'] = FOSG130H_blue_index 
   outwave['max_index'] = FOSG270H_red_index
   wave_outfile = os.path.join(basepath, 'HSTFOS_commonwave_wave.fits')
   fits = fitsio.FITS(wave_outfile, 'rw', clobber=True)
   fits.write(outwave)
   fits.close()

# Interpolation + Stitching (no Stitching yet)
# for i in np.arange(qso.size):
for i in np.arange(10):
    # Progress bar
    pbar.update(i+1)

    # Sub-directory for the quasar
    subpath = ((qso[i]['Name'].strip()).replace('+','p')).replace('-','m')

    # Output file
    thisqso_outfile = os.path.join(basepath, subpath, out_basefile)

    # Initialization of temporary arrays
    newflux = np.zeros((3, new_size))
    newivar = np.zeros((3, new_size))

    for j in np.arange(len(basefile)):
        thisqso_infile = os.path.join(basepath, subpath, basefile[j])

        if os.path.isfile(thisqso_infile): 
           # Read in
           indata = fitsio.read(thisqso_infile)
           inivar = 1./(indata[0]['error'])**2
           inivar = np.where(np.isfinite(inivar),inivar,0.0)

           # Interpolation
           tmp_flux, tmp_ivar=combine1fiber(np.log10(indata[0]['wave']), indata[0]['flux'], 
               objivar=inivar, newloglam=master_loglam[new_index[j][0]:new_index[j][1]])
#              objivar=1./(indata[0]['error'])**2, newloglam=master_loglam[new_index[j][0]:new_index[j][1]])

           # Assignment
           left_index = new_index[j][0] - new_index[0][0]
           right_index = new_index[j][1] - new_index[0][0]
           newflux[j, left_index:right_index] = tmp_flux


           newivar[j, left_index:right_index] = tmp_ivar

    # Stitch/Co-add spectra. Simple ivar-weighted mean. Likely need to revisit.
    # Bottleneck?
    # Also assuming most of objects have multiple spectra, otherwise should co-add only a portion of the spectra
    outivar = np.sum(newivar, axis=0)
    # Will using masked_array be faster?
    outflux = np.sum(newflux*newivar, axis=0)/(outivar + (outivar == 0.))*(outivar > 0.)
    outivar = outivar*(outivar > 0.)

    # Output
    tmp_outstr['flux'] = outflux
    tmp_outstr['ivar'] = outivar
    outstr = recfunctions.merge_arrays([qso[j], tmp_outstr], flatten=True, usemask=False)

    # Write out output
    fits = fitsio.FITS(thisqso_outfile, 'rw', clobber=True)
    fits.write(outstr)
    fits.close()
