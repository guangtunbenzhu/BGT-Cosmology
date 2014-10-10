
""" 
Useful small tools for spectroscopic analysis
"""

import numpy as np
import spec_utils
import spectrograph

# Read in the master catalog
basepath = '/home/gz323/SDATA/Quasars/HSTFOS'
infile = os.join.path(basepath, 'table1_master.fits')
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
new_loglam_index = ((FOSG130H_blue_index, FOSG130H_red_index), 
                    (FOSG190H_blue_index, FOSG190H_red_index),
                    (FOSG270H_blue_index, FOSG270H_red_index))

basefile = ['h130.fits', 'h190.fits', 'h270.fits']

# Interpolation + Stitching
for i in np.arange(qso.size):
    subpath = ((qso[i]['Name'].strip()).replace('+','p')).replace('-','m')
    for j in np.arange(len(basefile)):
        thisqso_infile = os.path.join(basepath, subpath, basefile[j])
        if os.path.isfile(thisqso_infile): 
           indata = fitsio.read(thisqso_infile)
           inivar = 1./(indata.error)**2
           tmp_flux, tmp_ivar=combine1fiber(inloglam, influx, objivar=inivar, newloglam=master_loglam(new_loglam_index[j][0]:new_loglam_index[j][1]))

           # Stitching
