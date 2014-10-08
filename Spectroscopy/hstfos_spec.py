
""" 
Useful small tools for spectroscopic analysis
"""

import numpy as np
import logging
from scipy import integrate


# Read in the master catalog
basepath = '/home/gz323/SDATA/Quasars/HSTFOS'
infile = os.join.path(basepath, 'table1_master.fits')
qso = fitsio.read(infile)

basefile_base = ('h130', 'h190', 'h270')
basefile = ('h130.fits', 'h190.fits', 'h270.fits')

for i in np.arange(qso.size):
    subpath = ((qso[i]['Name'].strip()).replace('+','p')).replace('-','m')
    for j in np.arange(len(basefile)):
        if qso[i][basefile_base[j]]==1:
           thisqso_infile = os.path.join(basepath, subpath, basefile[j])
           indata = fitsio.read(thisqso_infile)
           newflux, newivar=combine1fiber(inloglam, influx, objivar=inivar, newloglam=newloglam)


