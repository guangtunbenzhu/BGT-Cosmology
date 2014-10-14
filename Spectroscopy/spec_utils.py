
""" 
Useful small tools for spectroscopic analysis
"""

import numpy as np

# create a center wavelength grid with constant width in log (i.e., velocity) space:
# input is in Angstrom, output is log10(lambda/Angstrom)
get_loglambda = lambda minwave=400., maxwave=12800., dloglam=5.E-5: np.arange(np.log10(minwave), np.log10(maxwave)+dloglam, dloglam)

