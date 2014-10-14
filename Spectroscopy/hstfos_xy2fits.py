"""
Convert *.xy files to fits file
"""
# http://lithops.as.arizona.edu/~jill/QuasarSpectra/
# A Uniform Analysis of the Lyman Alpha Forest at z = 0 - 5: III. HST FOS Spectral Atlas
# http://lithops.as.arizona.edu/~jill/QuasarSpectra/HSTFOS/table.html
# Units:
#    -- The wavelength is given in units of angstroms.
#    -- The flux, error in the flux, and continuum are all given in units of 10^-16 erg cm^-2 s^-1 Å^-1
#    -- The flag can be ignored as it used only by specific software.

import fitsio
import numpy as np
import string
import os.path
import warnings
import subprocess

def xy2fits(infile, outfile=None):

    # check if file exists
    if os.path.isfile(infile):

       # define outfile
       # is vs. == remember the difference?
       if outfile is None:
          outfile = infile.replace('.XY', '.fits')

       # set up readin in format
       format = "f8, f8, f8, f8, i4"
       names = ['wave', 'flux', 'error', 'continuum', 'flag']

       # read in
       hstfos_obj = np.genfromtxt(infile, dtype=format, names=names)

       # set up write out format
       # Could use np.char module too, but not necessary
       out_size = str(hstfos_obj.size)
       out_format = format.replace('f', '('+out_size+',)f')
       out_format = out_format.replace('i', '('+out_size+',)i')
       out_dtype = np.dtype({'names': hstfos_obj.dtype.names, 'formats': out_format.split(', ')})

       # initialize output
       outstr = np.zeros(1, dtype=out_dtype)

       # copy to output
       outstr['wave'] = hstfos_obj['wave']
       outstr['flux'] = hstfos_obj['flux']
       outstr['error'] = hstfos_obj['error']
       outstr['continuum'] = hstfos_obj['continuum']
       outstr['flag'] = hstfos_obj['flag']

       # write out output
       fits = fitsio.FITS(outfile, 'rw', clobber=True)
       fits.write(outstr)
       fits.close()

       #
       return
    else:
       warnings.warn("Can't find the file: "+infile)
       return

# Main Program
# spectrographs
basepath = '/home/gz323/SDATA/Quasars/HSTFOS'
basefile = ('h130.XY', 'h190.XY', 'h270.XY')

# read in qso list
qso = fitsio.read('table1_processed_degree.fits')
for thisqso in qso:
    subpath = ((thisqso['Name'].strip()).replace('+','p')).replace('-','m')
    for thisbase in basefile:
        infile = os.path.join(basepath, subpath, thisbase)
        xy2fits(infile)

print "I'm done! Some quick check by looking at the file Numbers."
print "There are ", (subprocess.check_output("ls -l */*.XY | wc -l", shell=True)).split()[0], " .XY files"
print "and ", (subprocess.check_output("ls -l */*.fits | wc -l", shell=True)).split()[0], ".fits files"

