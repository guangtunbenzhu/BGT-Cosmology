"""
Convert *.txt files to fits file
"""
# http://www.stsci.edu/science/starburst99/docs/templates/
# An Ultraviolet Spectroscopic Atlas of Local Starbursts and Star-forming Galaxies: The Legacy of FOS and GHRS
# Units:
#    -- The wavelength is given in units of angstroms.
#    -- The flux in the flux, in units of erg cm^-2 s^-1 AA^-1
#    -- The error in the flux, in units of 10^{-14} erg cm^-2 s^-1 AA^-1

from os.path import isfile, join
import fitsio
import numpy as np
import string
import datapath
import warnings
import subprocess

_starburstfile = 'catalog.txt'
_subpath = 'starburst/fos_ghrs_spectra'
def starburst_filename():
    path = datapath.hstfos_path()
    return join(path, _subpath, _starburstfile)

def txt2fits(infile, outfile=None):

    # check if file exists
    if isfile(infile):

       # define outfile
       # is vs. == remember the difference?
       if outfile is None:
          outfile = infile.replace('.txt', '.fits')

       # set up readin in format
       formats = "f8, f8, f8"
       names = ['wave', 'flux', 'error']

       # read in
       hstfos_obj = np.genfromtxt(infile, dtype=formats, names=names)

       # set up write out format
       # Could use np.char module too, but not necessary
       out_size = str(hstfos_obj.size)
       out_formats = formats.replace('f', '('+out_size+',)f')
       out_formats = out_formats.replace('i', '('+out_size+',)i')
       out_dtype = np.dtype({'names': hstfos_obj.dtype.names, 'formats': out_formats.split(', ')})

       # initialize output
       outstr = np.zeros(1, dtype=out_dtype)

       # copy to output
       outstr['wave'] = hstfos_obj['wave']
       outstr['flux'] = hstfos_obj['flux']
       outstr['error'] = hstfos_obj['error']

       # write out output
       fits = fitsio.FITS(outfile, 'rw', clobber=True)
       fits.write(outstr)
       fits.close()

       #
       return
    else:
       warnings.warn("Can't find the file: "+repr(infile))
       return

# Main Program

# read in qso list
catfile = starburst_filename()
gal = np.genfromtxt(catfile, dtype=[('gal', 'S15')])

path = datapath.hstfos_path()
for thisgal in gal:
    specfile = thisgal['gal']+'.txt'
    infile = join(path, _subpath, specfile)
    txt2fits(infile)

#print "I'm done! Some quick check by looking at the file Numbers."
#print "There are ", (subprocess.check_output("ls -l */*.txt | wc -l", shell=True)).split()[0], " .txt files"
#print "and ", (subprocess.check_output("ls -l */*.fits | wc -l", shell=True)).split()[0], ".fits files"

