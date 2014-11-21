
""" 
Useful small tools for spectroscopic analysis
"""

import numpy as np
import fitsio
from scipy import interpolate
from astropy.wcs import WCS

def wcs_getval(longitude, latitude, infiles, interp=True, noloop=True, verbose=True, transpose=False):
    '''
    Extract values from a fits file of celestial sky map with WCS standards.
    '''


    # Inputs must be arrays with at least 1 dimension to be indexed
    infiles = np.ravel(infiles)
    if (infiles.size != 1 and infiles.size !=2):
       raise ValueError("The number of files to read must be either 1 or 2.")

    # Inputs must be arrays with at least 1 dimension to be indexed
    longitude = np.ravel(longitude)
    latitude = np.ravel(latitude)
    if (longitude.size !=latitude.size):
       raise ValueError("Longitudes and latitudes must have the same length.")

    # Initialization
    value = np.zeros(longitude.size, dtype=np.float_)

    # Looping over files
    # for ifile in np.column_stack((np.arange(infiles.size),infiles)):
    for ifile, this_infile in zip(np.arange(infiles.size), infiles):
        # this_infile = infiles[ifile]
        if infiles.size == 1:
           indx = np.ones(longitude.size, dtype=np.bool_)
        # If more than 1 files, assume the 1st file is for northern hemisphere, and 2nd for southern
        else:
           if ifile == 0: 
              indx = latitude>=0.
           if ifile == 1:
              indx = latitude<0.

        if np.count_nonzero(indx)>0:

           # Convert world coordinates to fractional pixel values
           hdr = fitsio.read_header(this_infile)
           w = WCS(hdr)
           xr, yr = w.wcs_world2pix(longitude[indx], latitude[indx], 0)
           print xr, yr
           xpix1, ypix1 = np.fix(xr), np.fix(yr)
           ixmin, ixmax = np.max([np.min(xpix1)-3, 0]), np.min([np.max(xpix1)+3, hdr['NAXIS1']])
           iymin, iymax = np.max([np.min(ypix1)-3, 0]), np.min([np.max(ypix1)+3, hdr['NAXIS2']])

           # Noloop: read the full image
           if noloop:
              # Be careful: SFD dust maps were created by IDL, so rows and columns are swapped
              image = fitsio.read(this_infile)
              if transpose: image=image.T
              subimage = image[ixmin:ixmax, iymin:iymax]
              # The degrees of spline function are set to be quadratic
              # The values are assumed to be at the centers of cells by default, thus the +0.5
              if interp:
                 fimage = interpolate.RectBivariateSpline(np.arange(subimage.shape[0])+ixmin+0.5, 
                          np.arange(subimage.shape[1])+iymin+0.5, subimage, kx=2, ky=2)
                 value[indx] = fimage(xr, yr, grid=False)
              else:
                 raise ValueError("Sorry! I haven't implemented nearest-neighbor interpolation yet. Use interp=True (defaulty).")
           else:
              raise ValueError("Sorry! I haven't implemented subset reading yet. Use noloop=True (default).")

    return value

