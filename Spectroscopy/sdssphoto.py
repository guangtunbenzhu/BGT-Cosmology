from __future__ import division, print_function
import os
import numpy as np
import healpy as hp
import fitsio
import datapath
from progressbar import ProgressBar
from glob import glob


def sheldon_pofz_filelist():
    filename = os.path.join(datapath.sdss_path(), 'PhotozWeight/allruns.fits')
    return fitsio.read(filename)

def read_sheldon_pofz(basefile):
    filename = os.path.join(datapath.sdss_path(), 'PhotozWeight', basefile)
    return fitsio.read(filename)

# re-bin photometric data (r<21.5) into healpix pixels
def sheldon_run_to_healpix():
    allruns = sheldon_pofz_filelist()
    nrun = allruns.size

    nside = 2
    npixel = hp.nside2npix(nside)
    pixels = np.arange(npixel)

    # read it out
    pbar = ProgressBar(maxval=nrun).start()
    for i, thisrun in zip(np.arange(nrun),allruns):
        pbar.update(i)
        data = read_sheldon_pofz(thisrun['FILE'])
        theta = np.pi/2.-data['dec']/180.*np.pi
        phi = data['ra']/180.*np.pi
        run_pixels = hp.ang2pix(nside, theta, phi)
        for thispixel in pixels:
            outdata = data[run_pixels==thispixel]
            if outdata.size > 1:
               subpath = os.path.join(datapath.sdss_path(), 'PhotozWeight', 'Healpix', "{0:02d}".format(thispixel))
               if not os.path.exists(subpath): os.makedirs(subpath)
               outfile = os.path.join(subpath, 'sub_'+thisrun['FILE'])
               # print("Writing {0!r}".format(outfile))
               fits = fitsio.FITS(outfile, 'rw', clobber=True)
               fits.write(outdata)
               fits.close()

def sheldon_healpix_uniteruns():
    nside = 2
    npixel = hp.nside2npix(nside)
    pixels = np.arange(npixel)
    #pbar = ProgressBar(maxval=npixel).start()
    for thispixel in pixels:
        #pbar.update(thispixel)
        subpath = os.path.join(datapath.sdss_path(), 'PhotozWeight', 'Healpix', "{0:02d}".format(thispixel))
        print(subpath)
        if os.path.exists(subpath):
           # find all files
           allfiles = glob(subpath+"/*.fits.gz")
           # print(allfiles)
           outdata = np.asarray([])
           for thisfile in allfiles:
               thisdata = fitsio.read(thisfile)
               if outdata.size==0:
                  outdata = thisdata
               else:
                  outdata = np.r_[outdata,thisdata]
           outfile = subpath+"/allruns_pixel{0:02d}.fits".format(thispixel)
           fits = fitsio.FITS(outfile, 'rw', clobber=True)
           fits.write(outdata)
           fits.close()

