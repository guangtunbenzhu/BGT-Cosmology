from __future__ import division
import numpy as np
import cosmology as cosmo
from scipy.spatial import cKDTree as KDT

_CosPar={'Omega_M':0.3, 'Omega_L':0.7, 'Omega_b':0.045, 'Omega_nu':1e-5, 'n_degen_nu':3., 'h':0.7, 'sigma_8':0.8, 'ns':0.96}

def spherical_to_cartesian(scoord):
    """
    http://mathworld.wolfram.com/SphericalCoordinates.html
    scoord[nobj,3]
    0: r - distance
    1: ra - azimuthal, in deg
    2: dec - pi/2-polar, in deg
    """
    x = scoord[:,0]*np.cos(scoord[:,1]/180.*np.pi)*np.cos(scoord[:,2]/180.*np.pi)
    y = scoord[:,0]*np.sin(scoord[:,1]/180.*np.pi)*np.cos(scoord[:,2]/180.*np.pi)
    z = scoord[:,0]*np.sin(scoord[:,2]/180.*np.pi)
    return np.c_[x,y,z]

def cartesian_to_spherical(ccoord):
    """
    ccoord[nobj,3]
    0: x
    1: y
    2: z
    """
    r = np.sqrt(np.sum(np.power(ccoord,2)))
    ra = np.arctan2(ccoord[:,1], ccoord[:,0])/np.pi*180.
    if r != 0.0:
       dec = np.arcsin(ccoord[:,2]/r)/np.pi*180.
    else:
       dec = 0.*np.sign(ccoord[:,2])/np.pi*180.
    return np.c_[r, ra, dec]

def radecz_to_comoving(zcoord):
    """
    zcoord[nobj,3]
    0: z - redshift
    1: ra - azimuthal
    2: dec - pi/2-polar
    """
    # use transverse comoving distance dM instead?
    dc = cosmo.comoving_distance(zcoord[:,0], _CosPar)
    scoord = np.c_[dc, zcoord[:,1], zcoord[:,2]]
    return spherical_to_cartesian(scoord)

def comoving_neighbors(zcoord1, zcoord2, nthneighbor=1, maxscale=None):
    """
    """
    ccoord1 = radecz_to_comoving(zcoord1)
    ccoord2 = radecz_to_comoving(zcoord2)
    kdt = KDT(ccoord2)
    if nthneighbor == 1:
        indx2 = kdt.query(ccoord1)[1]
    elif nthneighbor > 1:
        indx2 = kdt.query(ccoord1, nthneighbor)[1][:, -1]
    else:
        raise ValueError('{0}th neighbor cannot be found'.format(nthneighbor))

    ds = np.linalg.norm(ccoord1-ccoord2[indx2,:], axis=1)
    indx1 = np.arange(ds.size)
    if maxscale != None:
       iselect = ds < maxscale
       indx1 = indx1[iselect]
       indx2 = indx2[iselect]
       ds = ds[iselect]

    return indx1, indx2, ds

