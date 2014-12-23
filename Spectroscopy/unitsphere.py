from __future__ import division
import numpy as np


def spherical_to_cartesian(ra, dec, r):
    """
    http://mathworld.wolfram.com/SphericalCoordinates.html
    ra - azimuthal
    dec - pi/2-polar
    r - distance
    """
    x = r*np.cos(ra)*np.cos(dec)
    y = r*np.sin(ra)*np.cos(dec)
    z = r*np.sin(dec)
    return (x,y,z)

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2+y**2+z**2)
    ra = np.arctan2(y, x)
    if r != 0.0:
       dec = np.arcsin(z/r)
    else:
       dec = 0.*np.sign(z)
    return (ra, dec, r)


