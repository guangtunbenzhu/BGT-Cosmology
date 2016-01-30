__license__ = "MIT"
__author__ = "Guangtun Ben Zhu (BGT) @ Johns Hopkins University"
__startdate__ = "2016.01.11"
__name__ = "cnn"
__module__ = "Network"

__lastdate__ = "2016.01.19"
__version__ = "0.01"

import numpy as np
import scipy.optimize as op

def quick_amplitude(x, y, x_err, y_err):
    """
    Assume y = ax
    Calculate the amplitude only.
    """
    #x[x<0] = 1E-5
    #y[y<0] = 1E-5
    xy = x*y
    xx = x*x
    xy[xy<0] = 1E-10

    A = np.ones(x.shape[0])
    for i in np.arange(5):
        weight = 1./(np.square(y_err)+np.square(A).reshape(A.size,1)*np.square(x_err))
        #weight = 1./(np.square(y_err)+np.square(A)*np.square(x_err))
        A = np.einsum('ij, ij->i', xy, weight)/np.einsum('ij, ij->i', xx, weight)

    chi2 = np.einsum('ij, ij->i', np.square(A.reshape(A.size,1)*x - y), weight)
    #chi2 = np.einsum('ij, ij->i', np.square(A*x - y), weight)

    return (A, chi2)

def quick_totalleastsquares(x, y, x_err, y_err):
    """
    Assume y = ax
    Calculate the amplitude only.
    """
    chi2 = lambda A: np.einsum('ij, ij->i', np.square(y-A.reshape(A.size,1)*x), \
                        1./(np.square(y_err)+np.square(A).reshape(A.size,1)*np.square(x_err)))

    xy = x*y
    xx = x*x
    xy[xy<0] = 1E-10
    A0 = 0.
    for i in np.arange(5):
        weight = 1./(np.square(y_err)+np.square(A0)*np.square(x_err))
        A0 = np.einsum('ij, ij->i', xy, weight)/np.einsum('ij, ij->i', xx, weight)

    res = op.minimize(chi2, A0)

    return (res, chi2(res['x']))

