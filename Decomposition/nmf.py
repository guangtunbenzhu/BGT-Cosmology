"""
Code for Non-negative Matrix Factorization (NMF)
Why class: taking advantage of encapsulation and inheritance in this case

n_instance: Number of instances (input data points)
n_dimention: Number of dimension of each instance

To-do: 
    - Sparese matrix
"""

from __future__ import division
import scipy.linalg as LA

__all__ = [
    'NMF']

class _BaseNMF(object):
    """ Base class for NMF
    Meant to be subclassed
    Likely need to revisit if we want a more general matrix factorization class
    """

    def __new__(cls):
    
class MultiplicativeNMF(_BaseNMF):
    """NMF with multiplicative update rules

    Parameters
    ----------
    n_component : int | None
        Number of components

    init : 'nndsvd' | 'random' | 'user'

    sparseness : 'data' | 'component' | None, default: None

    tol : double, default: 1E-5

    maxiter : int, default: 1000

    random_state : int or RandomState

    Attributes
    ----------
    components_ : array, [n_components, n_dimention]

    chi_squared_ : double
        Frobenius norm: ``||X - WH||^2``

    n_iter_ : int
        Number of iterations

    Examples
    --------
    >>> import numpy as np
    ... X.shape = (100, 3000)
    >>> From nmf import  NMF
    >>> nmfbasis = MultiplicativeNMF(n_component=12, init='random', seed='0.13')
    >>> nmfbasis.fit(X)

    References
    ----------
    - Lee & Seung (2001) "Algorithms for Non-negative Matrix Factorization"
    - Blanton & Roweis, Kcorrect, ApJ, 133, 734 (2007)
    - jhusdss_nmf_engine.pro in jhusdss, Guangtun Ben Zhu
    - nmf.py in scikit-learn
    """

    def __init__(self, n_component=None, initial=None, sparseness=None, ranseed=None, 
                 tol=1E-5, maxiter=1000):
        self._n_component = n_component
        self._initial = initiial
        self._tol = tol
        if sparseness not in (None, 'data', 'component'):
           raise ValueError("Invalid sparseness parameter: got %r instead of one of %r" %
                           (sparseness, (None, 'data', 'components')))
        self._sparseness = sparseness
        self._maxiter = maxiter
        self._ranseed = ranseed

    def _initialize(self, X):
        if initial = 'random':
           W, H = pass
        return W, H

    # Main method
    def construct(self, X, Weight=None)
        """Construct (Learn) an NMF model for input data matrix X 

        """

        # Core code

        XWeight = X*Weight
        W, H = self._initialize(X)

        # np.dot: 1D - Euclidean norm; 2D - Frobenius norm
        # scipy.linalg.norm - faster than numpy.linalg.norm for 2D Frobenius norm?
        chi_squared = LA.norm((X-np.dot(W,H))*Weight)
        chi_squared_old = 1.e+100
        niter = 1


        # Need a sparse version of this
        while niter < self._maxiter and np.fabs(chi_squared-chi_squared_old)/chi_squared_old > tol):
            # Update H first. Does the order matter?
            H_up = np.dot(W.T, XWeight)
            H_down = np.dot(W.T, np.dot(W,H)*Weight)
            H = H*H_up/H_down

            # Update W
            W_up = np.dot(XWeight, H.T)
            W_down = np.dot(np.dot(W,H)*Weight, H.T)
            W = W*W_up/W_down

            # chi_squared
            chi_squared_old = chi_squared
            chi_squared = LA.norm((X-np.dot(W,H))*Weight)

            # Some quick check. May need its error class ...
            if not np.isfinite(chi_squared):
               raise ValueError("NMF construction failed, likely due to missing data")

            if verbose: 
               pass

            niter += 1

        self.basisvector = H
        self.coefficient = W
        self.niter = niter
        return 'Success'

