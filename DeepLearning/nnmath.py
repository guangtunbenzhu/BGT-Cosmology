__license__ = "MIT"
__author__ = "Guangtun Ben Zhu (BGT) @ Johns Hopkins University"
__startdate__ = "2016.01.19"
__name__ = "nnmath"
__module__ = "Network"

__lastdate__ = "2016.01.19"
__version__ = "0.01"
__comments__ = "math utils for neural net work"

import numpy as np
from scipy.special import expit

# some small number epsilon
_EPS = 1E-5

# collect the activation and cost functions in dictionaries
# define activation function
#### logistic sigmoidal
sigmoid = expit
sigmoid_deriv = lambda x: (0.5/np.cosh(0.5*x))**2
#### to add: softmax
def softmax(w, t=1.0):
    assert w.ndim==2, "the input must be in format of [n_vector, ndim_vector]."
    maxes = np.amax(w, axis=1).reshape(w.shape[0],1)
    e = numpy.exp((w-maxes)/1.)
    dist = e/numpy.sum(e, axis=1)
    return dist
def softmax_deriv(wsm): # revisit
    """
    """
    assert wsm.ndim==2, "the input must be in format of [n_vector, ndim_vector]."
    tmp = np.eye((wsm.shape[1]).reshape(1, wsm.shape[1], wsm.shape[1]))
    tmp_diag_indices = np.diag_indices(wsm.shape[1])
    output = np.tile(tmp, np.r_[wsm.shape[0], np.ones(3)])
    for i in np.arange(wsm.shape[0]):
        output[i, tmp_diag_indices] = wsm[i, :]
        output[i, ...]  += np.matmul(wsm[i,:].reshape(wsm.shape[1], 1),wsm[i,:].reshape(1,wsm.shape[1],1))
    return

# define cost function
#### quadratic
quadratic = lambda a, y: 0.5*np.sum(np.square(a-y), axis=1)
quadratic_deriv = lambda a, y: a-y
quadratic_delta = lambda z, a, y: (a-y)*sigmoid_deriv(z)
#### cross entropy
crossentropy = lambda a, y: np.sum(y*np.log(a)+(1.-y)*np.log(1.-a), axis=1)
crossentropy_deriv = lambda a, y: np.sum(y/a-(1.-y)/(1.-a), axis=1)
crossentropy_delta = lambda z, a, y: a-y
#### to add: log-likelihood
loglikelihood = lambda a, y: -np.sum(y*np.log(a), axis=1)
loglikelihood_deriv = lambda a, y: np.sum(a-y)
#loglikelihood_delta = lambda z, a, y: 

# This is the fastest convolution I can think of with Python+Numpy
def conv2d(x, y, mode='valid'):
    """
    """
    assert y.ndim == 2, "I can only do one kernel at a time."
    #assert x.ndim >= 2, "I can only do one kernel at a time."
    if (mode == 'full'):
        newshape = (x.shape[-2]+y.shape[0]-1, x.shape[-1]+y.shape[1]-1)
        if x.ndim > 2:
               x_newshape = np.r_[np.asarray(x.shape[:-2]), np.asarray(newshape)]
        print(x.shape, x_newshape, newshape, y.shape)
        return np.fft.irfft2(np.fft.rfft2(x, x_newshape) * np.fft.rfft2(y, newshape))
    elif (mode == 'same'):
        newshape = (x.shape[-2], x.shape[-1])
        return np.fft.irfft2(np.fft.rfft2(x) * np.fft.rfft2(y, newshape))
    elif (mode == 'valid'):
        newshape = (x.shape[-2], x.shape[-1])
        return np.fft.irfft2(np.fft.rfft2(x) * np.fft.rfft2(y, newshape))[..., y.shape[0]-1:, y.shape[1]-1:]
    else:
        raise ValueError("Only 'full', 'same', and 'valid' FFT modes are supported.")


def maxpooling22_down(input_image):
    """
    down-pooling with pool size 2x2, with numpy.fmax
    """
    nimages = np.prod(input_image.shape[:-2]) # collapse the first two arrays to speed up
    newshape = np.r_[nimages, np.asarray([input_image.shape[-2]//2, 2, input_image.shape[-1]//2, 2])]
    image_view = input_image.reshape(newshape)
    xtmp = np.fmax(image_view[...,0], image_view[...,1])
    newshape_out = np.r_[np.asarray(input_image.shape[:-2]), np.asarray([input_image.shape[-2]//2, input_image.shape[-1]//2])]
    return np.fmax(xtmp[...,0,:], xtmp[...,1,:]).reshape(newshape_out)

def maxpooling22_up(input_image):
    """
    up-pooling with pool size 2x2
    for dz[l+1]/da[l] and detal[l]
    """
    #nimages = np.prod(input_image.shape[:-2])
    newshape = np.r_[np.asarray(input_image.shape[:-2]), np.asarray([input_image.shape[-2], 2, input_image.shape[-1], 2])]
    output_image = np.zeros(newshape)
    for i in (0,1):
        for j in (0,1):
            output_image[...,i,:,j] = input_image
    newshape_out = np.r_[np.asarray(input_image.shape[:-2]), np.asarray([input_image.shape[-2]*2, input_image.shape[-1]*2])]
    return output_image.reshape(newshape_out)

activationfunc = {'sigmoid': {'function': sigmoid, 'derivative': sigmoid_deriv}}
#                   'rectified': {'function': rectified, 'derivative':rectified_deriv}}
costfunc = {'quadratic': {'function': quadratic, 'derivative': quadratic_deriv, 'delta': quadratic_delta},
             'crossentropy': {'function': crossentropy, 'derivative':crossentropy_deriv, 'delta': crossentropy_delta}}


