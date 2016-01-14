__license__ = "MIT"
__author__ = "Guangtun Ben Zhu (BGT) @ Johns Hopkins University"
__startdate__ = "2016.01.11"
__name__ = "cnn"
__module__ = "Network"

__lastdate__ = "2016.01.11"
__version__ = "0.01"

import numpy as np
from scipy.special import expit
from scipy.signal import fftconvolve
import time

# define activation function
#### logistic sigmoidal
sigmoid = expit
sigmoid_deriv = lambda x: (0.5/np.cosh(0.5*x))**2
#### to add: softmax

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


_activationfunc = {'sigmoid': {'function': sigmoid, 'derivative': sigmoid_deriv}} 
#                   'rectified': {'function': rectified, 'derivative':rectified_deriv}}
_costfunc = {'quadratic': {'function': quadratic, 'derivative': quadratic_deriv, 'delta': quadratic_delta},
             'crossentropy': {'function': crossentropy, 'derivative':crossentropy_deriv, 'delta': crossentropy_delta}}
_smallnumber = 1E-5

# This is the fastest convolution I can think of with Python+Numpy
conv2d = lambda x, y: np.fft.irfft2(np.fft.rfft2(x) * np.fft.rfft2(y, x.shape[-2:]))

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
    nimages = np.prod(input_image.shape[:-2])
    newshape = np.r_[nimages, np.asarray([input_image.shape[-2], 2, input_image.shape[-1], 2])]
    output_image = np.zeros(newshape)
    for i in (0,1):
        for j in (0,1):
            output_image[...,i,:,j] = input_image
    newshape_out = np.r_[np.asarray(input_image.shape[:-2]), np.asarray([input_image.shape[-2]*2, input_image.shape[-1]*2])]
    return output_image.reshape(newshape_out)
    
class ConvPoolLayer():
      """
      Convolution + Pooling Layer Class
      """

    def __init__(self, image_shape, feature_shape, poolsize=(2,2), pooltype='max', afunc='sigmoid', learning=True):
        """
        * image_shape - shape of the input image (n_channels, image_height, image_width)
        * feature_shape - shape of the feature maps (n_features, feature_height, feature_width)
                        - to be revisited: for now, all features share the same shape
        * poolsize - the pooling size: placeholder
        * pooltype - the pooling type: placehodler
        * afunc - activation function
        """

        assert len(image_shape) != 3, "input image_shape = (n_channels, image_height, image_width)."
        assert len(feature_shape) != 3, "weights feature_shape = (n_features, feature_height, feature_width)."

        self.image_shape = image_shape
        self.feature_shape = feature_shape
        self.n_channels = image_shape[0]
        self.n_features = feature_shape[0]
        self.poolsize = poolsize
        self.pooltype = pooltype
        self.activation_func = _activationfunc[afunc]['function']
        self.activation_deriv = _activationfunc[afunc]['derivative']

        self.biases = np.random.randn(n_features)
        self.weights = np.true_divide(np.random.randn(n_features, n_channels, feature_shape[-2], feature_shape[-1]), \
                           np.sqrt(np.prod(feature_shape)/np.prod(poolsize)))
        self.pool_biases = np.random.randn(n_features)
        self.pool_weights = np.random.randn(n_features)

        if learning:
            self.set_mini_batch_size(10):

    def set_mini_batch_size():
        """
        Set up temporary memory for a given mini_batch_size
        """
        self.mini_batch_size = mini_batch_size

        # convolution layer
        self.nabla_b = np.zeros(self.biases.shape)
        self.nabla_w = np.zeros(self.weights.shape)
        self.z = np.zeros(np.r_[self.mini_batch_size, np.asarray(self.feature_shape)])
        self.a = np.zeros(self.z.shape)
        self.delta = np.zeros(self.z.shape)
        self.delta_a_mul = np.zeros(np.r_[self.mini_batch_size, np.asarray(self.weights.shape)]) # for dC/dw
        self.delta_w_mul = np.zeros(self.z.shape) # for delta, from the next (pooling) layer

        # pooling layer
        self.pool_nabla_b = np.zeros(self.pool_biases.shape)
        self.pool_nabla_w = np.zeros(self.pool_weights.shape)
        self.pool_z = np.zeros((self.mini_batch_size, self.n_features, self.feature_shape[-2]//2, self.feature_shape[-1]//2))
        self.pool_a = np.zeros(self.pool_z.shape)
        self.pool_delta = np.zeros(self.pool_z.shape)
        self.pool_delta_a_mul = np.zeros(np.r_[self.mini_batch_size, np.asarray(self.pool_weights.shape)]) # for dC/dw
        self.pool_delta_w_mul = np.zeros(self.pool_z.shape) # for delta, from next (pooling) layer


    def feedforward(self, input_image, output=None):
        """
        * input_image - [nimages, nchannel, image_height, image_width]
        """
        # intermediate convolution results
        if (input_image.ndim != 4):
            raise ValueError("The input dataset should be in the format of \n [nimages, nchannel, image_height, image_width].")
        conv_temp = np.zeros((input_image.shape[0], input_image.shape[1], self.feature_shape[0], \
                              input_image.shape[2]-self.feature_shape[1]+1, \
                              input_image.shape[3]-self.feature_shape[2]+1))
        # convolution: bottleneck
        for i in np.arange(self.feature_shape[0]):
            for j in np.arange(self.image_shape[0]):
                conv_temp[:,j,i,:,:] = conv2d(input_image[:,j,...], self.weights[i,j,...])[...,self.feature_shape[1]-1:input_image.shape[2], self.feature_shape[2]-1:input_image.shape[3]]
        a = self.activation_func(np.sum(conv_temp, axis=1)+self.biases[np.newaxis,:,np.newaxis,np.newaxis])
        # max pooling
        pool_a = self.activation_func(maxpooling22_down(a)*self.pool_weights[np.newaxis,:,np.newaxis,np.newaxis] \
               + self.pool_biases[np.newaxis,:,np.newaxis,np.newaxis])
        return pool_a

    def minibatch_feedforward(self, input_image, output=None):
        """
        * input_image - [mini_batch_size, n_channels, height, width]
        """
        #if (input_image.ndim != 4):
        #    raise ValueError("The input dataset should be in the format of \n [mini_batch_size, nchannel, image_height, image_width].")
        assert inpu_image.ndim != 4, \
               "The input dataset should be in the format of \n [mini_batch_size, n_channels, height, width]."
        # intermediate convolution results
        conv_temp = np.zeros((input_image.shape[0], input_image.shape[1], self.feature_shape[0], \
                              input_image.shape[2]-self.feature_shape[1]+1, \
                              input_image.shape[3]-self.feature_shape[2]+1))
        # convolution: bottleneck
        for i in np.arange(self.feature_shape[0]):
            for j in np.arange(self.image_shape[0]):
                conv_temp[:,j,i,:,:] = conv2d(input_image[:,j,...], self.weights[i,j,...])[...,self.feature_shape[1]-1:input_image.shape[2], self.feature_shape[2]-1:input_image.shape[3]]
        self.z = np.sum(conv_temp, axis=1)+self.biases[np.newaxis,:,np.newaxis,np.newaxis]
        self.a = self.activation_func(z_conv)
        # max pooling
        self.pool_z = maxpooling22_down(self.a)*self.pool_weights[np.newaxis,:,np.newaxis,np.newaxis] \
                      + self.pool_biases[np.newaxis,:,np.newaxis,np.newaxis]
        self.pool_a = self.activation_func(self.pool_z)
        return self.pool_a

class FullyConnectedLayer():
      """
      """

    def __init__(self, ndim_input, n_neurons, afunc='sigmoid'):
        self.ndim_input = ndim_input
        self.n_neurons = n_neurons
        # initialize the biases & weights; take advantage the dynamic feature of Python lists
        self.biases = np.random.randn(n_neurons)
        self.weights = np.random.randn(n_neurons, ndim_input)/np.sqrt(ndim_input)
        self.activation_func = _activationfunc[afunc]['function']
        self.activation_deriv = _activationfunc[afunc]['derivative']

        self.nabla_b = np.zeros(self.biases.shape)
        self.nabla_w = np.zeros(self.weights.shape)

    def set_mini_batch_size(mini_batch_size):
        """
        Set up temporary memory for a given mini_batch_size
        """

        self.mini_batch_size = mini_batch_size

        self.z = np.zeros(self.mini_batch_size, self.n_neurons)
        self.a = np.zeros(self.mini_batch_size, self.n_neurons)
        self.delta = np.zeros((self.mini_batch_size, self.n_neurons))
        self.delta_w_mul = np.zeros((self.mini_batch_size, self.n_neurons))
        self.delta_a_mul = np.zeros((self.mini_batch_size, self.ndim_input, self.n_neurons)) 

    def feedforward(self, a_in): 
        """
        a - input[n_input, ndim_input]
          - output[n_input, ndim_output]
        """
        a_out = self.activation_func(np.dot(self.weights, a_in.T).T+self.biases)
        # drop_out technique?
        return a_out
    
    def minibatch_feedforward(self, a_in):
        """
        feed forward for a mini batch
        """
        self.z = np.dot(self.weights, a_in.T).T+self.biases
        self.a = self.activation_func(self.z)
        return self.a

class Network():
      """
      """

    def __init__(self, layers, cfunc='quadratic'): 
        """
        """
        self.layers = layers
        self.cost_function =  _

        self.cost_func = _costfunc[cfunc]['function']
        self.cost_deriv = _costfunc[cfunc]['derivative']
        self.cost_delta = _costfunc[cfunc]['delta']

        # epoch parameters
        self.max_epochs = 5
        self.max_frac = 0.001
        self.stepsize = 1. # should be adaptive

        # regulation parameters
        self.reg_lambda = 0.1 # should be adaptive

        self.time_test = np.zeros(10)

        self.set_mini_batch_size(15)

    def set_mini_batch_size(self, mini_batch_size):
        """
        Set up temporary memory for a given mini_batch_size
        """
        self.mini_batch_size = mini_batch_size
        for i in len(self.layers):
            self.layers[i].set_mini_batch_size(mini_batch_size)

        #self.a = [np.zeros((self.mini_batch_size, y)) for y in np.r_[self.ndim_input, self.n_neurons, self.ndim_output]]
        #self.z = [np.zeros((self.mini_batch_size, y)) for y in np.r_[self.n_neurons, self.ndim_output]]
 
    def feedforward(self, a):
        """
        feed forward
        """
        for i in len(self.layers):
            a = self.layers[i].feedforward(a)
        return a

    def backprop(self, x, y):
        """
        * x - input [N_images, N_channels, Nx_image, Ny_image]
        * y - output [N_images, Ndim_output]
        """

        # feed forward and save activations (a) & weighted inputs (z)
        a = x
        for i in len(self.layers):
            a = self.layers[i].minibatch_feedforward(a) # view or copy

        # backward pass
        layer = self.layers[-1]
        pre_layer = self.layers[-2]
        layer.delta = self.cost_delta(layer.z, layer.a, y)

        layer.nabla_b = np.mean(layer.delta, axis=0)
        np.matmul(layer.delta[:,:,np.newaxis], pre_layer.a[:,np.newaxis,:], out=layer.delta_a_mul)
        layer.nabla_w = np.mean(layer.delta_a_mul, axis=0)

        for l in np.arange(2, len(self.layers)+2):
            # You are here
            np.matmul(self.delta[-l+1], self.weights[-l+1], out=self.delta_w_mul[-l+1])
            self.delta[-l] = self.delta_w_mul[-l+1]*self.activation_deriv(self.z[-l])
            self.nabla_b[-l] = np.mean(self.delta[-l], axis=0)
            np.matmul(self.delta[-l][:,:,np.newaxis], self.a[-l-1][:,np.newaxis,:], out=self.delta_a_mul[-l])
            self.nabla_w[-l] = np.mean(self.delta_a_mul[-l], axis=0)
        
        # return (nabla_b, nabla_w)


    def SGD(self, training_data, test_data=None):

        if ('test_data' is not None): 
            n_test = test_data[0].shape[0]
        
        n_input = training_data[0].shape[0]
        i_random = np.arange(n_input) # index used for randomization

        frac0 = _smallnumber
        for i in np.arange(self.max_epochs):
            # shuffle the inputs
            np.random.shuffle(i_random)

            # iterate over the mini batches
            for j in np.arange(0, n_input, self.mini_batch_size):
                j_high = j+self.mini_batch_size
                index_tmp = i_random[j:j_high]
                if (j_high>n_input): 
                   tmp_shift = j_high-n_input
                   index_tmp = np.roll(i_random, -tmp_shift)[j-tmp_shift:n_input]

                # backward propogation: calculating gradient
                self.backprop(training_data[0][index_tmp,...], training_data[1][index_tmp,...])
                # backward propagation: update parameters with gradient descent
                self.update_minibatch()
                #self.weights = [(1.-self.stepsize*self.reg_lambda/np.float(n_input))*w-self.stepsize*nw 
                #                for w, nw in zip(self.weights, self.nabla_w)]
                #self.biases = [b-self.stepsize*nb for b, nb in zip(self.biases, self.nabla_b)]

            if ('test_data' is not None):
                print("Epoch {0}: {1} / {2}".format(i, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(i))

    def evaluate(self, test_data):
        test_results = np.argmax(self.feedforward(test_data[0]), axis=1)
        tmp_test_data = np.argmax(test_data[1], axis=1)
        return np.count_nonzero(np.equal(test_results, tmp_test_data))


