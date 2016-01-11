__license__ = "MIT"
__author__ = "Guangtun Ben Zhu (BGT) @ Johns Hopkins University"
__startdate__ = "2015.12.10"
__name__ = "ann"
__module__ = "Network"

__lastdate__ = "2016.01.11"
__version__ = "0.10"

import numpy as np
from scipy.special import expit
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
# define the core class
class Network:
    """Class for Neural Network

    Attributes: 
       * nlayers: the number of layers (between input and output) 
       * nneurons: the number of neurons for all the layers (a 1-D numpy array)
       * biases: 
       * weights: 
  
    Methods: 

    Cost function:
       quadratic: Sum (sqauredsum (y_i - y_out))

    To-do:

    """

    def __init__(self, ndim_input, ndim_output, n_neurons, afunc='sigmoid', cfunc='quadratic'):
        """
        * ndim_input - dimension (shape) of the input
        * ndim_output - dimension (shape) of the output
        * n_layers - number of hidden layers
        * n_neurons - number of neurons in the hidden layers
        * biases - biases (Python list of 1D arrays, hidden layers + output)
        * weights - biases (Python list of 2D arrays, input->hidden layers->output)
        """
        self.ndim_input = ndim_input
        self.ndim_output = ndim_output
        self.n_neurons = np.copy(n_neurons)
        self.n_layers = self.n_neurons.size
        # initialize the biases & weights; take advantage the dynamic feature of Python lists
        self.biases = [np.random.randn(y) for y in np.r_[n_neurons, ndim_output]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(np.r_[ndim_input, n_neurons], np.r_[n_neurons, ndim_output])]
        self.activation_func = _activationfunc[afunc]['function']
        self.activation_deriv = _activationfunc[afunc]['derivative']
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


        self.nabla_b = [np.zeros(b.shape) for b in self.biases]
        self.nabla_w = [np.zeros(w.shape) for w in self.weights]
        self.set_mini_batch_size(15)

    def set_mini_batch_size(self, mini_batch_size):
        self.mini_batch_size = mini_batch_size
        # allocate memory for temporary data to speed up the code
        self.a = [np.zeros((self.mini_batch_size, y)) for y in np.r_[self.ndim_input, self.n_neurons, self.ndim_output]]
        self.z = [np.zeros((self.mini_batch_size, y)) for y in np.r_[self.n_neurons, self.ndim_output]]
        self.delta = [np.zeros((self.mini_batch_size, y)) for y in np.r_[self.n_neurons, self.ndim_output]]
        self.delta_w_mul = [np.zeros((self.mini_batch_size, y)) for y in np.r_[self.ndim_input, self.n_neurons]]
        self.delta_a_mul = [np.zeros((self.mini_batch_size, y, x)) 
            for x, y in zip(np.r_[self.ndim_input, self.n_neurons], np.r_[self.n_neurons, self.ndim_output])]
        
    def feedforward(self, a):
        """
        a - input[n_input, ndim_input]
          - output[n_input, ndim_output]
        """
        for b, w in zip(self.biases, self.weights):
            a = self.activation_func(np.dot(w, a.T).T+b) # np.dot is 2x faster than np.einsum
        return a

    def backprop(self, x, y):
        """
        * x - input [Ninput, Ndim_input]
        * y - output [Ninput, Ndim_output]
        """

        self.a[0] = x
        l = 0
        for b, w in zip(self.biases, self.weights):
            self.z[l] = np.dot(w, self.a[l].T).T + b # z, b's shape - [Ninput, Nneurons (current layer)], a from previous layer
            self.a[l+1] = self.activation_func(self.z[l]) # new a's shape - [Ninput, Nneurons (current layer)]
            l += 1
        
        # backward pass
        self.delta[-1] = self.cost_delta(self.z[-1], self.a[-1], y)

        self.nabla_b[-1] = np.mean(self.delta[-1], axis=0)
        np.matmul(self.delta[-1][:,:,np.newaxis], self.a[-2][:,np.newaxis,:], out=self.delta_a_mul[-1])
        self.nabla_w[-1] = np.mean(self.delta_a_mul[-1], axis=0)

        for l in np.arange(2, self.n_layers+2):
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

                self.backprop(training_data[0][index_tmp,...], training_data[1][index_tmp,...])
                self.weights = [(1.-self.stepsize*self.reg_lambda/np.float(n_input))*w-self.stepsize*nw 
                                for w, nw in zip(self.weights, self.nabla_w)]
                self.biases = [b-self.stepsize*nb for b, nb in zip(self.biases, self.nabla_b)]

            # calculate the cost
            # frac = self.evaluate(training_data)
            # if (np.fabs(frac-frac0)<self.max_fracchange):
            # a = self.feedforward(training_data[0])
            # cost = self.cost_func(a, training_data[1])
            # if (np.fabs(cost-cost0)

            if ('test_data' is not None):
                print("Epoch {0}: {1} / {2}".format(i, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(i))

    def evaluate(self, test_data):
        test_results = np.argmax(self.feedforward(test_data[0]), axis=1)
        tmp_test_data = np.argmax(test_data[1], axis=1)
        return np.count_nonzero(np.equal(test_results, tmp_test_data))


