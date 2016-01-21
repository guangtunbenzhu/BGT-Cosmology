__license__ = "MIT"
__author__ = "Guangtun Ben Zhu (BGT) @ Johns Hopkins University"
__startdate__ = "2016.01.11"
__name__ = "cnn"
__module__ = "Network"

__lastdate__ = "2016.01.19"
__version__ = "0.01"

# To-do:
#   Check if stepsize is the problem
import numpy as np
import time

from nnmath import *

# some small number epsilon
_EPS = 1E-5

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

        assert len(image_shape) == 3, "input image_shape = (n_channels, image_height, image_width)."
        assert len(feature_shape) == 3, "weights feature_shape = (n_features, feature_height, feature_width)."

        self.image_shape = image_shape
        self.feature_shape = feature_shape
        self.conv_shape = (feature_shape[0], image_shape[1]-feature_shape[1]+1, image_shape[2]-feature_shape[2]+1)
        # The following needs to be fixed
        assert self.conv_shape[1] % 2 == 0 and self.conv_shape[2] % 2 == 0, \
            "The convolved images must be of even dimensions for maximum pooling."
        self.pool_shape = (feature_shape[0], self.conv_shape[1]//2, self.conv_shape[2]//2) # what if conv_shape[1]/[2] is odd?
        self.n_channels = image_shape[0]
        self.n_features = feature_shape[0]
        self.poolsize = poolsize
        self.pooltype = pooltype
        self.activation_func = activationfunc[afunc]['function']
        self.activation_deriv = activationfunc[afunc]['derivative']

        self.biases = np.random.randn(self.n_features)
        self.weights = np.true_divide(np.random.randn(self.n_features, self.n_channels, feature_shape[-2], feature_shape[-1]), \
                           np.sqrt(np.prod(image_shape))) # Is n_channels dimension necessary?
        self.pool_biases = np.random.randn(self.n_features)
        self.pool_weights = np.random.randn(self.n_features)/np.sqrt(np.prod(feature_shape))

        if learning:
            self.set_mini_batch_size(10)
            self.set_parameters()

    def set_mini_batch_size(self, mini_batch_size):
        """
        Set up temporary memory for a given mini_batch_size
        """
        self.mini_batch_size = mini_batch_size

        # convolution layer
        self.nabla_b = np.zeros(self.biases.shape)
        self.nabla_w = np.zeros(self.weights.shape)
        self.conv_z = np.zeros(np.r_[self.mini_batch_size, np.asarray(self.conv_shape)])
        self.conv_a = np.zeros(self.conv_z.shape)
        self.delta = np.zeros(self.conv_z.shape)
        self.delta_a_mul = np.zeros(np.r_[self.mini_batch_size, np.asarray(self.weights.shape)]) # for dC/dw
        self.all_conv_delta_w_mul = np.zeros((self.mini_batch_size, self.n_channels, self.n_features, \
                              self.image_shape[1], self.image_shape[2]))
        self.conv_delta_w_mul = np.zeros(self.image_shape) # for previous layer

        # pooling layer
        self.pool_nabla_b = np.zeros(self.pool_biases.shape)
        self.pool_nabla_w = np.zeros(self.pool_weights.shape)
        self.z = np.zeros(np.r_[self.mini_batch_size, np.asarray(self.pool_shape)])
        self.a = np.zeros(self.z.shape)
        self.pool_delta = np.zeros(self.z.shape)
        self.pool_delta_a_mul = np.zeros(np.r_[self.mini_batch_size, np.asarray(self.pool_weights.shape)]) # for dC/dw
        self.delta_w_mul = np.zeros(self.z.shape) # for delta in the previous (convolution) layer

        self.down_conv_a = np.zeros(self.z.shape)

    def set_parameters(self, stepsize=1.0, reg_lambda=1E-5):
        """
        Set the SGD parameters
        """
        self.stepsize = stepsize
        self.reg_lambda = reg_lambda

    def feedforward(self, input_image, minibatch=False):
        """
        * input_image - [nimages or mini_batch_size, nchannel, image_height, image_width]
        """
        # intermediate convolution results
        #if (input_image.ndim != 4):
        #    raise ValueError("The input dataset should be in the format of \n [nimages, nchannel, image_height, image_width].")
        assert input_image.ndim == 4, \
               "The input dataset should be in the format of \n [nimages, nchannel, image_height, image_width]."
        assert input_image.shape[1] == self.n_channels, \
               "The input image's n_channels does not match the weights."

        conv_temp = np.zeros((input_image.shape[0], input_image.shape[1], self.n_features, \
                              input_image.shape[2]-self.feature_shape[1]+1, \
                              input_image.shape[3]-self.feature_shape[2]+1))
        # convolution layer: bottleneck
        for i in np.arange(self.n_features):
            for j in np.arange(self.n_channels):
                # print(conv_temp[:,j,i,:,:].shape, input_image[:,j,:,:].shape, self.weights[i,j,:,:].shape)
                conv_temp[:,j,i,:,:] = conv2d(input_image[:,j,:,:], self.weights[i,j,:,:], mode='valid')

        conv_z = np.sum(conv_temp, axis=1)+self.biases[np.newaxis,:,np.newaxis,np.newaxis]       
        conv_a = self.activation_func(conv_z)

        # pooling layer, max(2,2)
        down_conv_a = maxpooling22_down(conv_a)
        z = down_conv_a*self.pool_weights[np.newaxis,:,np.newaxis,np.newaxis] \
                              + self.pool_biases[np.newaxis,:,np.newaxis,np.newaxis]
        a = self.activation_func(z)
        # print(conv_temp.shape, z.shape, a.shape, pool_z.shape, pool_a.shape)

        if minibatch: # revisit to check view or copy
            self.conv_z = conv_z
            self.conv_a = conv_a
            self.down_conv_a = down_conv_a
            self.z = z
            self.a = a
        return a

    def backprop(self, prevlayer=None, nextlayer=None, \
            finallayer=False, delta=None, \
            firstlayer=False, rawinput=None):
        """
        This should be the final step
        """
        if finallayer:
            assert delta is not None, \
                "If it is the final layer, please provide delta."
            self.pool_delta = delta

        else:
            assert nextlayer is not None, \
                "If it is not the final layer, please provide the next layer."
            self.pool_delta = nextlayer.delta_w_mul.reshape(self.z.shape)*self.activation_deriv(self.z)

        if firstlayer:
            assert rawinput is not None, \
                "If it is the first layer, please provide rawinput."
        else:
            assert prevlayer is not None, \
                "If it is not the first layer, please provide the previous layer."

        # pooling layer 
        self.pool_nabla_b = np.mean(np.einsum('ijkl->ij', self.pool_delta), axis=0)
        up_pool_delta = maxpooling22_up(self.pool_delta)
        # print(up_pool_delta.shape, self.pool_delta_a_mul.shape, self.conv_a.shape)
        self.pool_delta_a_mul = np.einsum('ijkl,ijkl->ij', self.down_conv_a, self.pool_delta) # This is right
        #self.pool_delta_a_mul = np.einsum('ijkl,ijkl->ij', self.conv_a/4., up_pool_delta) # This is wrong
        self.pool_nabla_w = np.mean(self.pool_delta_a_mul, axis=0)
        # Try 2 things:
        # 1: save the maximum position, other positions have delta == 0
        # 2: Use mean pooling
        self.delta_w_mul = up_pool_delta*self.pool_weights[np.newaxis,:,np.newaxis,np.newaxis]*0.5 # This is likely wrong

        # convolutional layer
        self.delta = self.delta_w_mul*self.activation_deriv(self.conv_z) # [mini_batch_size, n_features, conva_height, conva_width]
        self.nabla_b = np.mean(np.einsum('ijkl->ij', self.delta), axis=0)
        if firstlayer:
            tmp_input = rawinput
        else:
            tmp_input = prevlayer.a

        for i in np.arange(self.n_features):
            for j in np.arange(self.mini_batch_size):
                self.delta_a_mul[j,i,:,:,:] = conv2d(tmp_input[j,:,:,:], self.delta[j,i,:,:], mode='valid')
                # [mini_batch_size, n_features, n_channels, feature_height, feature_width]

        self.nabla_w = np.mean(self.delta_a_mul, axis=0)
        #if firstlayer:
            # print("ConvPool: ", np.mean(self.delta), np.mean(self.pool_delta))
        #    print("ConvPool: ", np.mean(self.nabla_b), np.mean(self.nabla_w), np.mean(self.nabla_b), np.mean(self.pool_nabla_w))
        # this is not necessary if it is the first layer
        if (not firstlayer):
            for i in np.arange(self.mini_batch_size):
                for j in np.arange(self.n_channels):
                    for k in np.arange(self.n_features):
                        self.all_delta_w_mul[i,j,k,...] = \
                            conv2d(self.delta[i,k,...], self.weights[k,j,::-1, ::-1], mode='full')
                        # all_delta_w_mul [mini_batch_size, n_channels, n_features, image_height, image_width]
                        # delta [mini_batch_size, n_features, conva_height, conva_width]
                        # weights [n_features, n_channels, feature_height, feature_width]
            self.delta_w_mul = np.sum(self.all_delta_w_mul, axis=2)
            # delta_w_mul [mini_batch_size, n_channels, image_height, image_width]

    def update_minibatch(self):
        """
        Update mini batch, if the convolution layer and pooling layer are separated, 
        this can be incorporated into the routine SGD as a single piece of code.
        """

        #updates pooling weights
        self.pool_weights = (1.-self.stepsize*self.reg_lambda)*self.pool_weights-self.stepsize*self.pool_nabla_w
        self.pool_biases = self.pool_biases-self.stepsize*self.pool_nabla_b

        #updates weights
        self.weights = (1.-self.stepsize*self.reg_lambda)*self.weights-self.stepsize*self.nabla_w
        self.biases = self.biases-self.stepsize*self.nabla_b

class FullyConnectedLayer():
    """
    """

    def __init__(self, ndim_input, n_neurons, afunc='sigmoid', learning=True):
        self.ndim_input = ndim_input
        self.n_neurons = n_neurons
        # initialize the biases & weights; take advantage the dynamic feature of Python lists
        self.biases = np.random.randn(n_neurons)
        self.weights = np.random.randn(n_neurons, ndim_input)/np.sqrt(ndim_input)
        self.activation_func = activationfunc[afunc]['function']
        self.activation_deriv = activationfunc[afunc]['derivative']

        self.nabla_b = np.zeros(self.biases.shape)
        self.nabla_w = np.zeros(self.weights.shape)

        if learning:
            self.set_mini_batch_size(10)
            self.set_parameters()

    def set_mini_batch_size(self, mini_batch_size):
        """
        Set up temporary memory for a given mini_batch_size
        """

        self.mini_batch_size = mini_batch_size

        self.z = np.zeros((self.mini_batch_size, self.n_neurons))
        self.a = np.zeros((self.mini_batch_size, self.n_neurons))
        self.delta = np.zeros((self.mini_batch_size, self.n_neurons))
        self.delta_a_mul = np.zeros(np.r_[self.mini_batch_size, np.asarray(self.weights.shape)]) # for dC/dw
        self.delta_w_mul = np.zeros((self.mini_batch_size, self.ndim_input)) # for previous layer

    def set_parameters(self, stepsize=1.0, reg_lambda=1E-5):
        """
        Set the SGD parameters
        """
        self.stepsize = stepsize
        self.reg_lambda = reg_lambda

    def feedforward(self, a_in, minibatch=False): 
        """
        a - input[n_input, ndim_input]
          - output[n_input, ndim_output]
        """
        #print(a_in.shape, (a_in.shape[0], self.ndim_input))
        #print(a_in.reshape(a_in.shape[0], self.ndim_input).shape)
        z = np.dot(self.weights, a_in.reshape(a_in.shape[0], self.ndim_input).T).T+self.biases
        a = self.activation_func(z)

        if minibatch: # revisit to check view or copy
            self.z = z
            self.a = a

        # add drop_out technique?

        return a
    
    def backprop(self, prevlayer=None, nextlayer=None, \
            finallayer=False, delta=None, \
            firstlayer=False, rawinput=None):
        """
        """
        if finallayer:
            assert delta is not None, \
                "If it is the final layer, please provide delta."
            self.delta = delta
        else:
            assert nextlayer is not None, \
                "If it is not the final layer, please provide the information of the next layer."
            self.delta = nextlayer.delta_w_mul.reshape(self.z.shape)*self.activation_deriv(self.z)
        if firstlayer:
            assert rawinput is not None, \
                "If it is the first layer, please provide rawinput."
        else:
            assert prevlayer is not None, \
                "If it is not the first layer, please provide the previous layer."

        self.nabla_b = np.mean(self.delta, axis=0)
        if firstlayer:
            newshape = (rawinput.shape[0], 1, np.prod(rawinput.shape[1:]))
            tmp_input = rawinput.reshape(newshape)
        else:
            newshape = (prevlayer.a.shape[0], 1, np.prod(prevlayer.a.shape[1:]))
            tmp_input = prevlayer.a.reshape(newshape)
            # print(self.delta.shape, prevlayer.a.shape, newshape, self.delta_a_mul.shape)
        np.matmul(self.delta[:,:,np.newaxis], tmp_input, out=self.delta_a_mul)
        self.nabla_w = np.mean(self.delta_a_mul, axis=0)
        #if firstlayer: 
        #if finallayer: print("FullyConnected: ", np.mean(self.delta))

        if not firstlayer:
            np.matmul(self.delta, self.weights, out=self.delta_w_mul)

    def update_minibatch(self):
        """
        Update mini batch, if the convolution layer and pooling layer are separated, 
        this can be incorporated into the routine SGD as a single piece of code.
        """
        self.weights = (1.-self.stepsize*self.reg_lambda)*self.weights-self.stepsize*self.nabla_w
        self.biases = self.biases-self.stepsize*self.nabla_b


class Network():
    """
    """

    def __init__(self, layers, cfunc='quadratic', learning=True): 
        """
        """
        self.layers = layers

        # cost function
        self.cost_func = costfunc[cfunc]['function']
        self.cost_deriv = costfunc[cfunc]['derivative']
        self.cost_delta = costfunc[cfunc]['delta']

        # epoch parameters
        self.max_epochs = 5
        self.max_frac = 0.001

        # set learning parameters
        if learning:
            self.set_mini_batch_size(10)
            self.set_parameters(stepsize=1.0, reg_lambda=1E-5)

        # only in test version
        self.time_test = np.zeros(10)

    def set_mini_batch_size(self, mini_batch_size):
        """
        allocate temporary memory for a given mini_batch_size
        """
        self.mini_batch_size = mini_batch_size
        for layer in self.layers:
            layer.set_mini_batch_size(mini_batch_size)

    def set_parameters(self, stepsize=1.0, reg_lambda=1E-5):
        """
        set learning parameters
        """
        # stepsize in gradient descent
        self.stepsize = stepsize # should be adaptive
        # regulation parameters 
        self.reg_lambda = reg_lambda # should be adaptive
        for layer in self.layers:
            layer.set_parameters(stepsize=stepsize, reg_lambda=reg_lambda)

    def feedforward(self, a):
        """
        feed forward
        """
        for layer in self.layers:
            a = layer.feedforward(a, minibatch=False)
        return a

    def backprop(self, x, y):
        """
        * x - input [N_images, N_channels, image_height, image_width]
        * y - output [N_images, Ndim_output]
        """

        # feed forward and save activations (a) & weighted inputs (z)
        a = x
        for layer in self.layers:
            a = layer.feedforward(a, minibatch=True) # view or copy

        # backward propagation
        # Final layer
        delta = self.cost_delta(self.layers[-1].z, self.layers[-1].a, y)
        self.layers[-1].backprop(prevlayer=self.layers[-2], finallayer=True, delta=delta)
        # Middle layers
        for l in np.arange(2, len(self.layers)+1):
            if l != len(self.layers):
                self.layers[-l].backprop(prevlayer=self.layers[-l-1], nextlayer=self.layers[-l+1]) # calculate delta, nabla_b, nabla_w
            else:
                # First layer
                # print(np.mean(x))
                self.layers[-l].backprop(nextlayer=self.layers[-l+1], firstlayer=True, rawinput=x) # calculate delta, nabla_b, nabla_w


    def update_minibatch(self):
        """
        """

        for layer in self.layers:
            layer.update_minibatch()

    def SGD(self, training_data, test_data=None):

        if ('test_data' is not None): 
            n_test = test_data[0].shape[0]
        
        n_input = training_data[0].shape[0]
        i_random = np.arange(n_input) # index used for randomization

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
                # print(np.median(self.layers[0].nabla_w), np.median(self.layers[0].nabla_b))
                self.update_minibatch()

            print("Cost = ", self.cost_func(self.layers[-1].a, training_data[1][index_tmp,...]))
            if ('test_data' is not None):
                print("Epoch {0}: {1} / {2}".format(i, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(i))

    def evaluate(self, test_data):
        test_results = np.argmax(self.feedforward(test_data[0]), axis=1)
        tmp_test_data = np.argmax(test_data[1], axis=1)
        return np.count_nonzero(np.equal(test_results, tmp_test_data))

