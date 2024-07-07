# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)​

from flatten import *
from Conv1d import *
from linear import *
from activation import *
from loss import *
import numpy as np
import os
import sys

sys.path.append('mytorch')


class CNN(object):

    """
    A simple convolutional neural network

    Here you build implement the same architecture described in Section 3.3
    You need to specify the detailed architecture in function "get_cnn_model" below
    The returned model architecture should be same as in Section 3.3 Figure 3
    """

    def __init__(self, input_width, num_input_channels, num_channels, kernel_sizes, strides,
                 num_linear_neurons, activations, conv_weight_init_fn, bias_init_fn,
                 linear_weight_init_fn, criterion, lr):
        """
        input_width           : int    : The width of the input to the first convolutional layer
        num_input_channels    : int    : Number of channels for the input layer
        num_channels          : [int]  : List containing number of (output) channels for each conv layer
        kernel_sizes          : [int]  : List containing kernel width for each conv layer
        strides               : [int]  : List containing stride size for each conv layer
        num_linear_neurons    : int    : Number of neurons in the linear layer
        activations           : [obj]  : List of objects corresponding to the activation fn for each conv layer
        conv_weight_init_fn   : fn     : Function to init each conv layers weights
        bias_init_fn          : fn     : Function to initialize each conv layers AND the linear layers bias to 0
        linear_weight_init_fn : fn     : Function to initialize the linear layers weights
        criterion             : obj    : Object to the criterion (SoftMaxCrossEntropy) to be used
        lr                    : float  : The learning rate for the class

        You can be sure that len(activations) == len(num_channels) == len(kernel_sizes) == len(strides)
        """

        # Don't change this -->

        self.train_mode = True
        self.nlayers = len(num_channels)

        self.activations = activations
        self.criterion = criterion

        self.lr = lr

        # <---------------------
        # Your code goes here -->
        # self.convolutional_layers (list Conv1d) = []
        # self.flatten              (Flatten)     = Flatten()
        # self.linear_layer         (Linear)      = Linear(???)
        # <---------------------

        # all convolutional layers
        self.convolutional_layers = [Conv1d(in_channels = int(([num_input_channels] + num_channels)[layer]), 
                                            out_channels = int(([num_input_channels] + num_channels)[layer+1]), 
                                            kernel_size = kernel_sizes[layer], 
                                            stride = strides[layer], 
                                            weight_init_fn = conv_weight_init_fn, 
                                            bias_init_fn = bias_init_fn) 
                                    for layer in range(self.nlayers)]
        
        # all activations & flatten layers
        self.tanh = Tanh()
        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        self.flatten = Flatten()

        # prepare to compute linear layer's input features
        for layer in range(self.nlayers):
            output_width = 1 + int((input_width - kernel_sizes[layer]) // strides[layer])
            input_width = output_width
        
        # linear layer
        self.linear_layer = Linear(in_features = output_width * int(([num_input_channels] + num_channels)[-1]), 
                                   out_features = num_linear_neurons, 
                                   weight_init_fn = linear_weight_init_fn, 
                                   bias_init_fn = bias_init_fn)
        

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, num_input_channels, input_width)
        Return:
            Z (np.array): (batch_size, num_linear_neurons)
        """

        for layer in [self.convolutional_layers[0],
                      self.tanh, 
                      self.convolutional_layers[1], 
                      self.relu, 
                      self.convolutional_layers[2], 
                      self.sigmoid, 
                      self.flatten, 
                      self.linear_layer]:
    
            A = layer.forward(A)
        
        self.Z = A

        return self.Z

    def backward(self, labels):
        """
        Argument:
            labels (np.array): (batch_size, num_linear_neurons)
        Return:
            grad (np.array): (batch size, num_input_channels, input_width)
        """
        # m, _ = labels.shape

        self.loss = self.criterion.forward(self.Z, labels).sum()
        grad = self.criterion.backward()

        for layer in [self.linear_layer,
                      self.flatten,
                      self.sigmoid,
                      self.convolutional_layers[2], 
                      self.relu,
                      self.convolutional_layers[1], 
                      self.tanh,
                      self.convolutional_layers[0]]:
            
            grad = layer.backward(grad)
        
        return grad

    def zero_grads(self):
        # Do not modify this method
        for i in range(self.nlayers):
            self.convolutional_layers[i].conv1d_stride1.dLdW.fill(0.0)
            self.convolutional_layers[i].conv1d_stride1.dLdb.fill(0.0)

        self.linear_layer.dLdW.fill(0.0)
        self.linear_layer.dLdb.fill(0.0)

    def step(self):
        # Do not modify this method
        for i in range(self.nlayers):
            self.convolutional_layers[i].conv1d_stride1.W = (self.convolutional_layers[i].conv1d_stride1.W -
                                                             self.lr * self.convolutional_layers[i].conv1d_stride1.dLdW)
            self.convolutional_layers[i].conv1d_stride1.b = (self.convolutional_layers[i].conv1d_stride1.b -
                                                             self.lr * self.convolutional_layers[i].conv1d_stride1.dLdb)

        self.linear_layer.W = (
            self.linear_layer.W -
            self.lr *
            self.linear_layer.dLdW)
        self.linear_layer.b = (
            self.linear_layer.b -
            self.lr *
            self.linear_layer.dLdb)

    def train(self):
        # Do not modify this method
        self.train_mode = True

    def eval(self):
        # Do not modify this method
        self.train_mode = False
