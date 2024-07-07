# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        self.A = A
        output_size = A.shape[2] - self.kernel_size + 1
        Z = np.zeros((A.shape[0], self.out_channels, output_size))
        for i in range(A.shape[0]):
            for j in range(self.out_channels):
                for k in range(output_size):
                    Z[i, j, k] = np.tensordot(A[i][:, k:(k+self.kernel_size)], self.W[j], axes=([0,1], [0,1])) + self.b[j]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        dLdA = np.zeros_like(self.A)

        for i in range(dLdZ.shape[0]):
            for j in range(self.out_channels):
                for k in range(dLdZ.shape[2]):
                    dLdA[i, :, k : (k + self.kernel_size)] += self.W[j] * dLdZ[i, j, k]
                    self.dLdW[j] += self.A[i][:, k : (k + self.kernel_size)] * dLdZ[i, j, k]
                    self.dLdb[j] += dLdZ[i, j, k]
        
        return dLdA

class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv1d() and Downsample1d() instance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  # TODO
        self.downsample1d = Downsample1d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        
        # Pad the input appropriately using np.pad() function
        A = np.pad(A, ((0,0), (0,0), (self.pad, self.pad)))

        # Call Conv1d_stride1
        Z = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        # Call downsample1d backward
        dLdZ0 = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdZ0)  # TODO

        # Unpad the gradient
        if self.pad > 0:
            dLdA = dLdA[:, :, self.pad:-self.pad]

        return dLdA
