import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """

        self.A = A
        output_height = int((A.shape[2] - self.kernel_size) + 1)
        output_width = int((A.shape[3] - self.kernel_size) + 1)
        Z = np.zeros((A.shape[0], self.out_channels, output_height, output_width))

        for i in range(A.shape[0]):
            for j in range(self.out_channels):
                for k in range(output_width):
                    for l in range(output_height):
                        segment = A[i][:, l:l+self.kernel_size, k:k+self.kernel_size]
                        Z[i,j,l,k] = np.sum(segment * self.W[j]) + self.b[j]
        return Z


    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        dLdA = np.zeros_like(self.A)

        for i in range(dLdZ.shape[0]):
            for j in range(self.out_channels):
                W = self.W[j,:,:,:] # shape => (in_channel, kernel_size, kernel_size)
                for k in range(dLdZ.shape[3]):
                    for l in range(dLdZ.shape[2]):
                        dLdA[i,:,l : l+self.kernel_size, k : k+self.kernel_size] += W * dLdZ[i,j,l,k]
                        self.dLdW[j,:,:,:] += self.A[i][:, l : l+self.kernel_size, k : k+self.kernel_size] * dLdZ[i,j,l,k]
                        self.dLdb[j] += dLdZ[i,j,l,k]
        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn) # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """

        # Pad the input appropriately using np.pad() function
        A = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant', constant_values=0) # TODO

        # Call Conv2d_stride1
        Z0 = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(Z0)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        dLdZ0 = self.downsample2d.backward(dLdZ) # TODO

        # Call Conv2d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ0) # TODO

        # Unpad the gradient
        dLdA = dLdA[:, :, self.pad:-self.pad, self.pad:-self.pad] # TODO

        return dLdA
