import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A

        batch_size, channels, input_width, input_height = A.shape # should in_channels = out_channels ? Maybe it should.
        output_width = int((input_width - self.kernel) + 1)
        output_height = int((input_height - self.kernel) + 1)

        Z = np.zeros((batch_size, channels, output_width, output_height))
        self.maxIndex = np.zeros((batch_size, channels, output_width, output_height, 2))

        for i in range(batch_size):
            for j in range(channels):
                for k in range(output_width):
                    for l in range(output_height):
                        segment = A[i, j][k:k+self.kernel, l:l+self.kernel]
                        Z[i, j, k, l] = np.max(segment)
                        index = np.array(np.where(segment == Z[i, j, k, l]))
                        self.maxIndex[i, j, k, l, 0] = int(index[0, 0]) + k # no need to + 1
                        self.maxIndex[i, j, k, l, 1] = int(index[1, 0]) + l # no need to + 1

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        dLdA = np.zeros_like(self.A)

        for i in range(dLdZ.shape[0]):
            for j in range(dLdZ.shape[1]):
                for k in range(dLdZ.shape[2]):
                    for l in range(dLdZ.shape[3]):
                        dLdA[i, j, int(self.maxIndex[i, j, k, l, 0]), int(self.maxIndex[i, j, k, l, 1])] += dLdZ[i, j, k, l]

        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A

        batch_size, channels, input_width, input_height = A.shape # should in_channels = out_channels ? Maybe it should.
        output_width = int((input_width - self.kernel) + 1)
        output_height = int((input_height - self.kernel) + 1)

        Z = np.zeros((batch_size, channels, output_width, output_height))
        self.Index = np.zeros((batch_size, channels, output_width, output_height, self.kernel, self.kernel, 2))

        for i in range(batch_size):
            for j in range(channels):
                for k in range(output_width):
                    for l in range(output_height):
                        Z[i, j, k, l] = np.mean(A[i, j][k:k+self.kernel, l:l+self.kernel])
                        for m in range(self.kernel):
                            for n in range(self.kernel):
                                self.Index[i, j, k, l, m, n, 0] = m + k # no need to + 1
                                self.Index[i, j, k, l, m, n, 1] = n + l # no need to + 1
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        batch_size, channels, output_width, output_height = dLdZ.shape
        dLdA = np.zeros_like(self.A)

        for i in range(batch_size):
            for j in range(channels):
                for k in range(output_width):
                    for l in range(output_height):
                        for m in range(self.kernel):
                            for n in range(self.kernel):
                                dLdA[i, j, int(self.Index[i, j, k, l, m, n, 0]), int(self.Index[i, j, k, l, m, n, 1])] += dLdZ[i, j, k, l] / (self.kernel * self.kernel)

        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        return self.downsample2d.forward(self.maxpool2d_stride1.forward(A))

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        return self.maxpool2d_stride1.backward(self.downsample2d.backward(dLdZ))


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        return self.downsample2d.forward(self.meanpool2d_stride1.forward(A))

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        return self.meanpool2d_stride1.backward(self.downsample2d.backward(dLdZ))
