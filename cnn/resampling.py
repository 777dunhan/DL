import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        output_width = int((A.shape[2] - 1) * self.upsampling_factor + 1)
        Z = np.zeros((A.shape[0], A.shape[1], output_width))  # TODO
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                for k in range(A.shape[2]):
                    Z[i, j, self.upsampling_factor*k] = A[i, j, k]
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        input_width = int(1 + (dLdZ.shape[2] - 1) / self.upsampling_factor)
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], input_width))  # TODO
        for i in range(dLdZ.shape[0]):
            for j in range(dLdZ.shape[1]):
                for k in range(input_width):
                    dLdA[i, j, k] = dLdZ[i, j, self.upsampling_factor*k]
        
        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor
        self.input_size = 0

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        self.input_size = A.shape[2]
        output_width = int(1 + (A.shape[2] - 1) / self.downsampling_factor)
        Z = np.zeros((A.shape[0], A.shape[1], output_width))  # TODO
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                for k in range(output_width):
                    Z[i, j, k] = A[i, j, self.downsampling_factor*k]
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        input_width = self.input_size
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], input_width))  # TODO
        for i in range(dLdZ.shape[0]):
            for j in range(dLdZ.shape[1]):
                for k in range(dLdZ.shape[2]):
                    dLdA[i, j, self.downsampling_factor*k] = dLdZ[i, j, k]
        
        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor
        self.input_height = 0
        self.input_width = 0

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        self.input_height = A.shape[2]
        self.input_width = A.shape[3]

        output_height = int((A.shape[2] - 1) * self.upsampling_factor + 1)
        output_width = int((A.shape[3] - 1) * self.upsampling_factor + 1)
        Z = np.zeros((A.shape[0], A.shape[1], output_height, output_width))  # TODO
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                for m in range(A.shape[2]):
                    for n in range(A.shape[3]):
                        Z[i, j, m*self.upsampling_factor, n*self.upsampling_factor] = A[i, j, m, n]
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        input_height = self.input_height
        input_width = self.input_width
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], input_height, input_width))  # TODO
        for i in range(dLdZ.shape[0]):
            for j in range(dLdZ.shape[1]):
                for m in range(input_height):
                    for n in range(input_width):
                        dLdA[i, j, m, n] = dLdZ[i, j, m*self.upsampling_factor, n*self.upsampling_factor]
        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor
        self.input_height = 0
        self.input_width = 0

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        self.input_height = A.shape[2]
        self.input_width = A.shape[3]

        output_height = int(1 + (A.shape[2] - 1) / self.downsampling_factor)
        output_width = int(1 + (A.shape[3] - 1) / self.downsampling_factor)
        Z = np.zeros((A.shape[0], A.shape[1], output_height, output_width))  # TODO
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                for m in range(output_height):
                    for n in range(output_width):
                        Z[i, j, m, n] = A[i, j, m*self.downsampling_factor, n*self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], self.input_height, self.input_width))  # TODO
        for i in range(dLdZ.shape[0]):
            for j in range(dLdZ.shape[1]):
                for m in range(dLdZ.shape[2]):
                    for n in range(dLdZ.shape[3]):
                        dLdA[i, j, m*self.downsampling_factor, n*self.downsampling_factor] = dLdZ[i, j, m, n]

        return dLdA
