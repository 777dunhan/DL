import numpy as np

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """
        self.A = A
        return np.reshape(A, (A.shape[0], A.shape[1] * A.shape[2]))

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """
        return np.reshape(dLdZ, (self.A.shape[0], self.A.shape[1], self.A.shape[2]))
