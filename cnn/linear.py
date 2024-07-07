import numpy as np


class Linear:

    def __init__(self, in_features, out_features, weight_init_fn=None, bias_init_fn=None, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        self.W = np.zeros((out_features, in_features))  # TODO
        self.b = np.zeros((out_features, 1))  # TODO

        if weight_init_fn is None:
            self.W = np.zeros((out_features, in_features))
        else:
            self.W = weight_init_fn(out_features, in_features)

        if bias_init_fn is None:
            self.b = np.zeros((out_features, 1))
        else:
            self.b = bias_init_fn(out_features)

        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A  # TODO
        self.N = np.size(A, 0)  # TODO store the batch size of input
        # Think how will self.Ones helps in the calculations and uncomment below
        self.Ones = np.ones((self.N, 1))
        Z = A @ np.transpose(self.W) + self.Ones @ np.transpose(self.b)  # TODO

        return Z

    def backward(self, dLdZ):

        dLdA = dLdZ @ self.W  # TODO
        self.dLdW = np.transpose(dLdZ) @ self.A  # TODO
        self.dLdb = np.transpose(dLdZ) @ self.Ones  # TODO

        if self.debug:
            
            self.dLdA = dLdA

        return dLdA