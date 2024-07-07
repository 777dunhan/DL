import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = np.size(A, 0)  # TODO
        self.C = np.size(A, 1)  # TODO
        se = (A - Y) * (A - Y)  # TODO
        sse = np.ones((1, self.N)) @ se @ np.ones((self.C, 1))  # TODO
        mse = sse / (self.N * self.C)  # TODO

        return np.squeeze(mse)

    def backward(self):

        dLdA = (self.A - self.Y) * 2 / (self.N * self.C)  # TODO

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        self.N = np.size(A, 0)  # TODO

        exp = np.exp(A)
        self.softmax = exp / np.sum(exp, axis=1, keepdims=True) # TODO
        crossentropy = (-Y * np.log(self.softmax)) @ np.ones((np.size(A, 1), 1))  # TODO
        sum_crossentropy = np.ones((1, self.N)) @ crossentropy  # TODO
        L = sum_crossentropy / self.N

        return np.squeeze(L)

    def backward(self):

        dLdA = (self.softmax - self.Y) / self.N  # TODO

        return dLdA
