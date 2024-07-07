import numpy as np
import scipy

class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """
    def forward(self, Z):

        self.A = 1 / (1 + np.exp(-Z)) # TODO

        return self.A
    
    def backward(self, dLdA):

        dAdZ = self.A - self.A * self.A # TODO
        dLdZ = dLdA * dAdZ # TODO

        return dLdZ


class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    def forward(self, Z):

        self.A = np.tanh(Z)

        return self.A
    
    def backward(self, dLdA):

        dAdZ = 1 - np.square(self.A)
        dLdZ = dLdA * dAdZ

        return dLdZ


class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """
    def forward(self, Z):

        self.A = np.maximum(0, Z) # TODO

        return self.A
    
    def backward(self, dLdA):

        return np.where(self.A > 0, dLdA, 0) # TODO


class GELU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on GELU.
    """
    def forward(self, Z):

        self.A = Z # TODO

        return (1 / 2) * Z * (1 + scipy.special.erf(Z / np.sqrt(2)))
    
    def backward(self, dLdA):

        dAdZ = (1 / 2) * (1 + scipy.special.erf(self.A / np.sqrt(2))) + (self.A / np.sqrt(2 * np.pi)) * np.exp(-self.A * self.A / 2) # TODO
        dLdZ = dLdA * dAdZ # TODO

        return dLdZ


class Softmax:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Softmax.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """

        # Calculate the batch size and number of features
        # N = np.size(Z, 0) # TODO
        # C = np.size(Z, 1) # TODO

        # Initialize the final output A with all zeros. Refer to the writeup and think about the shape.
        # self.A = np.zeros((N, C)) # TODO
        
        # for i in range(N):
        #     exp = np.exp(Z[i,:])
        #     self.A[i,:] = exp / np.sum(exp)

        exp = np.exp(Z)
        self.A = exp / np.sum(exp, axis=1, keepdims=True) # TODO
        
        return self.A
    

    def backward(self, dLdA):

        # Calculate the batch size and number of features
        N = np.size(dLdA, 0) # TODO
        C = np.size(dLdA, 1) # TODO

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ = np.zeros((N, C)) # TODO

        # Fill dLdZ one data point (row) at a time
        for i in range(N):

            # Initialize the Jacobian with all zeros.
            J = np.zeros((C, C)) # TODO

            # Fill the Jacobian matrix according to the conditions described in the writeup
            for m in range(C):
                for n in range(C):
                    if m == n: # TODO
                        J[m,n] = self.A[i,m] * (1 - self.A[i,m])
                    else: 
                        J[m,n] = -self.A[i,m] * self.A[i,n]

            # Calculate the derivative of the loss with respect to the i-th input
            dLdZ[i, :] = dLdA[i, :] @ J # TODO

        return dLdZ