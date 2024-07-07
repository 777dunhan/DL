import numpy as np
from nn.activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        # Add your code here.
        # Define your variables based on the writeup using the corresponding names below:
        self.r = self.r_act.forward(self.Wrx @ x + self.brx + self.Wrh @ h_prev_t + self.brh)
        self.z = self.z_act.forward(self.Wzx @ x + self.bzx + self.Wzh @ h_prev_t + self.bzh)
        self.n0 = self.Wnh @ h_prev_t + self.bnh
        self.n = self.h_act.forward(self.Wnx @ x + self.bnx + self.r * self.n0)
        self.h_t = (1 - self.z) * self.n + self.z * h_prev_t

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert self.h_t.shape == (self.h,)

        return self.h_t



    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (hidden_dim)
            derivative of the loss wrt the input hidden h.

        """

        # SOME TIPS:
        # 1) Make sure the shapes of the calculated dWs and dbs match the initalized shapes of the respective Ws and bs
        # 2) When in doubt about shapes, please refer to the table in the writeup.
        # 3) Know that the autograder grades the gradients in a certain order, and the local autograder will tell you which gradient you are currently failing.

        # initializations
        dr = np.zeros((1, self.z.shape[0]))
        dz = delta * (-self.n[np.newaxis, :] + self.hidden[np.newaxis, :])
        dn = delta * (1 - self.z[np.newaxis, :])
        dh_prev_t = delta * self.z[np.newaxis, :]
        dx = np.zeros((1, self.x.shape[0]))

        # n
        grad_nt = dn * (1 - self.n[np.newaxis, :]**2)
        grad_rnt = grad_nt * self.r[np.newaxis, :]
        dr += grad_nt * self.n0.T
        dh_prev_t += grad_rnt @ self.Wnh
        dx += grad_nt @ self.Wnx
        self.dWnx += grad_nt.T @ self.x[np.newaxis, :]
        self.dbnx += np.sum(grad_nt, axis=0)
        self.dWnh += grad_rnt.T @ self.hidden[np.newaxis, :]
        self.dbnh += np.sum(grad_rnt, axis=0)

        # z
        grad_z = dz * self.z[np.newaxis, :] * (1-self.z[np.newaxis, :])
        summed = np.sum(grad_z, axis=0)
        dh_prev_t += np.dot(grad_z, self.Wzh)
        dx += np.dot(grad_z, self.Wzx)
        self.dWzx += grad_z.T @ self.x[np.newaxis, :]
        self.dbzx += summed
        self.dWzh += grad_z.T @ self.hidden[np.newaxis, :]
        self.dbzh += summed

        # r
        grad_r = dr * self.r[np.newaxis, :] * (1 - self.r[np.newaxis, :])
        summed = np.sum(grad_r, axis=0)
        dh_prev_t += grad_r @ self.Wrh
        dx += grad_r @ self.Wrx
        self.dWrx += grad_r.T @ self.x[np.newaxis, :]
        self.dbrx += summed
        self.dWrh += grad_r.T @ self.hidden[np.newaxis, :]
        self.dbrh += summed

        return dx.reshape(-1), dh_prev_t.reshape(-1)
