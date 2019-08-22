""" A Recurrent Neural Network using LSTM by Cautre_6_PORC """

import numporc as np


class RNN:

    def __init__(self, size):

        #  Input gate matrices
        self.ig = {'weights': np.randomized((size, size)), 'biases': np.randomized((1, size))}

        #  Output gate matrices
        self.og = {'weights': np.randomized((size, size)), 'biases': np.randomized((1, size))}

        #  Forget gate matrices
        self.fg = {'weights': np.randomized((size, size)), 'biases': np.randomized((1, size))}

        #  Cell gate matrices
        self.cg = {'weights': np.randomized((size, size)), 'biases': np.randomized((1, size))}

        #  Cell state matrix
        self.cs = np.zeros((1, size))

        #  Hidden state matrix
        self.hs = np.zeros((1, size))

        #  Previous output [ h(t-1) ]
        self.po = np.zeros((1, size))

    def forward(self, input):

        equations

        return output


class LSTM:

    def __init__(self):

        self.best = 6
