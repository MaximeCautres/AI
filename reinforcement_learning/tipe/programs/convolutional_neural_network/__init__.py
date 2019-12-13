from deep_neural_network import *
import numpy as np


def initialize_parameters_cnn(topology):
    """
    here, it is define all the parameters of the convolutional neural network.
    - L: number of layer in the CNN
    - lt: the layer type: convolutional or pooling
    - kc: kernel count
    - kd: (kernel width, kernel height, kernel depth)
    - afc: activation function of the pooling layer
    - plf: pooling function of the pooling layer
    - ps: (stride w, stride h)
    - pd: (pooling format w, polling format h)
    return: the initialized parameters of the CNN
    """
    parameters = {'L': len(topology)}
    i = 0
    for l in range(1, parameters['L']):
        if topology[l][0] == 'convolution':
            parameters['lt' + str(l)] = 'convolution'
            parameters['kc' + str(l)] = topology[l][1]
            parameters['kd' + str(l)] = (*topology[l][2], parameters['kc' + str(i)] * (i != 0) + (i == 0) * 2)
            parameters['k' + str(l)] = np.random.randn(*topology[l][1:3], parameters['kd' + str(l), 1])
            i = l
        else:
            parameters['lt' + str(l)] = 'pooling'
            parameters['afc' + str(l)] = 'relu'
            parameters['plf' + str(l)] = topology[l][1]
            parameters['ps' + str(l)] = topology[l][2]
            parameters['pd' + str(l)] = topology[l][3]
    return parameters


def convolve(A, W):
    """
    Apply weights and biases on A

    Take :
    A -- numpy matrix, non linear values of the previous layer (w_A, h_A, d, n)
    W -- numpy matrix, weights to apply (count, w_W, h_W, d, 1)


    Return :
    Z -- numpy matrix, linear values of the current layer (w_Z, h_Z, count, n) by applying the zero padding method
    """
    w_A, h_A, _, _ = A.shape
    count, w_W, h_W, _, _ = W.shape
    Z = np.zeros((w_A + w_W - 1, h_A + h_W - 1, count, A.shape[3]))

    for k in range(count):
        for w in range(1 - w_W, w_A):
            for h in range(1 - h_W, h_A):
                Z[w + w_W - 1, h + h_W - 1, k] += np.sum(A[max(w, 0):min(w + w_W, w_A), max(h, 0):min(h + h_W, h_A)] *
                                     W[k][max(0, -w):min(w_W, w_A-w), max(0, -h):min(h_W, h_A-h)], axis=(0, 1, 2))
    return Z


def pool(A, stride, pooling_format, pool_function):
    """
    Compute the pooling operation to an entry image A with the chosen stride and pooling format.

    - w_A, h_A: image format
    - psw, psh: stride on each direction
    - pw, ph: format pf the pooling on each direction
    - lw, lh: the list of w, h positions where the pooling will be applied

    Return :
    Z -- numpy matrix, after the applying of the activation function
    """

    w_A, h_A, count, _ = A.shape
    psw, psh = stride
    pw, ph = pooling_format

    lw = [0]
    cw = psw
    while cw + pw < w_A:
        lw.append(cw)
        cw += psw
    if cw != w_A:
        lw.append(w_A - pw)

    lh = [0]
    ch = psw
    while ch + ph < h_A:
        lh.append(cw)
        ch += psw
    if ch != h_A:
        lh.append(h_A - ph)

    Z = np.zeros((len(lw), len(lh), count, A.shape[3]))
    pos = np.zeros((len(lw), len(lh), count, A.shape[3]))
    for w in range(len(lw)):
        for h in range(len(lh)):
            if pool_function == 'max':
                Z[w, h], pos[w, h] = arg_max_mult(A[w:w+pw, h:h+ph])
            elif pool_function == 'min':
                Z[w, h], pos[w, h] = arg_min_mult(A[w:w+pw, h:h+ph])
            else:
                Z[w, h], pos[w, h] = average_mult(A[w:w+pw, h:h+ph]), None

    return Z, pos, lw, lh

def arg_max_mult(A):
    w_A, h_A, d_A, count = A.shape

    result = np.zeros((d_A, count)) + A[0, 0, 0, 0]
    pos = np.zeros((d_A, count, 4))
    w_coor = np.argmax(A, 0)
    h_coor = np.argmax(A, 1)

    result[d, c] = A[x, y, d, c]
    pos[d, c] = [x, y, d, c]
    return result, pos


def arg_min_mult(A):
    w_A, h_A, d_A, count = A.shape

    result = np.zeros((d_A, count)) + A[0, 0, 0, 0]
    pos = np.zeros((d_A, count, 4))
    for c in range(count):
        for d in range(d_A):
            for x in range(w_A):
                for y in range(h_A):
                    if A[x, y, d, c] < result[d, c]:
                        result[d, c] = A[x, y, d, c]
                        pos[d, c] = [x, y, d, c]
    return result, pos


def average_mult(A):
    w_A, h_A, d_A, count = A.shape

    result = np.zeros((d_A, count)) + A[0, 0, 0, 0]
    for c in range(count):
        for d in range(d_A):
            for x in range(w_A):
                for y in range(h_A):
                    result[d, c] += A[x, y, d, c]
    return result

