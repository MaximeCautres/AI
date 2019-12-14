from deep_neural_network import *
import numpy as np


def initialize_parameters_cnn(parameters):
    """
    Create the parameters we need to train

    Take :
    parameters -- dictionary containing the whole network

    Notation :
    - L: number of layer in the CNN
    - lt: the layer type -> 'c' or 'p'
    - kd: kernel dimensions - > (k_w, k_h, k_c)
    - ps: stride -> (s_x, s_y)
    - pd: pooling dimensions -> (p_w, p_h)
    - af: pooling activation function -> 'relu'
    - pf: pooling function -> 'min' or 'max' or 'mean'

    Return :
    parameters -- dictionary containing the whole network
    """

    current = parameters['id']
    for l in range(1, parameters['L']):
        if parameters['lt' + str(l)] == 'c':
            k_d = current[2]
            k_w, k_h, k_c = parameters['kd' + str(l)]
            parameters['k' + str(l)] = np.random.randn(k_w, k_h, k_d, k_c, 1)
            current[2] = k_c
        else:
            s_x, s_y = parameters['ss' + str(l)]
            p_w, p_h = parameters['pd' + str(l)]
            w, h = current[:2]
            pr_x = [x for x in range(w) if not (x % s_x) and x + p_w <= w or x + p_w == w]
            pr_y = [y for y in range(h) if not (y % s_y) and y + p_h <= h or y + p_h == h]
            parameters['af' + str(l)] = 'relu'
            parameters['pf' + str(l)] = 'max'
            parameters['pr' + str(l)] = (pr_x, pr_y)
            current[:2] = [len(pr_x), len(pr_y)]

    return parameters


def convolve(A, K):
    """
    Make a convolution of K on A by applying the zero padding method

    Take :
    A -- previous layer image -> (w_A, h_A, d, n)
    K -- kernel to apply -> (w_K, h_K, d, count, 1)

    Return :
    Z -- current layer image -> (w_Z, h_Z, count, n)
    """

    w_A, h_A, d, n = A.shape
    w_K, h_K, _, count, _ = K.shape
    Z = np.zeros((w_A + w_K - 1, h_A + h_K - 1, count, n))
    A_ = A.reshape(w_A, h_A, d, 1, n)

    for x in range(1 - w_K, w_A):
        A_x = A_[max(x, 0):min(x + w_K, w_A)]
        K_x = K[max(0, -x):min(w_K, w_A - x)]
        for y in range(1 - h_K, h_A):
            prod = A_x[:, max(y, 0):min(y + h_K, h_A)] \
                   * K_x[:, max(0, -y):min(h_K, h_A - y)]
            Z[x, y] = np.sum(prod, axis=(0, 1, 2))

    return Z


def pool(A, pr, pd, pf, af):
    """
    Pool A

    Take :
    A -- previous layer image -> (w_A, h_A, d, n)
    pr -- pooling ranges -> (pr_x, pr_y)
    pd -- pooling dimensions -> (p_w, p_h)
    pf -- pooling function -> 'min' or 'max' or 'mean'
    af -- activation function -> 'relu'

    Return :
    Z -- current layer image -> (w_Z, h_Z, d, n)
    """

    w_A, h_A, d, n = A.shape
    pr_x, pr_y = pr
    p_w, p_h = pd
    w_Z, h_Z = len(pr_x), len(pr_y)
    Y = np.zeros((w_Z, h_Z, d, n))

    for i in range(w_Z):
        x = pr_x[i]
        for j in range(h_Z):
            y = pr_y[j]
            X = A[x:x + p_w, y:y + p_h]
            if pf == 'min':
                Y[i, j] = pool_min(X)
            elif pf == 'max':
                Y[i, j] = pool_max(X)
            elif pf == 'mean':
                Y[i, j] = pool_mean(X)

    Z = None
    if af == 'relu':
        Z = relu(Y)

    return Z


def pool_min(X):
    """
    Apply a pool_min on X

    Take :
    X -- image -> (w_X, h_X, d, n)

    Return :
    Y -- pooled_min image -> (1, 1, d, n)
    """

    _, _, d, n = X.shape
    X_bread = X.reshape(-1, d, n)
    X_min = np.argmin(X_bread, axis=0).reshape((1, 1, d, n))
    Y = np.min(X_bread, axis=0, keepdims=True)

    return Y


def pool_max(X):
    """
    Apply a pool_max on X

    Take :
    X -- image -> (w_X, h_X, d, n)

    Return :
    Y -- pooled_max image -> (1, 1, d, n)
    """

    _, _, d, n = X.shape
    X_bread = X.reshape(-1, d, n)
    X_max = np.argmax(X_bread, axis=0).reshape((1, 1, d, n))
    Y = np.max(X_bread, axis=0, keepdims=True)

    return Y


def pool_mean(X):
    """
    Apply a pool_mean on X

    Take :
    X -- image -> (w_X, h_X, d, n)

    Return :
    Y -- pooled_mean image -> (1, 1, d, n)
    """

    _, _, d, n = X.shape
    X_bread = X.reshape(-1, d, n)
    Y = np.mean(X_bread, axis=0, keepdims=True)

    return Y
