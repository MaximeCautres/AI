import numpy as np


def initialize_parameters(parameters, dnn_topology):
    """
    Create the parameters we need to train

    Take :
    parameters -- dictionary containing the whole network
    dnn_topology -- tuple the number of neurons on each layer

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

    current = list(parameters['idi'])
    for l in range(1, parameters['Lc']):
        if parameters['lt' + str(l)] == 'c':
            w, h, k_d = current
            k_w, k_h, k_c = parameters['kd' + str(l)]
            parameters['k' + str(l)] = np.random.randn(k_w, k_h, k_d, k_c, 1)
            current = [w + k_w - 1, h + k_h - 1, k_c]
        else:
            s_x, s_y = parameters['ss' + str(l)]
            p_w, p_h = parameters['pd' + str(l)]
            w, h = current[:2]
            pr_x = [x for x in range(w) if not (x % s_x) and x + p_w <= w or x + p_w == w]
            pr_y = [y for y in range(h) if not (y % s_y) and y + p_h <= h or y + p_h == h]
            parameters['afc' + str(l)] = 'relu'
            parameters['pf' + str(l)] = 'mean'
            parameters['pr' + str(l)] = (pr_x, pr_y)
            current[:2] = [len(pr_x), len(pr_y)]

    parameters['idf'] = tuple(current)
    parameters['Ld'] = len(dnn_topology)
    for l in range(1, parameters['Ld']):
        cond = l < parameters['Ld'] - 1
        parameters['afd' + str(l)] = 'relu' * cond + 'softmax' * (not cond)
        parameters['w' + str(l)] = np.random.randn(*dnn_topology[l - 1:l + 1][::-1]) * 10 ** -1
        parameters['b' + str(l)] = np.zeros((dnn_topology[l], 1))

    return parameters


def forward(parameters, X, return_cache=False):
    """
    Evaluate the whole network

    Take :
    parameters -- dictionary containing the whole network
    X -- input image (w_X, h_X, d, n)
    return_cache -- Specify if we want the cache

    Return :
    cache or output
    """

    A = X
    n = X.shape[3]
    cache = {}

    for l in range(1, parameters['Lc']):
        if parameters['lt' + str(l)] == 'c':
            K = parameters['k' + str(l)]
            cache['A' + str(l-1)] = A
            A = convolve(A, K)
        else:
            pr, pd = parameters['pr' + str(l)], parameters['pd' + str(l)]
            pf, af = parameters['pf' + str(l)], parameters['afc' + str(l)]
            Y, A = pool(A, pr, pd, pf, af)
            cache['Y' + str(l)] = Y

    a = A.reshape(-1, n)
    cache['a0'] = a

    for l in range(1, parameters['Ld']):
        w = parameters['w' + str(l)]
        b = parameters['b' + str(l)]
        af = parameters['afd' + str(l)]

        z = np.dot(w, a) + b
        if af == 'relu':
            a = relu(z)
        elif af == 'softmax':
            a = softmax(z)

        cache['z' + str(l)] = z
        cache['a' + str(l)] = a

    if return_cache:
        return cache
    else:
        return a


def backward(parameters, X, y):
    gradients = {}
    n = X.shape[1]
    cache = forward(parameters, X, True)
    y_hat = cache['a' + str(parameters['Ld'] - 1)]

    da = np.divide(1 - y, 1 - y_hat) - np.divide(y, y_hat)
    dz = None

    for l in reversed(range(1, parameters['Ld'])):
        z = cache['z' + str(l)]
        af = parameters['afd' + str(l)]

        if af == 'relu':
            dz = da * relu_prime(z)
        elif af == 'softmax':
            dz = y_hat - y

        a_p = cache['a' + str(l - 1)]
        w = parameters['w' + str(l)]

        gradients['dw' + str(l)] = (1 / n) * np.dot(dz, a_p.T)
        gradients['db' + str(l)] = (1 / n) * np.sum(dz, axis=1, keepdims=True)

        da = np.dot(w.T, dz)

    dA = da.reshape(*parameters['idf'], n)

    for l in reversed(range(1, parameters['Lc'])):
        if parameters['lt' + str(l)] == 'c':
            K = parameters['k' + str(l)]
            A_p = cache['A' + str(l - 1)]
            dA, dK = deconvolve(dA, K, A_p)
            gradients['dK' + str(l)] = dK
        else:
            Y_p = cache['Y' + str(l - 1)]
            dA = depool(dA, Y_p)
            gradients['dA' + str(l)] = dA

    return gradients


def update_parameters(parameters, gradients, alpha):
    for l in range(1, parameters['Ld']):
        parameters['w' + str(l)] -= alpha * gradients['dw' + str(l)]
        parameters['b' + str(l)] -= alpha * gradients['db' + str(l)]
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
            Z[x + w_K - 1, y + h_K - 1] = np.sum(prod, axis=(0, 1, 2))

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
            A_ = A[x:x + p_w, y:y + p_h]
            if pf == 'min':
                Y[i, j] = pool_min(A_)
            elif pf == 'max':
                Y[i, j] = pool_max(A_)
            elif pf == 'mean':
                Y[i, j] = pool_mean(A_)

    Z = None
    if af == 'relu':
        Z = relu(Y)

    return Y, Z


def deconvolve(dZ, K, A_p):
    """
    Make a deconvolution of K on Z by applying the zero padding method

    Take :
    dZ -- current layer gradient image -> (w_Z, h_Z, count, n)
    K -- kernel to apply -> (w_K, h_K, d, count, 1)
    A_p -- previous layer image -> (w_A, h_A, d, n)

    Return :
    dA -- previous layer gradient image -> (w_A, h_A, d, n)
    """

    w_Z, h_Z, count, n = dZ.shape
    w_K, h_K, d, _, _ = K.shape
    w_A, h_A = w_Z + 1 - w_K, h_Z + 1 - h_K

    dA = np.zeros((w_A, h_A, d, n))
    dK = np.zeros((w_K, h_K, d, count, 1))
    A_p_ = A_p.reshape(w_A, h_A, d, 1, n)

    for x in range(w_Z):
        K_x = K[max(0, w_K - 1 - x):min(w_K, w_Z - x)]
        A_p_x = A_p_[max(x - w_K + 1, 0):min(x + 1, w_A)]
        for y in range(h_Z):
            dZ_ = dZ[x, y].reshape(1, 1, 1, count, n)

            prod_dA = dZ_ * K_x[:, max(0, h_K - 1 - y):min(h_K, h_Z - y)]
            dA[max(x - w_K + 1, 0):min(x + 1, w_A), max(y - h_K + 1, 0):min(y + 1, h_A)] \
                += np.sum(prod_dA, axis=3)

            prod_dK = dZ_ * A_p_x[:, max(y - h_K + 1, 0):min(y + 1, h_A)]
            dK[max(0, w_K - 1 - x):min(w_K, w_Z - x), max(0, h_K - 1 - y):min(h_K, h_Z - y)] \
                += np.mean(prod_dK, axis=4, keepdims=True)

    return dA, dK


def depool(dZ, Y_p, pr, pd, pf, af):
    """
    Depool dZ

    Take :
    dZ -- current layer gradient image -> (w_Z, h_Z, d, n)
    Y_p -- previous linear image -> (w_Z, h_Z, d, n)
    pr -- pooling ranges -> (pr_x, pr_y)
    pd -- pooling dimensions -> (p_w, p_h)
    pf -- pooling function -> 'min' or 'max' or 'mean'
    af -- activation function -> 'relu'


    Return :
    dA -- previous layer gradient image -> (w_A, h_A, d, n)
    """

    w_Z, h_Z, d, n = dZ.shape
    pr_x, pr_y = pr
    pd_x, pd_y = pd
    area = pd_x * pd_y
    dA = np.zeros((pr_x[-1] + pd_x, pr_y[-1] + pd_y, d, n))

    dY = None
    if af == relu:
        dY = relu_prime(Y_p) * dZ

    for x in range(w_Z):
        for y in range(h_Z):
            dY_ = dY[x, y].reshape(1,1, d, n)
            if pf == 'min':
                pass
            elif pf == 'max':
                pass
            elif pf == 'mean':
                dA[pr_x[x]:pr_x[x] + pd_x, pr_y[y]:pr_y[y] + pd_y] += dY_ / area

    return dA


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


def relu(z):
    return np.maximum(z, 0)


def relu_prime(z):
    return z > 0


def softmax(z):
    return np.divide(np.exp(z), np.sum(np.exp(z), axis=0, keepdims=True))
