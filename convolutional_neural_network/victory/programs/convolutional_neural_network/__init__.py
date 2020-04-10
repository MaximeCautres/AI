""" Function for Convolutional Neural Network """

import time
import math
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

# Set constants
epsilon = pow(10, -6)
parameter_topology = {'Lc': ['wc', 'gc', 'bc'], 'Ld': ['wd', 'gd', 'bd']}


def initialize_parameters(cnn_topology, dnn_topology, input_shape):
    """
    Initialize parameters

    Take :
    cnn_topology -- dictionary, what topology the cnn have
    dnn_topology -- dictionary, what topology the dnn have
    input_shape -- numpy array containing the shape of the image

    Return :
    parameters -- dictionary containing all the information about the whole network
    """

    parameters = {**cnn_topology, **dnn_topology, 'kc0': input_shape[2]}
    current = input_shape

    parameters['gc0'] = np.ones(current + (1, ))
    parameters['bc0'] = np.zeros(current + (1, ))
    for l in range(1, parameters['Lc'] + 1):
        count, k = parameters['kc' + str(l)], 1
        w, h = parameters['kd' + str(l)]
        d = parameters['kc' + str(l - 1)]

        lx, sx, dx = [0], parameters['sc' + str(l)][0], current[0]
        while lx[-1] < dx-sx-w:
            lx.append(lx[-1]+sx)
        lx.append(dx-w)

        ly, sy, dy = [0], parameters['sc' + str(l)][1], current[1]
        while ly[-1] < dy - sy - h:
            ly.append(ly[-1] + sy)
        ly.append(dy - h)

        current = (len(lx), len(ly), count)
        parameters = set_p(parameters, (count, w, h, d, 1), current + (1, ), k, 'c', l)
        parameters['rc' + str(l)] = (lx, ly)

    parameters['nc0'] = np.prod(current)
    for l in range(1, parameters['Ld'] + 1):
        p, c, k = parameters['nc' + str(l-1)], parameters['nc' + str(l)], 10**-2
        parameters = set_p(parameters, (c, p), (c, 1), k, 'd', l)

    return parameters


def set_p(parameters, shape_w, shape_bn, k, s, l):
    """
    Set trivial parameters

    Take :
    parameters -- dictionary containing all the information about the whole network
    shape_w -- tuple, the shape of weights
    shape_bn -- tuple, the shape of mean and variance
    k -- scale random
    s -- 'c' or 'd', what network it is
    l -- the current layer

    Return :
    parameters -- dictionary containing all the information about the whole network
    """

    parameters['w' + s + str(l)] = np.random.randn(*shape_w) * k
    parameters['tw' + s + str(l)] = np.zeros(shape_w)
    parameters['qw' + s + str(l)] = np.zeros(shape_w)

    parameters['g' + s + str(l)] = np.ones(shape_bn)
    parameters['tg' + s + str(l)] = np.zeros(shape_bn)
    parameters['qg' + s + str(l)] = np.zeros(shape_bn)

    parameters['b' + s + str(l)] = np.zeros(shape_bn)
    parameters['tb' + s + str(l)] = np.zeros(shape_bn)
    parameters['qb' + s + str(l)] = np.zeros(shape_bn)

    return parameters


def train(parameters, epoch_count, mini_batch_size, optimizer, alpha, beta, gamma, rho, lambda2C, lambda2D, training, testing, save_type):
    """
    Train parameters

    Take :
    parameters -- dictionary containing all the information about the whole network
    epoch_count -- number of time we apply gradient descent
    mini_batch_size -- amount of couple (feature, label) per mini_batch
    optimizer -- 'adadelta' or 'momentum', which optimizer used
    alpha -- learning rate
    beta -- Momentum rate
    gamma -- RMS prop rate
    rho -- AdaDelta rate
    lambda2C -- L2 regularization rate for cnn
    lambda2D -- L2 regularization rate for dnn
    training -- training set
    testing -- testing set
    save_type -- 'train' or 'test' which performance matter

    Return :
    parameters -- dictionary containing all the information about the whole network
    """

    train_x, train_y = training
    test_x, test_y = testing

    costs = {'train': [], 'test': []}
    best = {'parameters': parameters, 'cost': pow(10, 6)}

    convert = lambda s: eval('lambda t: ' + s)
    alpha = convert(alpha)
    beta = convert(beta)
    gamma = convert(gamma)
    rho = convert(rho)

    count = 0
    total = epoch_count * math.ceil(train_x.shape[3] / mini_batch_size) - 1

    year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
    print("[{0} / {1} / {2} at {3} : {4}] Beginning of the training...".format(day, month, year, hour, minute))
    begin = time.time()

    for epoch in range(epoch_count):
        mini_batches = create_mini_batches(train_x, train_y, mini_batch_size)
        for x, y in mini_batches:
            t = count / total
            masks = create_masks(parameters)
            cache = forward(parameters, x, masks)
            gradients = backward(parameters, y, cache, lambda2C, lambda2D)
            parameters = update_parameters(parameters, gradients, optimizer, alpha(t), beta(t), gamma(t), rho(t))
            count += 1

        train_cost = compute_cost(parameters, train_x, train_y)
        costs['train'].append(train_cost)

        test_cost = compute_cost(parameters, test_x, test_y)
        costs['test'].append(test_cost)

        cost = costs[save_type][-1]
        if math.isnan(cost):
            print("[Iteration {0}] : early termination (overflow ?)".format(epoch+1))
            break
        elif cost < best['cost']:
            best['cost'] = cost
            best['parameters'] = deepcopy(parameters)

        print("[Iteration {0}] : train ~ {1} / test ~ {2}".format(epoch+1, train_cost, test_cost))

    delta = time.gmtime(time.time() - begin)
    print("Finished in {0} hour(s) {1} minute(s) {2} second(s).".format(delta.tm_hour, delta.tm_min, delta.tm_sec))

    # Compute statistics
    parameters = best['parameters']
    cache = forward(parameters, train_x)
    for l in range(parameters['Lc'] + 1):
        parameters['vc' + str(l)] = cache['vc' + str(l)]
        parameters['mc' + str(l)] = cache['mc' + str(l)]
    for l in range(1, parameters['Ld'] + 1):
        parameters['vd' + str(l)] = cache['vd' + str(l)]
        parameters['md' + str(l)] = cache['md' + str(l)]

    print("Accuracy train : {0} %".format(evaluate(parameters, train_x, train_y)))
    performance = evaluate(parameters, test_x, test_y)
    print("Accuracy test : {0} %".format(performance))

    plt.plot(costs['train'], color='blue')
    plt.plot(costs['test'], color='red')
    plt.ylabel("Cost")
    plt.xlabel("Iteration")
    plt.show()

    return parameters, str(performance), best['cost']


def forward(parameters, x, masks=None):
    """
    Apply a forward propagation in order to compute cache

    Take :
    parameters -- dictionary containing all the information about the whole network
    x -- features (w, h, d, n)
    masks -- dropout's mask

    Return :
    cache -- dictionary of results
    """

    n = x.shape[3]
    cache = {'xc0': x}

    g = parameters['gc0']
    b = parameters['bc0']
    xh, a, v, m = normalize(x, g, b)
    cache['xhc0'] = xh
    cache['ac0'] = a
    cache['vc0'] = v
    cache['mc0'] = m

    for l in range(1, parameters['Lc'] + 1):
        w = parameters['wc' + str(l)]
        rc = parameters['rc' + str(l)]
        x = convolve(a, w, rc)
        cache['xc' + str(l)] = x

        g = parameters['gc' + str(l)]
        b = parameters['bc' + str(l)]
        xh, z, v, m = normalize(x, g, b)
        cache['xhc' + str(l)] = xh
        cache['zc' + str(l)] = z
        cache['vc' + str(l)] = v
        cache['mc' + str(l)] = m

        af = parameters['afc' + str(l)]
        if af == 'relu':
            a = relu(z)
        elif af == 'tanh':
            a = tanh(z)
        elif af == 'sigmoid':
            a = sigmoid(z)
        cache['ac' + str(l)] = a

    a = a.reshape(-1, n)
    cache['ad0'] = a

    if masks is not None:
        a *= masks[0]

    for l in range(1, parameters['Ld'] + 1):
        w = parameters['wd' + str(l)]
        x = np.dot(w, a)
        cache['xd' + str(l)] = x

        g = parameters['gd' + str(l)]
        b = parameters['bd' + str(l)]
        xh, z, v, m = normalize(x, g, b)
        cache['xhd' + str(l)] = xh
        cache['zd' + str(l)] = z
        cache['vd' + str(l)] = v
        cache['md' + str(l)] = m

        af = parameters['afd' + str(l)]
        if af == 'relu':
            a = relu(z)
        elif af == 'softmax':
            a = softmax(z)
        elif af == 'tanh':
            a = tanh(z)
        elif af == 'sigmoid':
            a = sigmoid(z)

        if masks is not None and masks[l] is not None:
            a *= masks[l]

        cache['ad' + str(l)] = a

    return cache


def backward(parameters, y, cache, lambda2C=0, lambda2D=0):
    """
    Apply a backward propagation in order to compute gradients

    Take :
    parameters -- dictionary containing all the information about the whole network
    y -- labels (v, n)
    cache -- dictionary of results
    lambda2C -- L2 regularization rate for cnn
    lambda2D -- L2 regularization rate for dnn

    Return :
    gradients -- partial derivative of each parameters with respect to cost
    """

    gradients = {}
    n = y.shape[1]

    y_hat = cache['ad' + str(parameters['Ld'])]
    da, dz = (None, ) * 2

    for l in reversed(range(1, parameters['Ld'] + 1)):
        z = cache['zd' + str(l)]
        af = parameters['afd' + str(l)]

        if af == 'relu':
            dz = da * relu_prime(z)
        elif af == 'softmax':
            dz = y_hat - y
        elif af == 'tanh':
            dz = da * tanh_prime(z)
        elif af == 'sigmoid':
            dz = da * sigmoid_prime(z)

        g = parameters['gd' + str(l)]
        x_p = cache['xd' + str(l)]
        xh_p = cache['xhd' + str(l)]
        v = cache['vd' + str(l)]
        m = cache['md' + str(l)]
        dz, dg, db = normalize_prime(dz, g, x_p, xh_p, v, m)
        gradients['dgd' + str(l)] = dg
        gradients['dbd' + str(l)] = db

        a_p = cache['ad' + str(l - 1)]
        w = parameters['wd' + str(l)]
        gradients['dwd' + str(l)] = (1 / n) * (np.dot(dz, a_p.T) + lambda2D * np.power(w, 2))

        da = np.dot(w.T, dz)

    da = da.reshape(cache['ac' + str(parameters['Lc'])].shape)

    for l in reversed(range(1, parameters['Lc'] + 1)):
        z = cache['zc' + str(l)]
        af = parameters['afc' + str(l)]

        if af == 'relu':
            dz = da * relu_prime(z)
        elif af == 'tanh':
            dz = da * tanh_prime(z)
        elif af == 'sigmoid':
            dz = da * sigmoid_prime(z)

        g = parameters['gc' + str(l)]
        x_p = cache['xc' + str(l)]
        xh_p = cache['xhc' + str(l)]
        v = cache['vc' + str(l)]
        m = cache['mc' + str(l)]
        dx, dg, db = normalize_prime(dz, g, x_p, xh_p, v, m)
        gradients['dgc' + str(l)] = dg
        gradients['dbc' + str(l)] = db

        a_p = cache['ac' + str(l - 1)]
        w = parameters['wc' + str(l)]
        rc = parameters['rc' + str(l)]
        da, dw = deconvolve(dx, w, a_p, rc)

        gradients['dwc' + str(l)] = (1 / n) * (dw + lambda2C * np.power(w, 2))

    g = parameters['gc0']
    x_p = cache['xc0']
    xh_p = cache['xhc0']
    v = cache['vc0']
    m = cache['mc0']
    dx, dg, db = normalize_prime(da, g, x_p, xh_p, v, m)
    gradients['dgc0'] = dg
    gradients['dbc0'] = db

    gradients['dxc0'] = dx

    return gradients


def convolve(A, W, rc):
    """
    Apply weights and biases on A

    Take :
    A -- numpy matrix, non linear values of the previous layer (w_A, h_A, d, n)
    W -- numpy matrix, weights to apply (count, w_W, h_W, d, 1)
    rc -- tuple, range of values of w and h

    Return :
    Z -- numpy matrix, linear values of the current layer (w_Z, h_Z, count, n)
    """

    lx, ly = rc
    tx, ty = len(lx), len(ly)
    count, w_W, h_W, _, _ = W.shape
    Z = np.zeros((tx, ty, count, A.shape[3]))

    for k in range(count):
        for x in range(tx):
            for y in range(ty):
                w, h = lx[x], ly[y]
                Z[x, y, k] += np.sum(A[w:w+w_W, h:h+h_W] * W[k], axis=(0, 1, 2))

    return Z


def normalize(x, g, b):
    """
    Normalize x

    Take :
    x -- numpy matrix, linear values of the current layer (w_Z, h_Z, count, n)
    g -- new variance
    b -- new mean

    Return :
    xh --
    Z -- numpy matrix, normalized linear values of the current layer (w_Z, h_Z, count, n)
    v --
    m --
    """

    v = np.var(x, axis=-1, keepdims=True)
    m = np.mean(x, axis=-1, keepdims=True)

    xh = (x - m) / rms(v)
    Z = g * xh + b

    return xh, Z, v, m


def normalize_prime(dZ, g, x_p, xh_p, v, m):
    """
    Normalize_prime

    Take :
    dZ -- numpy matrix, linear values of the current layer (w_Z, h_Z, count, n)
    g -- variance
    x_p --
    xh_p --
    v --
    m --

    Return :
    dx -- numpy matrix, normalized linear values of the current layer (w_Z, h_Z, count, n)
    dg --
    db --
    """

    n = dZ.shape[-1]
    dx = dZ * g * (n-1 - np.power(x_p - m, 2) * np.power(v + epsilon, -1)) / (n * rms(v))

    dg = np.mean(dZ * xh_p, axis=-1, keepdims=True)
    db = np.mean(dZ, axis=-1, keepdims=True)

    return dx, dg, db


def deconvolve(dZ, W, A_p, rc):
    """
    Compute gradients

    Take :
    dZ -- numpy matrix, gradients of the next layer (w_dZ, h_dZ, count, n)
    W -- numpy matrix, weights (count, w_W, h_W, d, 1)
    A_p -- previous A (w_dA, h_dA, d, n)
    rc -- range of values of w and h

    Return :
    dA -- numpy matrix, gradients of the current layer (w_dA, h_dA, d, n)
    dW -- numpy matrix, weights gradients (count, w_W, h_W, d)
    db -- numpy matrix, biases gradients (1, 1, count, 1)
    """

    lx, ly = rc
    w_dZ, h_dZ, count, n = dZ.shape
    _, w_W, h_W, _, _ = W.shape

    dA = np.zeros_like(A_p)
    dW = np.zeros_like(W)

    for k in range(count):
        for x in range(w_dZ):
            for y in range(h_dZ):
                w, h, dZ_n = lx[x], ly[y], dZ[x, y, k].reshape(1, 1, 1, n)
                dA[w:w+w_W, h:h+h_W] += W[k] * dZ_n
                dW[k] += np.sum(A_p[w:w+w_W, h:h+h_W] * dZ_n, axis=3, keepdims=True)

    return dA, dW


def update_parameters(parameters, gradients, optimizer, alpha, beta, gamma, rho):
    """
    Update each parameters in order to reduce cost

    Take :
    parameters -- dictionary containing all the information about the whole network
    gradients -- partial derivative of each parameters with respect to cost
    optimizer -- 'adadelta' or 'momentum', which optimizer used
    alpha -- learning rate
    beta -- Momentum rate
    gamma -- RMS prop rate
    rho -- AdaDelta rate

    Return :
    parameters -- dictionary containing all the information about the whole network
    """

    if optimizer == 'momentum':
        parameters = apply_momentum(parameters, gradients, alpha, beta)
    elif optimizer == 'adadelta':
        parameters = apply_adadelta(parameters, gradients, rho)
    elif optimizer == 'rmsprop':
        parameters = apply_rmsprop(parameters, gradients, alpha, gamma)
    elif optimizer == 'shit':
        parameters = apply_shit(parameters, gradients, alpha)
    else:
        print("Warning, no optimizer selected !")

    return parameters


def apply_momentum(parameters, gradients, alpha, beta):
    """
    Apply adadelta in order to update weights and biases

    Take :
    parameters -- dictionary containing all the information about the whole network
    gradients -- partial derivative of each parameters with respect to cost
    alpha -- learning rate
    beta -- Momentum rate

    Return :
    parameters -- dictionary containing all the information about the whole network
    """

    for k, types in parameter_topology.items():
        for l in range(1, parameters[k] + 1):
            for v in types:
                d = gradients['d' + v + str(l)]
                q = beta * parameters['q' + v + str(l)] + (1 - beta) * d

                parameters[v + str(l)] -= alpha * q
                parameters['q' + v + str(l)] = q

    return parameters


def apply_adadelta(parameters, gradients, rho):
    """
    Apply adadelta in order to update weights and biases

    Take :
    parameters -- dictionary containing all the information about the whole network
    gradients -- partial derivative of each parameters with respect to cost
    rho -- AdaDelta rate

    Return :
    parameters -- dictionary containing all the information about the whole network
    """

    for k, types in parameter_topology.items():
        for l in range(1, parameters[k] + 1):
            for v in types:
                d = gradients['d' + v + str(l)]
                t = parameters['t' + v + str(l)]
                q = parameters['q' + v + str(l)]

                t = rho * t + (1 - rho) * d * d
                c = np.divide(rms(q), rms(t)) * d
                q = rho * q + (1 - rho) * c * c

                parameters[v + str(l)] -= c
                parameters['t' + v + str(l)] = t
                parameters['q' + v + str(l)] = q

    return parameters


def apply_rmsprop(parameters, gradients, alpha, gamma):
    """
    Apply RMS prop in order to update weights and biases

    Take :
    parameters -- dictionary containing all the information about the whole network
    gradients -- partial derivative of each parameters with respect to cost
    alpha -- learning rate
    gamma -- RMS prop rate

    Return :
    parameters -- dictionary containing all the information about the whole network
    """

    for k, types in parameter_topology.items():
        for l in range(1, parameters[k] + 1):
            for v in types:
                d = gradients['d' + v + str(l)]
                q = gamma * parameters['q' + v + str(l)] + (1 - gamma) * d * d

                parameters[v + str(l)] -= alpha * np.divide(d, rms(q))
                parameters['q' + v + str(l)] = q

    return parameters


def apply_shit(parameters, gradients, alpha):
    """
    Apply shit

    Take :
    parameters -- dictionary containing all the information about the whole network
    gradients -- partial derivative of each parameters with respect to cost
    alpha -- learning rate

    Return :
    parameters -- dictionary containing all the information about the whole network
    """

    for k, types in parameter_topology.items():
        for l in range(1, parameters[k] + 1):
            for v in types:
                parameters[v + str(l)] -= alpha * gradients['d' + v + str(l)]

    return parameters


def predict(parameters, x):
    """
    Make a prediction of what x is

    Take :
    parameters -- dictionary containing all the information about the whole network
    x -- features (w, h, d, n)

    Return :
    a -- prediction (v)
    """

    normal = lambda x_, g_, b_, v_, m_: g_ * (x_ - m_) / rms(v_) + b_

    n = x.shape[3]

    g = parameters['gc0']
    b = parameters['bc0']
    v = parameters['vc0']
    m = parameters['mc0']
    a = normal(x, g, b, v, m)

    for l in range(1, parameters['Lc'] + 1):
        w = parameters['wc' + str(l)]
        rc = parameters['rc' + str(l)]
        x = convolve(a, w, rc)

        g = parameters['gc' + str(l)]
        b = parameters['bc' + str(l)]
        v = parameters['vc' + str(l)]
        m = parameters['mc' + str(l)]
        z = normal(x, g, b, v, m)

        af = parameters['afc' + str(l)]
        if af == 'relu':
            a = relu(z)
        elif af == 'tanh':
            a = tanh(z)
        elif af == 'sigmoid':
            a = sigmoid(z)

    a = a.reshape(-1, n)

    for l in range(1, parameters['Ld'] + 1):
        w = parameters['wd' + str(l)]
        x = np.dot(w, a)

        g = parameters['gd' + str(l)]
        b = parameters['bd' + str(l)]
        v = parameters['vd' + str(l)]
        m = parameters['md' + str(l)]
        z = normal(x, g, b, v, m)

        af = parameters['afd' + str(l)]
        if af == 'relu':
            a = relu(z)
        elif af == 'softmax':
            a = softmax(z)
        elif af == 'tanh':
            a = tanh(z)
        elif af == 'sigmoid':
            a = sigmoid(z)

    return np.squeeze(a)


def create_mini_batches(x, y, size):
    """
    Generate an array of couple (features, labels) shuffled

    Take :
    x -- features (w, h, d, n)
    y -- labels (v, n)
    size -- number of couple per mini_batch

    Return :
    couples -- array of couple
    """

    couples = []
    n = y.shape[1]

    permutation = list(np.random.permutation(np.arange(n, dtype=np.int16)))

    shuffled_x = x[:, :, :, permutation]
    shuffled_y = y[:, permutation]

    num_complete_mini_batches = int(n / size)

    for k in range(num_complete_mini_batches):
        dx = shuffled_x[:, :, :, k * size: (k+1) * size]
        dy = shuffled_y[:, k * size: (k+1) * size]
        couples.append((dx, dy))

    if n % size != 0:
        dx = shuffled_x[:, :, :, num_complete_mini_batches * size: n]
        dy = shuffled_y[:, num_complete_mini_batches * size: n]
        couples.append((dx, dy))

    return couples


def create_masks(parameters):
    """
    Generate an array of mask

    Take :
    parameters -- dictionary containing all the information about the whole network

    Return :
    masks -- dropout's mask
    """

    masks = []
    acc = 0

    for l in range(parameters['Ld']):
        dor = parameters['dor' + str(l)]
        acc += dor
        mask = np.random.rand(parameters['nc' + str(l)], 1) > dor
        masks.append(np.divide(mask, 1-dor))

    if acc == 0:
        return None

    masks.append(None)

    return masks


def compute_cost(parameters, x, y):
    """
    Compute the cost for estimation y_hat of y

    Take :
    parameters -- dictionary containing all the information about the whole network
    x -- features (w, h, d, n)
    y -- labels (v, n)

    Return :
    cost -- global error
    """

    n = y.shape[1]
    y_hat = forward(parameters, x)['ad' + str(parameters['Ld'])]

    if parameters['afd' + str(parameters['Ld'])] == 'softmax':
        loss = np.multiply(np.log(y_hat), y)
    else:
        loss = np.multiply(np.log(y_hat), y) - np.multiply(np.log(1 - y_hat), 1 - y)

    cost = (-1 / n) * np.sum(loss)

    return round(float(np.squeeze(cost)), 4)


def evaluate(parameters, x, y):
    """
    Evaluate the performance of this network

    Take :
    parameters -- dictionary containing all the information about the whole network
    x -- features (w, h, d, n)
    y -- labels (v, n)

    Return :
    percent -- rate of success
    """

    percent = 0
    count, n = y.shape
    y_hat = forward(parameters, x)['ad' + str(parameters['Ld'])]

    for i, j in zip(y_hat.T, y.T):
        m = np.max(i)
        for k in range(count):
            if i[k] == m:
                i[k] = 1
            else:
                i[k] = 0
        if np.all(i == j):
            percent += 1

    return round(percent * 100 / n, 4)


def relu(z):
    """
    Apply the relu function on each element of z

    Take :
    z -- numpy matrix

    Return :
    relu(z)
    """

    return np.maximum(z, 0)


def relu_prime(z):
    """
    Apply the derivative of the relu function on each element of z

    Take :
    z -- numpy matrix

    Return :
    relu_prime(z)
    """

    return z > 0


def tanh(z):
    """
    Apply the tanh function on each element of z

    Take :
    z -- numpy matrix

    Return :
    tanh(z)
    """

    return np.tanh(z)


def tanh_prime(z):
    """
    Apply the derivative of the tanh function on each element of z

    Take :
    z -- numpy matrix

    Return :
    tanh_prime(z)
    """

    tanhz = tanh(z)

    return 1 - tanhz * tanhz


def sigmoid(z):
    """
    Apply the sigmoid function on each element of z

    Take :
    z -- numpy matrix

    Return :
    sigmoid(z)
    """

    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    """
    Apply the derivative of the sigmoid function on each element of z

    Take :
    z -- numpy matrix

    Return :
    sigmoid_prime(z)
    """

    sigmoidz = sigmoid(z)

    return sigmoidz * (1 - sigmoidz)


def softmax(z):
    """
    Apply the softmax function on each element of z

    Take :
    z -- numpy matrix

    Return :
    softmax(z)
    """

    return np.divide(np.exp(z), np.sum(np.exp(z), axis=0, keepdims=True))


def rms(z):
    """
    Root Mean Square

    Take :
    z -- numpy matrix

    Return :
    rms(z)
    """

    return np.sqrt(z + epsilon)
