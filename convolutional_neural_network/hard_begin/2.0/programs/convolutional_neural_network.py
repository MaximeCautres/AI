""" Function for Convolutional Neural Network """

import time
import numpy as np
from scipy import ndimage


def initialize_parameters(cnn_topology, dnn_topology):
    """
    Initialize parameters

    Take :
    inputs_dim -- tuple, the dimensions of the inputs images
    cnn_topology -- dictionary, what topology the cnn have
    dnn_topology -- tuple, what topology the dnn have

    Return :
    parameters -- dictionary, all the information needed for the training
    """

    parameters = {**cnn_topology, **dnn_topology}

    for l in range(1, parameters['Lc'] + 1):
        count, k = parameters['kc' + str(l)], 10**-3
        (w, h) = parameters['kd' + str(l)]
        d = parameters['kc' + str(l - 1)]
        parameters['wc' + str(l)] = np.random.randn(count, d, h, w).astype(np.float32) * k
        parameters['bc' + str(l)] = np.zeros((parameters['kc' + str(l)], )).astype(np.float32)

    for l in range(1, parameters['Ld'] + 1):
        k = 10**-3
        parameters['wd' + str(l)] = np.random.randn(parameters['nc' + str(l)], parameters['nc' + str(l-1)]).astype(np.float32) * k
        parameters['bd' + str(l)] = np.zeros((parameters['nc' + str(l)], 1)).astype(np.float32)

    return parameters


def train(parameters, epoch_count, mini_batch_size, alpha, training, testing):
    """
    Train parameters

    Take :
    parameters -- dictionary containing all the information about the whole network
    epoch_count -- number of time we apply gradient descent
    mini_batch_size -- amount of couple (feature, label) per mini_batch
    alpha -- amount of learning at each gradient descent
    training -- training set
    testing -- testing set
    Return :
    parameters -- dictionary containing all the information about the whole network
    """

    (train_x, train_y) = training
    (test_x, test_y) = testing

    year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
    print("[{0} / {1} / {2} at {3} : {4}] Beginning of the training...".format(day, month, year, hour, minute))
    begin = time.time()

    for epoch in range(1, epoch_count + 1):
        mini_batches = create_mini_batches(train_x, train_y, mini_batch_size)
        for x, y in mini_batches:
            cache = forward(parameters, x)
            # gradients = backward(parameters, y, cache)
            # parameters = update_parameters(parameters, gradients, alpha)
        cost = 0
        for x, y in mini_batches:
            cost += compute_cost(parameters, x, y)
        cost /= len(mini_batches)
        print("Cost after iteration {} : {}".format(epoch, cost))

    delta = time.gmtime(time.time() - begin)
    print("Finished in {0} hour(s) {1} minute(s) {2} second(s).".format(delta.tm_hour, delta.tm_min, delta.tm_sec))

    print("Accuracy train : {0} %".format(evaluate(parameters, train_x, train_y) * 100))
    print("Accuracy test : {0} %".format(evaluate(parameters, test_x, test_y) * 100))

    return parameters


def forward(parameters, x):
    """
    Apply a forward propagation in order to compute cache

    Take :
    parameters -- dictionary containing all the information about the whole network
    x -- features (n, d, h, w)

    Return :
    cache -- dictionary of results
    """

    cache = {'ac0': x}
    n = x.shape[0]
    a = x

    for l in range(1, parameters['Lc'] + 1):
        w = parameters['wc' + str(l)]
        b = parameters['bc' + str(l)]
        s = parameters['sc' + str(l)]

        z = convolve(a, w, b, s)

        if parameters['afc' + str(l)] == 'relu':
            a = relu(z)
        elif parameters['afc' + str(l)] == 'tanh':
            a = tanh(z)
        elif parameters['afc' + str(l)] == 'sigmoid':
            a = sigmoid(z)

        cache['zc' + str(l)] = z
        cache['ac' + str(l)] = a

    a = a.T.reshape(-1, n)
    cache['ad0'] = a

    for l in range(1, parameters['Ld'] + 1):
        w = parameters['wd' + str(l)]
        b = parameters['bd' + str(l)]

        z = np.dot(w, a) + b

        if parameters['afd' + str(l)] == 'relu':
            a = relu(z)
        elif parameters['afd' + str(l)] == 'tanh':
            a = tanh(z)
        elif parameters['afd' + str(l)] == 'sigmoid':
            a = sigmoid(z)
        elif parameters['afd' + str(l)] == 'softmax':
            a = softmax(z)

        cache['zd' + str(l)] = z
        cache['ad' + str(l)] = a

    return cache


def backward(parameters, y, cache):
    """
    Apply a backward propagation in order to compute gradients

    Take :
    parameters -- dictionary containing all the information about the whole network
    y -- labels (v, n)
    cache -- dictionary of results

    Return :
    gradients -- partial derivative of each parameters with respect to cost
    """

    gradients = {}
    n = y.shape[1]

    y_hat = cache['ad' + str(parameters['Ld'])]
    da = np.divide(1 - y, 1 - y_hat) - np.divide(y, y_hat)
    dz = None

    for l in reversed(range(1, parameters['Ld'] + 1)):
        z = cache['zd' + str(l)]

        if parameters['afd' + str(l)] == 'relu':
            dz = da * relu_prime(z)
        elif parameters['afd' + str(l)] == 'tanh':
            dz = da * tanh_prime(z)
        elif parameters['afd' + str(l)] == 'sigmoid':
            dz = da * sigmoid_prime(z)
        elif parameters['afd' + str(l)] == 'softmax':
            dz = y_hat - y

        a_prev = cache['ad' + str(l - 1)]
        w = parameters['wd' + str(l)]

        gradients['dwd' + str(l)] = (1 / n) * np.dot(dz, a_prev.T)
        gradients['dbd' + str(l)] = (1 / n) * np.sum(dz, axis=1, keepdims=True)

        da = np.dot(w.T, dz)

    n, d, h, w = cache['ac' + str(parameters['Lc'])].shape
    da = da.reshape(w, h, d, n).T

    for l in reversed(range(1, parameters['Lc'] + 1)):
        z = cache['zc' + str(l)]

        if parameters['afc' + str(l)] == 'relu':
            dz = da * relu_prime(z)
        elif parameters['afc' + str(l)] == 'tanh':
            dz = da * tanh_prime(z)
        elif parameters['afc' + str(l)] == 'sigmoid':
            dz = da * sigmoid_prime(z)

        w = parameters['wc' + str(l)]
        b = parameters['bc' + str(l)]
        ns = cache['ac' + str(l)].shape

        da, dw, db = deconvolve(dz, w, b, ns)

        gradients['dwc' + str(l)], gradients['dbc' + str(l)] = dw, db

    return gradients


def update_parameters(parameters, gradients, alpha):
    """
    Update each parameters in order to reduce cost

    Take :
    parameters -- dictionary containing all the information about the whole network
    gradients -- partial derivative of each parameters with respect to cost
    alpha -- amount of learning at each gradient descent

    Return :
    parameters -- dictionary containing all the information about the whole network
    """

    for l in range(1, parameters['Lc'] + 1):
        dw = gradients['dwc' + str(l)]
        db = gradients['dbc' + str(l)]
        parameters['wc' + str(l)] -= dw * alpha
        parameters['bc' + str(l)] -= db * alpha

    for l in range(1, parameters['Ld'] + 1):
        dw = gradients['dwd' + str(l)]
        db = gradients['dbd' + str(l)]
        parameters['wd' + str(l)] -= dw * alpha
        parameters['bd' + str(l)] -= db * alpha

    return parameters


def convolve(a, w, b, s):
    """
    Apply weights on a

    Take :
    a -- numpy matrix, non linear values of the previous layer (n, depth, ah, aw)
    w -- numpy matrix, weights to apply (count, d, kh, kw)
    b -- numpy matrix, biases to apply (count,)
    s -- tuple, (sh, sw)

    Return :
    z -- numpy matrix, linear values of this layer (n, count, zh, zw)
    """

    n, _, ah, aw = a.shape
    count, d, _, _ = w.shape
    p, (q, r) = int((d - 1) / 2), s

    z = np.zeros((n, count, int(ah / q), int(aw / r)))
    for t in range(n):
        stack = np.zeros((1,))
        for k in range(count):
            chunk = ndimage.convolve(a[t], w[k], mode='constant')[p, ::q, ::r] + b[k]
            if (stack != 0).any():
                chunk = np.concatenate((stack, chunk), axis=0)
            stack = chunk
        z[t] = stack

    return z


def deconvolve(dz, w, b, ns):
    """
    Compute gradients

    Take :
    dz -- numpy matrix, gradients of the next layer (n, depth, ah, aw)
    w -- numpy matrix, weights to apply (count, d, kh, kw)
    b -- numpy matrix, biases to apply (count,)
    ns -- tuple

    Return :
    da -- numpy matrix, linear values of this layer (n, count, zh, zw)
    dw -- numpy matrix, weights gradients
    db -- numpy matrix, biases gradients
    """

    n, _, ah, aw = dz.shape
    count, d, _, _ = w.shape

    da = np.zeros(ns)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    for t in range(n):
        stack = np.zeros((1,))
        for k in range(count):
            chunk = ndimage.convolve(np.ones_like(da[t]), w[k], mode='constant')

            if (stack != 0).any():
                chunk = np.concatenate((stack, np.array([chunk])), axis=0)
            stack = chunk
        da[t] = stack

    return da


def compute_cost(parameters, x, y):
    """
    Compute the cost for estimation y_hat of y

    Take :
    parameters -- dictionary containing all the information about the whole network
    x -- features (n, d, h, w)
    y -- labels (v, n)

    Return :
    cost -- global error (1, 1)
    """

    n = y.shape[1]
    y_hat = forward(parameters, x)['ad' + str(parameters['Ld'])]

    if parameters['afd' + str(parameters['Ld'])] == 'softmax':
        loss = np.multiply(np.log(y_hat), y)
    else:
        loss = np.multiply(np.log(y_hat), y) - np.multiply(np.log(1 - y_hat), 1 - y)

    cost = (-1 / n) * np.sum(loss)

    return np.squeeze(cost)


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

    shuffled_x = x[permutation]
    shuffled_y = y[:, permutation]

    num_complete_mini_batches = int(n / size)

    for k in range(num_complete_mini_batches):
        dx = shuffled_x[k * size: (k+1) * size]
        dy = shuffled_y[:, k * size: (k+1) * size]
        couple = (dx, dy)
        couples.append(couple)

    if n % size != 0:
        dx = shuffled_x[num_complete_mini_batches * size: n]
        dy = shuffled_y[:, num_complete_mini_batches * size: n]
        couple = (dx, dy)
        couples.append(couple)

    return couples


def evaluate(parameters, x, y):
    """
    Evaluate the performance of this network

    Take :
    parameters -- dictionary containing all the information about the whole network
    x -- features (n, d, h, w)
    y -- labels (v, n)

    Return :
    percent -- rate of success
    """

    percent = 0
    count, n = y.shape
    cache = forward(parameters, x)
    y_hat = cache['ad' + str(parameters['Ld'])]

    for i, j in zip(y_hat.T, y.T):
        m = np.max(i)
        for k in range(count):
            if i[k] == m:
                i[k] = 1
            else:
                i[k] = 0
        if np.all(i == j):
            percent += 1

    return percent / n


def relu(z):
    """
    Apply the relu function on each element of z

    Take :
    z -- a numpy matrix

    Return :
    a -- a numpy matrix with the same shape as z
    """

    a = np.maximum(z, 0)

    return a


def relu_prime(z):
    """
    Apply the derivative of the relu function on each element of z

    Take :
    z -- a numpy matrix

    Return :
    a -- a numpy matrix with the same shape as z
    """

    a = (z > 0).astype(np.int)

    return a


def tanh(z):
    """
    Apply the tanh function on each element of z

    Take :
    z -- a numpy matrix

    Return :
    a -- a numpy matrix with the same shape as z
    """

    a = np.tanh(z)

    return a


def tanh_prime(z):
    """
    Apply the derivative of the tanh function on each element of z

    Take :
    z -- a numpy matrix

    Return :
    a -- a numpy matrix with the same shape as z
    """

    tanhz = tanh(z)
    a = 1 - tanhz * tanhz

    return a


def sigmoid(z):
    """
    Apply the sigmoid function on each element of z

    Take :
    z -- a numpy matrix

    Return :
    a -- a numpy matrix with the same shape as z
    """

    a = 1 / (1 + np.exp(-z))

    return a


def sigmoid_prime(z):
    """
    Apply the derivative of the sigmoid function on each element of z

    Take :
    z -- a numpy matrix

    Return :
    a -- a numpy matrix with the same shape as z
    """

    sigmoidz = sigmoid(z)
    a = sigmoidz * (1 - sigmoidz)

    return a


def softmax(z):
    """
    Apply the softmax function on each element of z

    Take :
    z -- a numpy matrix

    Return :
    a -- a numpy matrix with the same shape as z
    """

    a = np.divide(np.exp(z), np.sum(np.exp(z), axis=0, keepdims=True))

    return a
