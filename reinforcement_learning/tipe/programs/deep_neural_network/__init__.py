import numpy as np


def initialize_parameters(topology):
    parameters = {'layers': topology, 'L': len(topology)}
    for l in range(1, parameters['L']):
        cond = l < parameters['L'] - 1
        parameters['af' + str(l)] = 'relu' * cond + 'softmax' * (not cond)
        parameters['w' + str(l)] = np.random.randn(*topology[l-1:l+1][::-1]) * 10**-1
        parameters['b' + str(l)] = np.zeros((topology[l], 1))
    return parameters


def forward(parameters, x, return_cache=False):
    a = x
    cache = {'a0': a}

    for l in range(1, parameters['L']):
        w = parameters['w' + str(l)]
        b = parameters['b' + str(l)]
        af = parameters['af' + str(l)]

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


def backward(parameters, x, y):
    gradients = {}
    n = x.shape[1]
    cache = forward(parameters, x, True)
    y_hat = cache['a' + str(parameters['L']-1)]

    da = np.divide(1 - y, 1 - y_hat) - np.divide(y, y_hat)
    dz = None

    for l in reversed(range(1, parameters['L'])):
        z = cache['z' + str(l)]
        af = parameters['af' + str(l)]

        if af == 'relu':
            dz = da * relu_prime(z)
        elif af == 'softmax':
            dz = y_hat - y

        a_prev = cache['a' + str(l - 1)]
        w = parameters['w' + str(l)]

        gradients['dw' + str(l)] = (1 / n) * np.dot(dz, a_prev.T)
        gradients['db' + str(l)] = (1 / n) * np.sum(dz, axis=1, keepdims=True)

        da = np.dot(w.T, dz)

    return gradients


def update_parameters(parameters, gradients, alpha):
    for l in range(1, parameters['L']):
        parameters['w' + str(l)] -= alpha * gradients['dw' + str(l)]
        parameters['b' + str(l)] -= alpha * gradients['db' + str(l)]
    return parameters


def relu(z):
    return np.maximum(z, 0)


def relu_prime(z):
    return z > 0


def softmax(z):
    return np.divide(np.exp(z), np.sum(np.exp(z), axis=0, keepdims=True))