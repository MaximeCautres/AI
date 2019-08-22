""" Function for Encoder Decoder Network """

import time
import math
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


def initialize_parameters(encoder, decoder):
    """
    Initialize parameters

    Take :
    encoder -- topology of the encoder
    decoder -- topology of the decoder

    Return :
    parameters -- dictionary containing all the information about the whole network
    """

    if encoder[-1] != decoder[0]:
        raise Exception('invalid bottle neck')

    parameters = {'eL': len(encoder), 'dL': len(decoder),
                  'ec0': encoder[0], 'dc0': decoder[0]}

    for l in range(1, parameters['eL'] + 1):
        p, c = encoder[l-1], encoder[l]
        parameters = set_p(parameters, (c, p), (c, 1), 'e', l)
        parameters['ec' + str(l)] = c

    for l in range(1, parameters['dL'] + 1):
        p, c = decoder[l-1], decoder[l]
        parameters = set_p(parameters, (c, p), (c, 1), 'd', l)
        parameters['dc' + str(l)] = c

    return parameters


def set_p(parameters, shape_w, shape_b, e_or_d, l):
    """
    Set trivial parameters

    Take :
    parameters -- dictionary containing all the information about the whole network
    shape_w -- tuple, the shape of weights
    shape_b -- tuple, the shape of biases
    e_or_d -- encoder or decoder
    l -- the current layer

    Return :
    parameters -- dictionary containing all the information about the whole network
    """

    parameters[e_or_d + 'w' + str(l)] = np.random.randn(*shape_w) * 10**-2
    parameters[e_or_d + 'tw' + str(l)] = np.zeros(shape_w)
    parameters[e_or_d + 'qw' + str(l)] = np.zeros(shape_w)

    parameters[e_or_d + 'b' + str(l)] = np.zeros(shape_b)
    parameters[e_or_d + 'tb' + str(l)] = np.zeros(shape_b)
    parameters[e_or_d + 'qb' + str(l)] = np.zeros(shape_b)

    return parameters


def train(parameters, epoch_count, mini_batch_size, alpha, training, testing, save_type):
    """
    Train parameters

    Take :
    parameters -- dictionary containing all the information about the whole network
    epoch_count -- number of time we apply gradient descent
    mini_batch_size -- amount of couple (feature, label) per mini_batch
    alpha -- learning rate
    training -- training set
    testing -- testing set
    save_type -- 'train' or 'test', which performance matter

    Return :
    parameters -- dictionary containing all the information about the whole network
    """

    train_x, train_y = training
    test_x, test_y = testing

    costs = {'train': [], 'test': []}
    best = {'parameters': parameters, 'cost': pow(10, 6)}

    year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
    print("[{0} / {1} / {2} at {3} : {4}] Beginning of the training...".format(day, month, year, hour, minute))
    begin = time.time()

    for epoch in range(epoch_count):
        mini_batches = create_mini_batches(train_x, train_y, mini_batch_size)
        for x, y in mini_batches:
            for net in ['e', 'd']:
                cache = forward(parameters, x, net)
                gradients = backward(parameters, y, cache)
                parameters = update_parameters(parameters, gradients, alpha)

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

    plt.plot(costs['train'], color='blue')
    plt.plot(costs['test'], color='red')
    plt.ylabel("Cost")
    plt.xlabel("Iteration")
    plt.show()

    return parameters, str(costs[save_type][-1]).replace('.', '_'), best['cost']


def forward(parameters, x, e_or_d):
    """
    Apply a forward propagation in order to compute cache

    Take :
    parameters -- dictionary containing all the information about the whole network
    x -- features (w, h, d, n)
    e_or_d -- encoder or decoder

    Return :
    cache -- dictionary of results
    """

    n = x.shape[3]
    a = x.reshape(-1, n)
    cache = {e_or_d + 'a0': a}

    for l in range(1, parameters[e_or_d + 'L'] + 1):
        w = parameters[e_or_d + 'w' + str(l)]
        b = parameters[e_or_d + 'b' + str(l)]

        z = np.dot(w, a) + b
        cache[e_or_d + 'z' + str(l)] = z

        a = relu(z)
        cache[e_or_d + 'a' + str(l)] = a

    return cache


def backward(parameters, y, cache):
    """
    Apply a backward propagation in order to compute gradients

    Take :
    parameters -- dictionary containing all the information about the whole network
    y -- labels (w, h, d, n)
    cache -- dictionary of results

    Return :
    gradients -- partial derivative of each parameters with respect to cost
    """

    gradients = {}
    n = y.shape[3]

    y_hat = cache['a' + str(parameters['Lc'])]
    da = y_hat - y

    for l in reversed(range(1, parameters['Lc'] + 1)):
        z = cache['z' + str(l)]
        dz = da * relu_prime(z)

        a_p = cache['a' + str(l - 1)]
        w = parameters['w' + str(l)]
        gradients['dw' + str(l)] = (1 / n) * np.dot(dz, a_p.T)

        da = np.dot(w.T, dz)

    gradients['dx0'] = da

    return gradients


def update_parameters(parameters, gradients, alpha):
    """
    Apply shit

    Take :
    parameters -- dictionary containing all the information about the whole network
    gradients -- partial derivative of each parameters with respect to cost
    alpha -- learning rate

    Return :
    parameters -- dictionary containing all the information about the whole network
    """

    for net in ['e', 'd']:
        for l in range(1, parameters[net + 'L'] + 1):
            for v in ['w', 'b']:
                parameters[net + v + str(l)] -= alpha * gradients['d' + net + v + str(l)]

    return parameters


def predict(parameters, x_o):
    """
    Generate a prediction

    Take :
    parameters -- dictionary containing all the information about the whole network
    x_o -- features (w, h, d, n)

    Return :
    a -- prediction (w, h, d, n)
    """

    n = x_o.shape[3]
    a = x_o.reshape(-1, n)

    for l in range(1, parameters['L'] + 1):
        w = parameters['w' + str(l)]
        z = np.dot(w, a)
        a = relu(z)

    return a


def create_mini_batches(x, y, size):
    """
    Generate an array of couple (features, labels) shuffled

    Take :
    x -- features (w, h, d, n)
    y -- labels (w, h, d, n)
    size -- number of couple per mini_batch

    Return :
    couples -- array of couple
    """

    couples = []
    n = x.shape[3]

    permutation = list(np.random.permutation(np.arange(n, dtype=np.int16)))

    shuffled_x = x[:, :, :, permutation]
    shuffled_y = y[:, :, :, permutation]

    num_complete_mini_batches = int(n / size)

    for k in range(num_complete_mini_batches):
        dx = shuffled_x[:, :, :, k * size: (k+1) * size]
        dy = shuffled_y[:, :, :, k * size: (k+1) * size]
        couples.append((dx, dy))

    if n % size != 0:
        dx = shuffled_x[:, :, :, num_complete_mini_batches * size: n]
        dy = shuffled_y[:, :, :, num_complete_mini_batches * size: n]
        couples.append((dx, dy))

    return couples


def compute_cost(parameters, x, y):
    """
    Compute the cost for estimation y_hat of y

    Take :
    parameters -- dictionary containing all the information about the whole network
    x -- features (w, h, d, n)
    y -- labels (w, h, d, n)

    Return :
    cost -- global error
    """

    return


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
