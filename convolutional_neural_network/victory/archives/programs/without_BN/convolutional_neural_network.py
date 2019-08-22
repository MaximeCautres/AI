""" Function for Convolutional Neural Network """

import time
import copy
import numpy as np
import matplotlib.pyplot as plt

# Constants
epsilon = pow(10, -6)
parameter_topology = {'Lc': ['wc', 'bc'], 'Ld': ['wd', 'bd']}
# labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
labels = ['beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium_fish', 'flatfish',
          'ray', 'shark', 'trout', 'sea', 'crab', 'lobster', 'crocodile', 'ship']


def initialize_parameters(cnn_topology, dnn_topology, current):
    """
    Initialize parameters

    Take :
    cnn_topology -- dictionary, what topology the cnn have
    dnn_topology -- dictionary, what topology the dnn have
    current -- numpy array containing the shape of the image

    Return :
    parameters -- dictionary containing all the information about the whole network
    """

    parameters = {**cnn_topology, **dnn_topology, 'kc0': current[2]}

    for l in range(1, parameters['Lc'] + 1):
        count, k = parameters['kc' + str(l)], 1
        w, h = parameters['kd' + str(l)]
        d = parameters['kc' + str(l - 1)]
        parameters = set_p(parameters, (count, w, h, d, 1), (1, 1, count, 1), k, 'c', l)

        lx, sx, dx = [0], parameters['sc' + str(l)][0], current[0]
        while lx[-1] < dx-sx-w:
            lx.append(lx[-1]+sx)
        lx.append(dx-w)

        ly, sy, dy = [0], parameters['sc' + str(l)][1], current[1]
        while ly[-1] < dy - sy - h:
            ly.append(ly[-1] + sy)
        ly.append(dy - h)

        current = (len(lx), len(ly), count)
        parameters['rc' + str(l)] = (lx, ly)

    parameters['nc0'] = np.prod(current)
    for l in range(1, parameters['Ld'] + 1):
        p, c, k = parameters['nc' + str(l-1)], parameters['nc' + str(l)], 10**-2
        parameters = set_p(parameters, (c, p), (c, 1), k, 'd', l)

    return parameters


def set_p(parameters, shape_w, shape_b, k, s, l):
    """
    Set trivial parameters

    Take :
    parameters -- dictionary containing all the information about the whole network
    shape_w -- tuple, the shape of weights
    shape_b -- tuple, the shape of biases
    k -- scale random
    s -- 'c' or 'd', what network it is
    l -- the current layer

    Return :
    parameters -- dictionary containing all the information about the whole network
    """
    parameters['w' + s + str(l)] = np.random.randn(*shape_w) * k
    parameters['b' + s + str(l)] = np.zeros(shape_b)

    parameters['gw' + s + str(l)] = np.zeros(shape_w)
    parameters['gb' + s + str(l)] = np.zeros(shape_b)

    parameters['xw' + s + str(l)] = np.zeros(shape_w)
    parameters['xb' + s + str(l)] = np.zeros(shape_b)

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

    year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
    print("[{0} / {1} / {2} at {3} : {4}] Beginning of the training...".format(day, month, year, hour, minute))
    begin = time.time()

    costs = {'train': [], 'test': []}
    best = {'parameters': parameters, 'cost': pow(10, 6)}

    for epoch in range(epoch_count):
        mini_batches = create_mini_batches(train_x, train_y, mini_batch_size)
        masks = create_masks(parameters)
        for x, y in mini_batches:
            cache = forward(parameters, x, masks)
            gradients = backward(parameters, y, cache, lambda2C, lambda2D)
            parameters = update_parameters(parameters, gradients, optimizer, alpha, beta, gamma, rho)

        train_cost = compute_cost(parameters, train_x, train_y)
        costs['train'].append(train_cost)

        test_cost = compute_cost(parameters, test_x, test_y)
        costs['test'].append(test_cost)

        cost = costs[save_type][-1]
        if cost < best['cost']:
            best['cost'] = cost
            best['parameters'] = copy.deepcopy(parameters)

        print("[Iteration {0}] : train ~ {1} and test ~ {2}".format(epoch+1, train_cost, test_cost))

    delta = time.gmtime(time.time() - begin)
    print("Finished in {0} hour(s) {1} minute(s) {2} second(s).".format(delta.tm_hour, delta.tm_min, delta.tm_sec))

    parameters = best['parameters']

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
    a = x
    cache = {'ac0': a}

    for l in range(1, parameters['Lc'] + 1):
        w = parameters['wc' + str(l)]
        b = parameters['bc' + str(l)]
        rc = parameters['rc' + str(l)]
        af = parameters['afc' + str(l)]

        z = convolve(a, w, b, rc)

        if af == 'relu':
            a = relu(z)
        elif af == 'tanh':
            a = tanh(z)
        elif af == 'sigmoid':
            a = sigmoid(z)

        cache['zc' + str(l)] = z
        cache['ac' + str(l)] = a

    a = a.reshape(-1, n)
    cache['ad0'] = a

    if masks is not None:
        a *= masks[0]

    for l in range(1, parameters['Ld'] + 1):
        w = parameters['wd' + str(l)]
        b = parameters['bd' + str(l)]
        af = parameters['afd' + str(l)]

        z = np.dot(w, a) + b

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

        cache['zd' + str(l)] = z
        cache['ad' + str(l)] = a

    return cache


def backward(parameters, y, cache, lambda2C=0, lambda2D=0):
    """
    Apply a backward propagation in order to compute gradients

    Take :
    parameters -- dictionary containing all the information about the whole network
    y -- labels (v, n)
    cache -- dictionary of results
    lambda2 -- L2 regularization rate

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
        af = parameters['afd' + str(l)]

        if af == 'relu':
            dz = da * relu_prime(z)
        elif af == 'softmax':
            dz = y_hat - y
        elif af == 'tanh':
            dz = da * tanh_prime(z)
        elif af == 'sigmoid':
            dz = da * sigmoid_prime(z)

        a_p = cache['ad' + str(l - 1)]
        w = parameters['wd' + str(l)]
        b = parameters['bd' + str(l)]

        gradients['dwd' + str(l)] = (1 / n) * (np.dot(dz, a_p.T) + lambda2D * w ** 2)
        gradients['dbd' + str(l)] = (1 / n) * (np.sum(dz, axis=1, keepdims=True) + lambda2D * b ** 2)

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

        a_p = cache['ac' + str(l - 1)]
        w = parameters['wc' + str(l)]
        b = parameters['bc' + str(l)]
        rc = parameters['rc' + str(l)]

        da, dw, db = deconvolve(dz, w, a_p, rc)

        gradients['dwc' + str(l)] = (1 / n) * (dw + lambda2C * w ** 2)
        gradients['dbc' + str(l)] = (1 / n) * (db + lambda2C * b ** 2)

    gradients['dac0'] = da

    return gradients


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

    if optimizer == 'adadelta':
        parameters = apply_adadelta(parameters, gradients, rho)
    elif optimizer == 'rmsprop':
        parameters = apply_rmsprop(parameters, gradients, alpha, gamma)
    elif optimizer == 'momentum':
        parameters = apply_momentum(parameters, gradients, alpha, beta)
    elif optimizer == 'shit':
        parameters = apply_shit(parameters, gradients, alpha)

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

    for k, v in parameter_topology.items():
        for l in range(1, parameters[k] + 1):
            for s in v:
                d = gradients['d' + s + str(l)]
                g = parameters['g' + s + str(l)]
                x = parameters['x' + s + str(l)]

                g = rho * g + (1 - rho) * d * d
                c = np.divide(rms(x), rms(g)) * d
                x = rho * x + (1 - rho) * c * c

                parameters[s + str(l)] -= c
                parameters['g' + s + str(l)] = g
                parameters['x' + s + str(l)] = x

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

    for k, v in parameter_topology.items():
        for l in range(1, parameters[k] + 1):
            for s in v:
                d = gradients['d' + s + str(l)]
                x = gamma * parameters['x' + s + str(l)] + (1 - gamma) * d * d

                parameters[s + str(l)] -= alpha * np.divide(d, rms(x))
                parameters['x' + s + str(l)] = x

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

    for k, v in parameter_topology.items():
        for l in range(1, parameters[k] + 1):
            for s in v:
                d = gradients['d' + s + str(l)]
                x = beta * parameters['x' + s + str(l)] + (1 - beta) * d

                parameters[s + str(l)] -= alpha * x
                parameters['x' + s + str(l)] = x

    return parameters


def apply_shit(parameters, gradients, alpha):
    """
    Apply shit

    Take :
    parameters -- dictionary containing all the information about the whole network
    gradients -- partial derivative of each parameters with respect to cost
    beta -- Momentum rate

    Return :
    parameters -- dictionary containing all the information about the whole network
    """

    for k, v in parameter_topology.items():
        for l in range(1, parameters[k] + 1):
            for s in v:
                parameters[s + str(l)] -= alpha * gradients['d' + s + str(l)]

    return parameters


def convolve(A, W, b, rc):
    """
    Apply weights and biases on A

    Take :
    A -- numpy matrix, non linear values of the previous layer (w_A, h_A, d, n)
    W -- numpy matrix, weights to apply (count, w_W, h_W, d, 1)
    b -- numpy matrix, biases to apply (1, 1, count, 1)
    rc -- tuple, range of values of w and h

    Return :
    Z -- numpy matrix, linear values of the current layer (w_Z, h_Z, count, n)
    """

    lx, ly = rc
    tx, ty = len(lx), len(ly)
    count, w_W, h_W, _, _ = W.shape
    Z = np.zeros((tx, ty, count, A.shape[3])) + b

    for k in range(count):
        for x in range(tx):
            for y in range(ty):
                w, h = lx[x], ly[y]
                Z[x, y, k] += np.sum(A[w:w+w_W, h:h+h_W] * W[k], axis=(0,1,2))

    return Z


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
    db = np.sum(dZ, axis=(0, 1, 3), keepdims=True)

    for k in range(count):
        for x in range(w_dZ):
            for y in range(h_dZ):
                w, h, dZ_n = lx[x], ly[y], dZ[x, y, k].reshape(1, 1, 1, n)
                dA[w:w+w_W, h:h+h_W] += W[k] * dZ_n
                dW[k] += np.sum(A_p[w:w+w_W, h:h+h_W] * dZ_n, axis=3, keepdims=True)

    return dA, dW, db


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


def generate(parameters, inputs_dimensions, goal, iteration_count, galpha, refresh_rate):
    """
    Generate goal with parameters

    Take :
    parameters -- dictionary containing all the information about the whole network
    inputs_dimensions -- shape of the input image
    goal -- the digit we want
    iteration_count -- number of iteration
    galpha -- learning rate for generation
    refresh_rate -- frame rate

    Return :
    good question
    """

    x = np.ones(inputs_dimensions + (1, ))
    y = np.zeros((15, 1))
    y[goal, 0] = 1

    plt.close()
    plt.title(labels[goal])

    for _ in range(iteration_count):
        plt.imshow(x.reshape(inputs_dimensions))
        plt.pause(refresh_rate)

        cache = forward(parameters, x)
        gradients = backward(parameters, y, cache)
        x = np.minimum(np.maximum(x - gradients['dac0'] * galpha, 0), 1)

    plt.show()


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
