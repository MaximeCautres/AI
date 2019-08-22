""" Function for Deep Neural Network """

import time
import math
import numpy as np
import matplotlib.pyplot as plt

# Small value
epsilon = pow(10, -8)


def generate(parameters, goal, iteration_count, galpha, refresh_rate):
    """
    Generate goal with parameters

    Take :
    parameters -- dictionary fill of parameter (L, w, b, af, vw, sw)
    goal -- the digit we want
    iteration_count -- number of iteration
    galpha -- learning rate for generation
    refresh_rate -- frame rate

    Return :
    good question
    """

    x = np.zeros((784, 1))
    y = np.zeros((10, 1))
    y[goal, 0] = 1

    plt.close()

    for _ in range(iteration_count):
        plt.imshow(x.reshape(28, 28), cmap='gray')
        plt.pause(refresh_rate)

        cache = forward(parameters, x)
        gradients = backward(parameters, y, cache)
        x -= gradients['da0'] * galpha

    plt.show()


def initialize_parameters(layers_dims, activations):
    """
    Initialize parameters

    Take :
    layers_dims -- tuple describe deep neural network topology
    activations -- tuple describe which activation function to use at each layer

    Return :
    parameters -- dictionary fill of parameter (L, w, b, af, vw, sw)
    """

    parameters = {'L': len(layers_dims) - 1, 'layers_dims' : layers_dims}

    for l in range(1, len(layers_dims)):
        # Scale random
        k = math.sqrt(2 / layers_dims[l - 1])

        shape_w = (layers_dims[l], layers_dims[l - 1])
        shape_b = (layers_dims[l], 1)

        parameters['w' + str(l)] = np.random.randn(*shape_w) * k
        parameters['b' + str(l)] = np.zeros(shape_b)

        parameters['vw' + str(l)] = np.zeros(shape_w)
        parameters['vb' + str(l)] = np.zeros(shape_b)

        parameters['sw' + str(l)] = np.zeros(shape_w)
        parameters['sb' + str(l)] = np.zeros(shape_b)

        parameters['af' + str(l)] = activations[l - 1]

    return parameters


def train(parameters, epoch_count, print_rate, mini_batch_size, alpha_zeros,
          ad_rate, beta1, beta2, lambda2, keep_dims, training, testing, take_best):
    """
    Train parameters

    Take :
    parameters -- dictionary fill of parameter (L, w, b, af, vw, sw)
    epoch_count -- number of time we apply gradient descent
    print_rate -- amount of epoch between each cost print
    mini_batch_size -- amount of couple (feature, label) per mini_batch
    alpha_zeros -- amount of learning at each gradient descent
    beta1 -- momentum rate
    beta2 -- RMS'prop rate
    lambda2 -- L2 regularization rate
    keep_dims -- dropout rate
    training -- training set
    testing -- testing set
    take_best -- boolean allow or not to save parameters with the best cost

    Return :
    parameters -- updated dictionary fill of parameter (L, w, b, af, vw, sw)
    performance -- string of performance for this deep neural network
    best_cost -- the best cost during this training
    """

    (train_x, train_y) = training
    (test_x, test_y) = testing

    year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
    print("[{0} / {1} / {2} at {3} : {4}] Beginning of the training...".format(day, month, year, hour, minute))
    begin = time.time()
    costs = []

    best = {'parameters' : parameters, 'cost' : pow(10, 6)}

    for epoch in range(1, epoch_count + 1):
        # Alpha decay
        alpha_current = pow(ad_rate, epoch) * alpha_zeros

        # Mini-batch
        mini_batches = create_mini_batches(train_x, train_y, mini_batch_size)

        # Dropout
        mask = create_mask(parameters, keep_dims)

        for x, y in mini_batches:
            cache = forward(parameters, x, mask)
            gradients = backward(parameters, y, cache, lambda2)
            parameters = update_parameters(parameters, gradients, alpha_current, epoch, beta1, beta2)

        if epoch % print_rate == 0:
            cache = forward(parameters, train_x)
            y_hat = cache['a' + str(parameters['L'])]
            cost = compute_cost(parameters, train_y, y_hat, lambda2)
            costs.append(cost)

            if cost < best['cost']:
                best['cost'] = cost
                best['parameters'] = parameters

            print("Cost after iteration {} : {}".format(epoch, cost))

    delta = time.gmtime(time.time() - begin)
    print("Finished in {0} hour(s) {1} minute(s) {2} second(s).".format(delta.tm_hour, delta.tm_min, delta.tm_sec))

    if take_best:
        parameters = best['parameters']

    performance = evaluate(parameters, train_x, train_y) * 100

    print("Accuracy train : {0} %".format(performance))
    print("Accuracy test : {0} %".format(evaluate(parameters, test_x, test_y) * 100))

    plt.plot(costs)
    plt.ylabel("Cost")
    plt.xlabel("Iteration per " + str(print_rate))
    plt.title("Learning rate : " + str(alpha_zeros))
    plt.show()

    return parameters, str(performance), best['cost']


def forward(parameters, x, mask=None):
    """
    Apply a forward propagation

    Take :
    parameters -- dictionary fill of parameter (L, w, b, af, vw, sw)
    x -- features (n0, m)
    mask -- dropout's mask

    Return :
    cache -- dictionary of results (z1, ..., zL, a1, ..., aL)
    """

    cache = {}
    a = x
    cache['a0'] = a

    for l in range(1, parameters['L'] + 1):
        w = parameters['w' + str(l)]
        b = parameters['b' + str(l)]

        z = np.dot(w, a) + b

        if parameters['af' + str(l)] == 'relu':
            a = relu(z)
        elif parameters['af' + str(l)] == 'tanh':
            a = tanh(z)
        elif parameters['af' + str(l)] == 'sigmoid':
            a = sigmoid(z)
        elif parameters['af' + str(l)] == 'softmax':
            a = softmax(z)

        if mask is not None:
            a *= mask[l]

        cache['z' + str(l)] = z
        cache['a' + str(l)] = a

    return cache


def backward(parameters, y, cache, lambda2=0):
    """
    Apply a backward propagation

    Take :
    parameters -- dictionary fill of parameter (L, w, b, af, vw, sw)
    y -- labels (nL, m)
    cache -- dictionary of results (z1, ..., zL, a1, ..., aL)
    lambda2 -- L2 reguralization rate

    Return :
    gradients -- partial derivative of each parameters with respect to cost (dw1, ..., dwL, db1, ...,dbL)
    """

    m = y.shape[1]
    y_hat = cache['a' + str(parameters['L'])]
    da = np.divide(1 - y, 1 - y_hat) - np.divide(y, y_hat)
    dz = None

    gradients = {}

    for l in reversed(range(1, parameters['L'] + 1)):
        z = cache['z' + str(l)]

        if parameters['af' + str(l)] == 'relu':
            dz = da * relu_prime(z)
        elif parameters['af' + str(l)] == 'tanh':
            dz = da * tanh_prime(z)
        elif parameters['af' + str(l)] == 'sigmoid':
            dz = da * sigmoid_prime(z)
        elif parameters['af' + str(l)] == 'softmax':
            dz = y_hat - y

        a_prev = cache['a' + str(l - 1)]
        w = parameters['w' + str(l)]

        gradients['dw' + str(l)] = (1 / m) * (np.dot(dz, a_prev.T) + lambda2 * w)
        gradients['db' + str(l)] = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        da = np.dot(w.T, dz)
        
        if l == 1:
            gradients['da0'] = da

    return gradients


def update_parameters(parameters, gradients, alpha, t, beta1, beta2):
    """
    Update each parameters with the ADAM method in order to reduce cost

    Take :
    parameters -- dictionary fill of parameter (L, w, b, af, vw, sw)
    gradients -- partial derivative of each parameters with respect to cost (dw1, ..., dwL, db1, ...,dbL)
    alpha -- amount of learning at each gradient descent
    t -- current epoch
    beta1 -- momentum rate
    beta2 -- RMS'prop rate

    Return :
    parameters -- updated dictionary fill of parameter (L, w, b, af, vw, sw)
    """

    for l in range(1, parameters['L'] + 1):
        dw = gradients['dw' + str(l)]
        db = gradients['db' + str(l)]

        # Momentum
        vw = beta1 * parameters['vw' + str(l)] + (1 - beta1) * dw
        vb = beta1 * parameters['vb' + str(l)] + (1 - beta1) * db
        parameters['vw' + str(l)] = vw
        parameters['vb' + str(l)] = vb
        vw_ = np.divide(vw, 1 - np.power(beta1, t))
        vb_ = np.divide(vb, 1 - np.power(beta1, t))

        # RMS'prop
        sw = beta2 * parameters['sw' + str(l)] + (1 - beta2) * dw * dw
        sb = beta2 * parameters['sb' + str(l)] + (1 - beta2) * db * db
        parameters['sw' + str(l)] = sw
        parameters['sb' + str(l)] = sb
        sw_ = np.divide(sw, 1 - np.power(beta2, t))
        sb_ = np.divide(sb, 1 - np.power(beta2, t))

        parameters['w' + str(l)] -= np.divide(alpha * vw_, np.sqrt(sw_ + epsilon))
        parameters['b' + str(l)] -= np.divide(alpha * vb_, np.sqrt(sb_ + epsilon))

    return parameters


def compute_cost(parameters, y, y_hat, lambda2):
    """
    Compute the cost for estimation y_hat of y

    Take :
    parameters -- dictionary fill of parameter (L, w, b, af, vw, sw)
    y -- labels (nL, m)
    y_hat -- estimation (nL, m)
    lambda2 -- L2 regularization rate

    Return :
    cost -- global error (1, 1)
    """

    m = y.shape[1]

    if parameters['af' + str(parameters['L'])] == 'softmax':
        loss = np.multiply(np.log(y_hat), y)
    else:
        loss = np.multiply(np.log(y_hat), y) - np.multiply(np.log(1 - y_hat), 1 - y)

    cost = (1 / m) * ((lambda2 / 2) * frobenius_norm_square(parameters) - np.sum(loss))

    return np.squeeze(cost)


def evaluate(parameters, x, y):
    """
    Evaluate the performance of this network

    Take :
    parameters -- dictionary fill of parameter (L, w, b, af, vw, sw)
    x -- features (n0, m)
    y -- labels (nL, m)

    Return :
    percent -- rate of success
    """

    m = x.shape[1]
    count = 0
    cache = forward(parameters, x)
    y_hat = cache['a' + str(parameters['L'])]

    for i, j in zip(y_hat.T, y.T):
        for k in np.nditer(i, op_flags=['readwrite']):
            if k == np.max(i):
                k[...] = 1
            else:
                k[...] = 0
        if np.all(i == j):
            count += 1

    percent = count / m

    return percent


def guess(parameters, x):
    """
    For x as input, return a guess of what digit is it

    Take :
    parameters -- dictionary fill of parameter (L, w, b, af, vw, sw)
    x -- input (nL, 1)

    Return :
    guess -- digit guess
    """

    x.shape = (784, 1)
    cache = forward(parameters, x)
    y_hat = cache['a' + str(parameters['L'])]
    digit_guess = None

    for i in range(10):
        if y_hat[i][0] == np.max(y_hat):
            digit_guess = i

    return digit_guess


def create_mini_batches(x, y, size):
    """
    Generate an array of couple (features, labels) shuffled

    Take :
    x -- features (n0, m)
    y -- labels (nL, m)
    size -- scalar

    Return :
    couples -- array of couple
    """

    couples = []
    m = x.shape[1]

    permutation = list(np.random.permutation(np.arange(m, dtype=np.int16)))

    shuffled_x = x[:, permutation]
    shuffled_y = y[:, permutation]

    num_complete_mini_batches = int(m / size)

    for k in range(num_complete_mini_batches):
        dx = shuffled_x[:, k * size: (k+1) * size]
        dy = shuffled_y[:, k * size: (k+1) * size]
        couple = (dx, dy)
        couples.append(couple)

    if m % size != 0:
        dx = shuffled_x[:, num_complete_mini_batches * size: m]
        dy = shuffled_y[:, num_complete_mini_batches * size: m]
        couple = (dx, dy)
        couples.append(couple)

    return couples


def create_mask(parameters, keep_dims):
    """
    Return a random mask base on keep_dims

    Take :
    parameters -- dictionary fill of parameter (L, w, b, af, vw, sw)
    keep_dims -- dropout rate

    Return :
    Mask -- dropout's mask
    """

    mask = [None]

    for l in range(1, parameters['L'] + 1):
        node_count = parameters['layers_dims'][l]
        value = (np.random.rand(node_count) < keep_dims[l]).reshape(node_count, 1)
        value = np.divide(value, keep_dims[l])
        mask.append(value)

    return mask


def frobenius_norm_square(parameters):
    """
    Return the squared sum of the Frobenius norm of all the weights matrices

    Take :
    parameters -- dictionary fill of parameter (L, w, b, af, vw, sw)

    Return :
    norm -- a positive real number
    """

    norm = 0

    for l in range(1, parameters['L'] + 1):
        w = parameters['w' + str(l)]
        norm += np.sum(w * w)

    return np.squeeze(norm)


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
