""" Function for Convolutional Neural Network """

import time
import math
import numpy as np


def initialize_parameters(inputs_dim, layer_count, cnn_topology, cnn_functions, dnn_topology, dnn_functions):
    """
    Initialize parameters

    Take :
    inputs_dim -- tuple, the dimensions of the inputs images
    layer_count -- dictionary, how many layer each network have
    cnn_topology -- dictionary, what topology the cnn have
    cnn_functions -- dictionary, with which functions the cnn work
    dnn_topology -- tuple, what topology the dnn have
    dnn_functions -- tuple, with which functions the dnn work

    Return :
    parameters -- dictionary, all the information needed for the training
    """

    cnn_parameters = {'L': layer_count['cnn']}
    dnn_parameters = {'L': layer_count['dnn']}

    depth = inputs_dim[2]
    current = np.array(inputs_dim[:2])

    for l in range(1, cnn_parameters['L'] + 1):
        cnn_parameters['kc' + str(l)] = test(cnn_topology, 'kc' + str(l), 6)
        cnn_parameters['kd' + str(l)] = test(cnn_topology, 'kd' + str(l), np.array([3, 3]))
        cnn_parameters['pd' + str(l)] = test(cnn_topology, 'pd' + str(l), np.array([2, 2]))
        cnn_parameters['pf' + str(l)] = test(cnn_functions, 'pf' + str(l), 'max')
        cnn_parameters['af' + str(l)] = test(cnn_functions, 'af' + str(l), 'relu')

        count = cnn_parameters['kc' + str(l)]
        (w, h) = cnn_parameters['kd' + str(l)]

        scale = math.sqrt((w * h) / np.prod(current))
        cnn_parameters['k' + str(l)] = np.random.randn(w, h, depth * count) * scale

        depth = count
        current = (current / cnn_parameters['pd' + str(l)]).astype(int)

    for l in range(dnn_parameters['L'] + 1):
        dnn_parameters['nc' + str(l)] = dnn_topology[l]
        dnn_parameters['af' + str(l)] = dnn_functions[l]

        if l > 0:
            scale = math.sqrt(2 / dnn_topology[l-1])
            dnn_parameters['w' + str(l)] = np.random.randn(dnn_topology[l], dnn_topology[l-1]) * scale
            dnn_parameters['b' + str(l)] = np.zeros((dnn_topology[l], 1))

    parameters = {'cnn': cnn_parameters, 'dnn': dnn_parameters}

    return parameters


def test(dictionary, string, default):
    """
    Check if dictionary[string] exist
    In case of not, set it to default

    Take :
    dictionary -- dictionary, the bank of data
    string -- string, the key to index
    default -- the default value

    Return :
    dictionary -- dictionary, the bank of data
    """

    try:
        dictionary[string]
    except KeyError:
        dictionary[string] = default

    return dictionary[string]


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
    (count, n) = y.shape
    cache = forward(parameters, x)
    y_hat = cache['ad' + str(parameters['dnn']['L'])]

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
            gradients = backward(parameters, y, cache)
            parameters = update_parameters(parameters, gradients, alpha)
        cost = compute_cost(parameters, train_x, train_y)
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
    x -- features (w, h, d, n)

    Return :
    cache -- dictionary of results
    """

    cache = {'ac0': x}
    n = x.shape[3]

    cnn_p = parameters['cnn']
    ac = x

    for l in range(1, cnn_p['L'] + 1):
        k = cnn_p['k' + str(l)]
        pd, pf, af = cnn_p['pd' + str(l)], cnn_p['pf' + str(l)], cnn_p['af' + str(l)]

        zc, sa = apply_kernels(ac, k)
        ac, acc = focus(zc, pd, pf, af)

        cache['zc' + str(l)] = zc
        cache['sa' + str(l)] = sa
        cache['acc' + str(l)] = acc
        cache['ac' + str(l)] = ac

    dnn_p = parameters['dnn']
    ad = ac.reshape(-1, n)
    cache['ad0'] = ad

    for l in range(1, dnn_p['L'] + 1):
        w = dnn_p['w' + str(l)]
        b = dnn_p['b' + str(l)]

        zd = np.dot(w, ad) + b

        if dnn_p['af' + str(l)] == 'relu':
            ad = relu(zd)
        elif dnn_p['af' + str(l)] == 'tanh':
            ad = tanh(zd)
        elif dnn_p['af' + str(l)] == 'sigmoid':
            ad = sigmoid(zd)
        elif dnn_p['af' + str(l)] == 'softmax':
            ad = softmax(zd)

        cache['zd' + str(l)] = zd
        cache['ad' + str(l)] = ad

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

    dnn_p = parameters['dnn']
    y_hat = cache['ad' + str(dnn_p['L'])]
    da = np.divide(1 - y, 1 - y_hat) - np.divide(y, y_hat)
    dz = None

    for l in reversed(range(1, dnn_p['L'] + 1)):
        z = cache['zd' + str(l)]

        if dnn_p['af' + str(l)] == 'relu':
            dz = da * relu_prime(z)
        elif dnn_p['af' + str(l)] == 'tanh':
            dz = da * tanh_prime(z)
        elif dnn_p['af' + str(l)] == 'sigmoid':
            dz = da * sigmoid_prime(z)
        elif dnn_p['af' + str(l)] == 'softmax':
            dz = y_hat - y

        a_prev = cache['ad' + str(l - 1)]
        w = dnn_p['w' + str(l)]

        gradients['dw' + str(l)] = (1 / n) * np.dot(dz, a_prev.T)
        gradients['db' + str(l)] = (1 / n) * np.sum(dz, axis=1, keepdims=True)

        da = np.dot(w.T, dz)

    cnn_p = parameters['cnn']
    da = da.reshape(cache['ac' + str(cnn_p['L'])].shape)

    for l in reversed(range(1, cnn_p['L'] + 1)):
        pd = cnn_p['pd' + str(l)]
        zc = cache['zc' + str(l)].shape
        mask = cache['acc' + str(l)]

        dz = rescale(da, pd, zc) * mask

        sa = cache['sa' + str(l)]
        kernels = cnn_p['k' + str(l)]
        ac_prev = cache['ac' + str(l - 1)]

        sdz = sum_k(dz, kernels)
        side = ac_prev.shape[2]
        for i in range(cache['ac' + str(l)].shape[2]):
            sdz[:, :, i*side:(i+1)*side] *= sa
        gradients['dk' + str(l)] = sdz * (1 / n) * (1 / n)

        if l > 1:
            dim = ac_prev.shape
            k = cnn_p['k' + str(l)]
            z, _ = apply_kernels(np.ones(dim), k)
            m = (1 / z.shape[2]) * np.sum(z * dz, axis=2, keepdims=True)
            da = np.repeat(m, dim[2], axis=2)

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

    cnn_p = parameters['cnn']

    for l in range(1, cnn_p['L'] + 1):
        dk = gradients['dk' + str(l)]
        cnn_p['k' + str(l)] -= dk * alpha

    parameters['cnn'] = cnn_p
    dnn_p = parameters['dnn']

    for l in range(1, dnn_p['L'] + 1):
        dw = gradients['dw' + str(l)]
        db = gradients['db' + str(l)]
        dnn_p['w' + str(l)] -= dw * alpha
        dnn_p['b' + str(l)] -= db * alpha

    parameters['dnn'] = dnn_p

    return parameters


def apply_kernels(a, kernels):
    """
    Apply kernels on input

    Take :
    a -- numpy matrix, non linear values of the previous layer (aw, ah, depth, n)
    kernels -- numpy matrix, weights to apply (kw, kh, length)

    Return :
    z -- numpy matrix, linear values of this layer (aw, ah, count, n)
    sa -- numpy matrix (kw, kh, depth)
    """

    (aw, ah, depth, n) = a.shape
    (kw, kh, length) = kernels.shape

    count = int(length / depth)
    z = np.zeros((aw, ah, count, n))
    sa = np.zeros((kw, kh, depth))

    nw, nh = aw + kw - 1, ah + kh - 1
    padded_a = np.zeros((nw, nh, depth, n))

    ow, oh = int((kw - 1) * 0.5), int((kh - 1) * 0.5)
    padded_a[ow:ow+aw, oh:oh+ah] = a

    for h in range(ah):
        for w in range(aw):
            chunk = padded_a[w:w + kw, h:h + kh]
            sa += np.sum(chunk, axis=3)
            for d in range(count):
                for l in range(n):
                    z[w, h, d, l] = np.sum(chunk[:, :, :, l] * kernels[:, :, d*depth:(d+1)*depth])

    return z, sa


def focus(z, kernel_dimension, pooling_function, activation_function):
    """
    Pool input with pooling_dimension by applying method
    And pass the result into an activation function

    Take :
    z -- numpy matrix, linear values of this layer (zw, zh, count, n)
    kernel_dimension -- tuple, width and height of the kernel (kw, kh)
    pooling_function -- string, which function is applied to pool
    activation_function -- string, which function is applied to activate

    Return :
    a -- numpy matrix, non linear values of this layer (aw, ah, count, n)
    acc -- numpy matrix (zw, zh, count, n)
    """

    (zw, zh, count, n) = z.shape
    (kw, kh) = kernel_dimension
    cell_count = kw * kh

    aw, ah = int(zw / kw), int(zh / kh)
    p = None
    acc = None

    if activation_function == 'relu':
        p = relu(z)
        acc = relu_prime(z)
    elif activation_function == 'tanh':
        p = tanh(z)
        acc = tanh_prime(z)
    elif activation_function == 'sigmoid':
        p = sigmoid(z)
        acc = sigmoid_prime(z)

    mb = np.zeros(z.shape).astype(float)
    a = np.zeros((aw, ah, count, n))

    for d in range(count):
        for h in range(ah):
            for w in range(aw):
                iw, ih = w*kw, h*kh
                chunk = p[iw:iw+kw, ih:ih+kh, d]

                if pooling_function == 'mean':
                    a[w, h, d] = np.sum(chunk, axis=(0, 1)) / cell_count
                    acc = acc / cell_count
                elif pooling_function == 'max':
                    greater = np.max(chunk, axis=(0, 1))
                    a[w, h, d] = greater
                    mb[iw:iw+kw, ih:ih+kh, d] = (greater == chunk)
                    acc = acc * mb
                elif pooling_function == 'sum':
                    a[w, h, d] = np.sum(chunk, axis=(0, 1))

    return a, acc


def rescale(m, pd, zc):
    """
    Rescale m into zd with kd

    Take :
    m -- numpy matrix (mw, mh, d, n)
    pd -- (kw, kh)
    zc -- (zw, zh, zd, n)

    Return :
    new -- (zw, zh, zd, n)
    """

    (kw, kh) = pd
    (mw, mh, _, _) = m.shape
    new = np.zeros(zc)

    for w in range(mw):
        for h in range(mh):
            new[w*kw:(w+1)*kw, h*kh:(h+1)*kh] = np.ones((kw, kh, 1, 1)) * m[w, h]

    return new


def sum_k(dz, kernels):
    """
    Apply kernels on input

    Take :
    dz -- numpy matrix, gradients of the z layer (dzw, dzh, count, n)
    kernels -- numpy matrix, weights to apply (kw, kh, length)

    Return :
    sdz -- numpy matrix (kw, kh, length)
    """

    (dzw, dzh, count, n) = dz.shape
    (kw, kh, length) = kernels.shape
    depth = int(length / count)
    sdz = np.zeros((kw, kh, length))

    nw, nh = dzw + kw - 1, dzh + kh - 1
    padded_dz = np.zeros((nw, nh, count, n))

    ow, oh = int((kw - 1) * 0.5), int((kh - 1) * 0.5)
    padded_dz[ow:ow + dzw, oh:oh + dzh] = dz

    for h in range(dzh):
        for w in range(dzw):
            chunk = padded_dz[w:w + kw, h:h + kh]
            for l in range(length):
                sdz[:, :, l] += np.sum(chunk[:, :, l // depth], axis=2)

    return sdz


def compute_cost(parameters, x, y):
    """
    Compute the cost for estimation y_hat of y

    Take :
    parameters -- dictionary containing all the information about the whole network
    x -- features (w, h, d, n)
    y -- labels (v, n)

    Return :
    cost -- global error (1, 1)
    """

    n = y.shape[1]
    dnn_p = parameters['dnn']

    cache = forward(parameters, x)
    y_hat = cache['ad' + str(dnn_p['L'])]

    if dnn_p['af' + str(dnn_p['L'])] == 'softmax':
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

    shuffled_x = x[:, :, :, permutation]
    shuffled_y = y[:, permutation]

    num_complete_mini_batches = int(n / size)

    for k in range(num_complete_mini_batches):
        dx = shuffled_x[:, :, :, k * size: (k+1) * size]
        dy = shuffled_y[:, k * size: (k+1) * size]
        couple = (dx, dy)
        couples.append(couple)

    if n % size != 0:
        dx = shuffled_x[:, :, :, num_complete_mini_batches * size: n]
        dy = shuffled_y[:, num_complete_mini_batches * size: n]
        couple = (dx, dy)
        couples.append(couple)

    return couples


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
