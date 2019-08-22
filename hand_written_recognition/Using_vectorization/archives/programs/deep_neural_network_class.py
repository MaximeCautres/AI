""" The Deep Neural Network class """

import json
import time
import numpy as np
import matplotlib.pyplot as plt

print('Loading...')

" Load data "

data_file = open('data.txt', 'r')
data = json.loads(data_file.read())
data_file.close()

" Condition data "

train_x = np.array(data['train_x'])
train_y = np.array(data['train_y'])

test_x = np.array(data['test_x'])
test_y = np.array(data['test_y'])

print('Ready !')


class DeepNeuralNetwork:

    def __init__(self, layer_dims, activations):
        """
        Initialize DeepNeuralNetwork properties

        Take :
        layer_dims -- dimensions of this deep neural network (n0, ..., nL)
        activation -- the main activation function

        Return :
        nothing
        """

        self.layer_dims = layer_dims
        self.Lp1 = len(layer_dims)

        self.parameters = self.initialize_parameters()
        self.activations = activations

    def initialize_parameters(self):
        """
        Initialize this DeepNeuralNetwork parameters

        Take :
        nothing

        Return :
        parameters -- a dictionary fill of weights and biases (w1, ..., wL, b1, ...,bL)
        """

        parameters = {}

        for l in range(1, self.Lp1):
            k = (2 / self.layer_dims[l - 1]) ** 0.5

            parameters['w' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l - 1]) * k
            parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

        return parameters

    def train(self, generation_count, learning_rate, print_rate, disp_plot=True, x=test_x, y=test_y):
        """
        Train this DeepNeuralNetwork

        Take :
        generation_count -- number of time we apply gradient descent
        learning_rate -- amount of learning at each gradient descent
        print_rate -- amount of generation between each cost print
        disp_plot -- define if we show the graph at the end of the training
        x -- features (n0, m)
        y -- labels (nL, m)

        Return :
        nothing
        """

        year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
        print("[{0} / {1} / {2} at {3} : {4}] Beginning of the training...".format(day, month, year, hour, minute))
        begin = time.time()
        costs = []

        for generation in range(1, generation_count + 1):
            cache = self.forward(x)
            gradients = self.backward(y, cache)
            self.parameters = self.update_parameters(gradients, learning_rate)

            if generation % print_rate == 0:
                cost = cross_entropy(y, cache['a' + str(self.Lp1 - 1)])
                print("Cost after iteration {} : {}".format(generation, np.squeeze(cost)))
                costs.append(cost)

        delta = time.gmtime(time.time() - begin)
        print("Finished in {0} hour(s) {1} minute(s) {2} second(s).".format(delta.tm_hour, delta.tm_min, delta.tm_sec))

        print("Accuracy train : {0} %".format(self.evaluate(test_x, test_y) * 100))
        print("Accuracy test : {0} %".format(self.evaluate(train_x, train_y) * 100))

        if disp_plot:
            plt.plot(costs)
            plt.ylabel("Cost")
            plt.xlabel("Iteration per " + str(print_rate))
            plt.title("Learning rate : " + str(learning_rate))
            plt.show()

    def forward(self, x):
        """
        Apply a forward propagation

        Take :
        x -- features (n0, m)

        Return :
        cache -- dictionary of results (z1, ..., zL, a1, ..., aL)
        """

        cache = {}
        a = x
        cache['a0'] = a

        for l in range(1, self.Lp1):
            w = self.parameters['w' + str(l)]
            b = self.parameters['b' + str(l)]

            z = np.dot(w, a) + b

            if self.activations[l] == 'relu':
                a = relu(z)
            elif self.activations[l] == 'tanh':
                a = tanh(z)
            elif self.activations[l] == 'sigmoid':
                a = sigmoid(z)

            cache['z' + str(l)] = z
            cache['a' + str(l)] = a

        return cache

    def backward(self, y, cache):
        """
        Apply a backward propagation

        Take :
        y -- labels (nL, m)
        cache -- dictionary of results (z1, ..., zL, a1, ..., aL)

        Return :
        gradients -- partial derivative of each parameters with respect to cost (dw1, ..., dwL, db1, ...,dbL)
        """

        m = y.shape[1]
        y_hat = cache['a' + str(self.Lp1 - 1)]
        da = np.divide(1 - y, 1 - y_hat) - np.divide(y, y_hat)

        gradients = {}

        for l in reversed(range(1, self.Lp1)):
            z = cache['z' + str(l)]

            if self.activations[l] == 'relu':
                dz = da * relu_prime(z)
            elif self.activations[l] == 'tanh':
                dz = da * tanh_prime(z)
            elif self.activations[l] == 'sigmoid':
                dz = da * sigmoid_prime(z)

            a_prev = cache['a' + str(l - 1)]

            gradients['dw' + str(l)] = (1 / m) * np.dot(dz, a_prev.T)
            gradients['db' + str(l)] = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            w = self.parameters['w' + str(l)]

            da = np.dot(w.T, dz)

        return gradients

    def update_parameters(self, gradients, learning_rate):
        """
        Update each parameters in order to reduce cost

        Take :
        gradients -- partial derivative of each parameters with respect cost (dw1, ..., dwL, db1, ...,dbL)
        learning_rate -- amount of learning at each gradient descent

        Return :
        parameters -- new parameters fill of weights and biases (w1, ..., wL, b1, ...,bL)
        """

        parameters = {}

        for l in range(1, self.Lp1):
            w = self.parameters['w' + str(l)]
            b = self.parameters['b' + str(l)]

            dw = gradients['dw' + str(l)]
            db = gradients['db' + str(l)]

            parameters['w' + str(l)] = w - learning_rate * dw
            parameters['b' + str(l)] = b - learning_rate * db

        return parameters

    def evaluate(self, x, y):
        """
        Evaluate the performance of this network

        Take :
        x -- features (n0, m)
        y -- labels (nL, m)

        Return :
        percent -- rate of success
        """

        m = x.shape[1]
        count = 0
        cache = self.forward(x)
        y_hat = cache['a' + str(self.Lp1 - 1)]

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

    def guess(self, x):
        """
        For x as input, return a guess of what digit is it

        Take :
        x -- input (nL, 1)

        Return :
        guess -- digit guess
        """

        x.shape = (784, 1)
        cache = self.forward(x)
        y_hat = cache['a' + str(self.Lp1 - 1)]
        guess = None

        for i in range(10):
            if y_hat[i][0] == np.max(y_hat):
                guess = i

        return guess


def cross_entropy(y, y_hat):
    """
    Compute the cost for estimation y_hat of y

    Take :
    y -- labels (nL, m)

    Return :
    cost -- global error (nL, 1)
    """

    m = y.shape[1]

    cost = -(1 / m) * np.sum(np.multiply(np.log(y_hat), y) + np.multiply(np.log(1 - y_hat), (1 - y)))
    cost = np.squeeze(cost)

    return cost

def tanh(z):
    """
    Apply the tanh function at each element of z

    Take :
    z -- a numpy matrix

    Return :
    a -- a numpy matrix with the same shape that z
    """

    a = np.tanh(z)

    return a


def tanh_prime(z):
    """
    Apply the derivative of the tanh function at each element of z

    Take :
    z -- a numpy matrix

    Return :
    a -- a numpy matrix with the same shape that z
    """

    tanhz = tanh(z)
    a = 1 - np.power(tanhz, 2)

    return a


def relu(z):
    """
    Apply the relu function at each element of z

    Take :
    z -- a numpy matrix

    Return :
    a -- a numpy matrix with the same shape that z
    """

    a = np.maximum(z, 0)

    return a


def relu_prime(z):
    """
    Apply the derivative of the relu function at each element of z

    Take :
    z -- a numpy matrix

    Return :
    a -- a numpy matrix with the same shape that z
    """

    a = (z > 0).astype(np.int)

    return a


def sigmoid(z):
    """
    Apply the sigmoid function at each element of z

    Take :
    z -- a numpy matrix

    Return :
    a -- a numpy matrix with the same shape that z
    """

    a = 1 / (1 + np.exp(-z))

    return a


def sigmoid_prime(z):
    """
    Apply the derivative of the sigmoid function at each element of z

    Take :
    z -- a numpy matrix

    Return :
    a -- a numpy matrix with the same shape that z
    """

    sigmoidz = sigmoid(z)
    a = sigmoidz * (1 - sigmoidz)

    return a
