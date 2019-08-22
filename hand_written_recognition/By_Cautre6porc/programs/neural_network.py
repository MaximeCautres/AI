""" A Neural Network using back-propagation by Cautre_6_PORC """

import time
import json
import random
import numporc as np

" Load training and testing data "

print('Loading data...')

train_file = open('train.txt', 'r')
train = json.loads(train_file.read())
train_file.close()

test_file = open('test.txt', 'r')
test = json.loads(test_file.read())
test_file.close()

print('Loading completed.')


class Network:
    """ Initialize the neuronal network """

    def __init__(self, dim):

        self.dim = dim
        self.layers = len(dim)

        self.hidden = dim[1:-2]

        self.weights = [np.randomized((dim[x], dim[x - 1])) for x in range(self.layers)]
        self.biases = [np.randomized((1, x)) for x in dim]

        self.effect_weight = [np.zeros((dim[x], dim[x - 1])) for x in range(self.layers)]
        self.effect_bias = [np.zeros((1, x)) for x in dim]

        self.z = [np.zeros((1, x)) for x in dim]
        self.a = [np.zeros((1, x)) for x in dim]

        self.best = [0, self.weights, self.biases]

    """ Train the neuronal network """

    def training(self, learning_rate, batch_size, generation_count):

        year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
        print("[{0} / {1} / {2} at {3} : {4}] Beginning of the training...".format(day, month, year, hour, minute))
        begin = time.time()

        for generation in range(generation_count):

            current = time.time()

            random.shuffle(train)
            batches = [train[x:x + batch_size] for x in range(0, len(train), batch_size)]

            for batch in batches:
                self.update(batch, learning_rate)

            success, total, percent = self.success_count()
            delta = time.gmtime(time.time() - current)

            print("Generation {0} done in {1} hour(s) {2} minute(s) {3} second(s) : {4} / {5} -> {6} %"
                  .format(generation + 1, delta.tm_hour, delta.tm_min, delta.tm_sec, success, total, percent))

            self.check_best(percent)

        delta = time.gmtime(time.time() - begin)
        print("Finished in {0} hour(s) {1} minute(s) {2} second(s).".format(delta.tm_hour, delta.tm_min, delta.tm_sec))

        self.save_data()

    """ Update weights and biases """

    def update(self, batch, rate):

        stack_weight = [np.zeros((self.dim[x], self.dim[x - 1])) for x in range(self.layers)]
        stack_bias = [np.zeros((1, x)) for x in self.dim]

        rate /= len(batch)

        for inp, out in batch:
            self.back_propagation(inp, out)

            for layer in range(1, self.layers):
                stack_weight[layer] = stack_weight[layer].add(self.effect_weight[layer])
                stack_bias[layer] = stack_bias[layer].add(self.effect_bias[layer])

        for layer in range(1, self.layers):
            self.weights[layer] = self.weights[layer].add(stack_weight[layer].hadamard(rate))
            self.biases[layer] = self.biases[layer].add(stack_bias[layer].hadamard(rate))

    """ Use back-propagation to get the influences of each weight and bias """

    def back_propagation(self, inp, out):

        self.execute(inp)

        delta = np.zeros([out]).subtract(self.a[-1]).hadamard(self.z[-1].sigmoid_prime())

        self.effect_weight[-1] = delta.transpose().dot(self.a[-2])
        self.effect_bias[-1] = delta

        for layer in range(2, self.layers):
            delta = delta.dot(self.weights[-layer + 1]).hadamard(self.z[-layer].sigmoid_prime())

            self.effect_weight[-layer] = delta.transpose().dot(self.a[-layer - 1])
            self.effect_bias[-layer] = delta

    """ Execute the neuronal network """

    def execute(self, entry):

        self.a[0].matrix = [entry]

        for layer in range(1, self.layers):
            self.z[layer] = self.biases[layer].add(self.a[layer - 1].dot(self.weights[layer].transpose()))
            self.a[layer] = self.z[layer].sigmoid()

    """ Return the number of test inputs for which the neural network outputs the correct result """

    def success_count(self):

        count = 0

        for couple in test:
            self.execute(couple[0])

            if self.a[-1].smooth() == couple[1]:
                count += 1

        return count, len(test), round(count * 100 / len(test), 2)

    """ Return the number of test inputs for which the neural network outputs the correct result base on a data set """

    def compute_fitness(self, data):

        count = 0

        for couple in data:
            self.execute(couple[0])

            if self.a[-1].smooth() == couple[1]:
                count += 1

        return count

    """ Save the best matrices """

    def check_best(self, new):

        if new > self.best[0]:
            self.best[0] = new
            self.best[1] = self.weights
            self.best[2] = self.biases

    """ Return the result when entry is the input of this neural network """

    def generate(self, entry):

        self.a[0].matrix = [entry]

        for layer in range(1, self.layers):
            self.a[layer] = self.biases[layer].add(self.a[layer - 1].dot(self.weights[layer].transpose())).sigmoid()

        return self.a[-1].best_number()

    """ Save the weights and biases matrices in a file """

    def save_data(self):

        print('Backup in progress...')

        w = []
        for x in self.best[1]:
            w.append(x.matrix)

        b = []
        for x in self.best[2]:
            b.append(x.matrix)

        data = {'weights': w, 'biases': b, 'sizes': self.dim}

        name = str(self.best[0]) + '.txt'

        saving_file = open(name, 'w')
        saving_file.write(json.dumps(data))
        saving_file.close()

        print('Finished backup.')

    def __str__(self):
        return 'Neural_Network_object : ' + str(self.dim)


""" Restore a network from a file """


def restore(string):
    print('Restoration in progress...')

    restore_file = open(string, 'r')
    content = json.loads(restore_file.read())
    restore_file.close()

    w = content['weights']
    b = content['biases']
    dimension = content['sizes']

    new = Network(dimension)

    for i in range(new.layers):
        new.weights[i].matrix = w[i]
        new.biases[i].matrix = b[i]

    print('Finished restoration.')

    return new
