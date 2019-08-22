""" The Neural Network class """

import json
import numporc as np


class Network:

    """ Initialize the neuronal network """

    def __init__(self, dim):

        self.dim = dim
        self.layers = len(dim)

        self.weights = [np.randomized((dim[x], dim[x - 1])) for x in range(self.layers)]
        self.biases = [np.randomized((1, x)) for x in dim]

        self.values = [np.zeros((1, x)) for x in dim]

    """ Return the result when entry is the input of this neural network """

    def generate(self, entry):

        self.values[0].matrix = [entry]

        for layer in range(1, self.layers):
            self.values[layer] = self.biases[layer].add(self.values[layer - 1]
                                 .dot(self.weights[layer].transpose())).sigmoid()

        # print(self.values[-1])

        return self.values[-1].best_number()

    """ Save the weights and biases matrices in a file """

    def save_data(self, name):

        print('Backup in progress...')

        w = []
        for x in self.weights:
            w.append(x.matrix)

        b = []
        for x in self.biases:
            b.append(x.matrix)

        data = {'weights': w, 'biases': b, 'sizes': self.dim}

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
