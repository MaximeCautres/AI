""" The numporc library """

import math
import random


class Numporc:

    """ Initialize a new Numporc object """

    def __init__(self, data):

        if isinstance(data, list) and isinstance(data[0], list):
            self.row, self.column = len(data), len(data[0])
            self.size = (self.row, self.column)
            self.matrix = data
            
        elif isinstance(data, tuple) and len(data) == 2:
            self.size = data
            self.row, self.column = data
            self.matrix = [[0 for _ in range(self.column)] for _ in range(self.row)]

        else:
            raise TypeError("This Numporc object cannot be created.")

    """ Randomize a Numporc object without returning anything """

    def randomize(self):

        self.matrix = [[random.gauss(0, 1) for _ in range(self.column)] for _ in range(self.row)]

        return self

    """ Return the transposed version of this Numporc object """

    def transpose(self):

        new = []

        for y in range(self.column):
            new.append([])
            for x in range(self.row):
                new[y].append(self.matrix[x][y])

        return Numporc(new)

    """ Return a Numporc object which is [this, other] """

    def concatenate(self, other):

        row, column = self.size

        if isinstance(other, Numporc):
            new = [[self.matrix[x][y] * 10 ** int(math.log(other.matrix[x][y], 10) + 1) + other.matrix[x][y] for y in range(column)] for x in range(row)]

        elif isinstance(other, int) or isinstance(other, float):
            new = [[self.matrix[x][y] * 10 ** int(math.log(other, 10) + 1) + other for y in range(column)] for x in range(row)]

        else:
            raise TypeError("Error in the concatenate method, the argument doesn't match with this Numporc object.")

        return Numporc(new)

    """ Return a Numporc object which is this - other """

    def subtract(self, other):

        row, column = self.size

        if isinstance(other, Numporc):
            new = [[self.matrix[x][y] - other.matrix[x][y] for y in range(column)] for x in range(row)]

        elif isinstance(other, int) or isinstance(other, float):
            new = [[self.matrix[x][y] - other for y in range(column)] for x in range(row)]
            
        else:
            raise TypeError("Error in the subtract method, the argument doesn't match with this Numporc object.")

        return Numporc(new)

    """ Return a Numporc object which is this + other """

    def add(self, other):

        row, column = self.size

        if isinstance(other, Numporc):
            new = [[self.matrix[x][y] + other.matrix[x][y] for y in range(column)] for x in range(row)]

        elif isinstance(other, int) or isinstance(other, float):
            new = [[self.matrix[x][y] + other for y in range(column)] for x in range(row)]

        else:
            raise TypeError("Error in the add method, the argument doesn't match with this Numporc object.")

        return Numporc(new)

    """ Return a Numporc object which is this * other """

    def hadamard(self, other):

        row, column = self.size

        if isinstance(other, Numporc):
            new = [[self.matrix[x][y] * other.matrix[x][y] for y in range(column)] for x in range(row)]

        elif isinstance(other, int) or isinstance(other, float):
            new = [[self.matrix[x][y] * other for y in range(column)] for x in range(row)]

        else:
            raise TypeError("Error in the hadamard method, the argument doesn't match with this Numporc object.")

        return Numporc(new)

    """ Return a Numporc object which is this . other """

    def dot(self, other):

        if isinstance(other, Numporc) and self.column == other.row:

            new = []

            for x in range(self.row):
                new.append([])
                for y in range(other.column):
                    stack = 0
                    for z in range(self.column):
                        stack += self.matrix[x][z] * other.matrix[z][y]
                    new[x].append(stack)

            return Numporc(new)

        else:

            raise TypeError("Error in the dot method, the argument doesn't match with this Numporc object.")

    """ Return the sigmoid version of this Numporc object """

    def sigmoid(self):

        new = []

        for i in range(self.row):
            new.append([])
            for j in range(self.column):
                new[i].append(1 / (1 + math.exp(-self.matrix[i][j])))

        return Numporc(new)

    """ Return the sigmoid prime version of this Numporc object """

    def sigmoid_prime(self):

        new = []
        sig = self.sigmoid()

        for i in range(self.row):
            new.append([])
            for j in range(self.column):
                x = sig.matrix[i][j]
                new[i].append(x * (1 - x))

        return Numporc(new)

    """ Return the tanh version of this Numporc object """

    def tan_hyper(self):

        new = []

        for i in range(self.row):
            new.append([])
            for j in range(self.column):
                new[i].append(math.tanh(self.matrix[i][j]))

        return Numporc(new)

    """ Return the tanh prime version of this Numporc object """

    def tan_hyper_prime(self):

        new = []

        for i in range(self.row):
            new.append([])
            for j in range(self.column):
                new[i].append(1 - math.tanh(self.matrix[i][j]) ** 2)

        return Numporc(new)

    """ Return the smoothed version of this Numporc object """

    def smooth(self):

        new = []
        best = 0
        best_index = 0

        for j in range(self.column):
            new.append(0)
            if self.matrix[0][j] > best:
                best = self.matrix[0][j]
                best_index = j

        new[best_index] = 1

        return new

    """ Return the best number of this Numporc object """

    def best_number(self):

        best = 0
        best_index = 0

        for i in range(self.column):
            if self.matrix[0][i] > best:
                best = self.matrix[0][i]
                best_index = i

        return best_index

    def __str__(self):

        return 'Numporc_object : ' + str(self.matrix)


""" Return a Numporc object filled of random values base on data """


def randomized(data):

    new = Numporc(data)
    return new.randomize()


""" Return a Numporc object filled of zeros base on data """


def zeros(data):

    return Numporc(data)
