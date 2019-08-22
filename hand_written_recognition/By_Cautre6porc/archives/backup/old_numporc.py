""" The numporc library """

import math
import random


class Numporc:

    def __init__(self, row, column):

        self.size = (row, column)
        self.row = row
        self.column = column
        self.matrix = [[0 for _ in range(column)] for _ in range(row)]

    def randomize(self):

        self.matrix = [[(2 * random.random()) - 1 for _ in range(self.column)] for _ in range(self.row)]

    def transpose(self):

        row, column = self.size
        new = Numporc(column, row)

        for x in range(row):
            for y in range(column):
                new.matrix[y][x] = self.matrix[x][y]

        return new

    def subtract(self, other):

        row, column = self.size
        new = Numporc(row, column)

        if isinstance(other, int) or isinstance(other, float):

            new.matrix = [[self.matrix[x][y] - other for y in range(column)] for x in range(row)]

            return new

        elif self.size == other.size:

            new.matrix = [[self.matrix[x][y] - other.matrix[x][y] for y in range(column)] for x in range(row)]

            return new

        else:
            print("Error in the subtract method, the argument doesn't match with this Numporc object.")

    def add(self, other):

        row, column = self.size
        new = Numporc(row, column)

        if isinstance(other, int) or isinstance(other, float):

            new.matrix = [[self.matrix[x][y] + other for y in range(column)] for x in range(row)]

            return new

        elif self.size == other.size:

            new.matrix = [[self.matrix[x][y] + other.matrix[x][y] for y in range(column)] for x in range(row)]

            return new

        else:
            print("Error in the add method, the argument doesn't match with this Numporc object.")

    def hadamard(self, other):

        row, column = self.size
        new = Numporc(row, column)

        if isinstance(other, int) or isinstance(other, float):

            new.matrix = [[self.matrix[x][y] * other for y in range(column)] for x in range(row)]

            return new

        elif self.size == other.size:

            new.matrix = [[self.matrix[x][y] * other.matrix[x][y] for y in range(column)] for x in range(row)]

            return new

        else:
            print("Error in the hadamard method, the argument doesn't match with this Numporc object.")

    def dot(self, other):

        if self.column == other.row:

            new = Numporc(self.row, other.column)

            for x in range(self.row):
                for y in range(other.column):
                    stack = 0
                    for z in range(self.column):
                        stack += self.matrix[x][z] * other.matrix[z][y]
                    new.matrix[x][y] = stack

            return new

        else:
            print("Error in the dot method, the argument doesn't match with this Numporc object.")

    def sigmoid(self):

        row, column = self.size
        new = Numporc(row, column)

        for i in range(row):
            for j in range(column):
                new.matrix[i][j] = 1 / (1 + math.exp(-self.matrix[i][j]))

        return new

    def sigmoid_prime(self):

        row, column = self.size
        new = Numporc(row, column)
        sig = self.sigmoid()
        new.matrix = sig.hadamard(sig.hadamard(-1).add(1)).matrix

        return new

    def smooth(self):

        row, column = self.size
        new = Numporc(row, column)
        best = 0
        best_index = 0

        for i in range(row):
            for j in range(column):
                if self.matrix[i][j] > best:
                    best = self.matrix[i][j]
                    best_index = j
        new.matrix[0][best_index] = 1

        return new

    def best_number(self):

        row, column = self.size
        best = 0
        best_index = 0

        for i in range(row):
            for j in range(column):
                if self.matrix[i][j] > best:
                    best = self.matrix[i][j]
                    best_index = j

        return best_index


def convert(array):

    new = Numporc(1, len(array))
    new.matrix[0] = array

    return new


def create(row, column):

    return Numporc(row, column)
