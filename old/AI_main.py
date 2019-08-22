import math
import random

# Create variables

bits = 64

inputs = [[0 for y in range(bits)] for x in range(bits)]
for i in range(bits):  # All the inputs
    inputs[i][i] = 1

for i in range(len(inputs)):  # Add bias
  inputs[i] += [1]

outputs = [[0 for y in range(int(math.log(bits, 2)))] for x in range(bits)]
index = 0
for i in range(2):  # All the outputs
    for j in range(2):
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    for n in range(2):
                        outputs[index] = [i, j, k, l, m, n]
                        index += 1

parameters = [len(inputs[0]), 10, len(outputs[0])]  # Length of each line of neuron

generationCount = 100  # Set the number of time the neuronal network have to be activated
step = 10 ** -2  # Set the step rate
learningRate = 10  # Set the learning rate

""" Set of weight [line][neuron][previous] """
weights = [[[2 * random.random() - 1 for z in range(parameters[x - 1])]
            for y in range(parameters[x])] for x in range(len(parameters))]


# Define functions

""" Return a matrix of rates for this entry """


def calculus(entry):
    rates = [[[[0 for w in range(parameters[y - 1])]
              for z in range(parameters[y])] for y in range(len(parameters))] for x in range(1)]
    for line in range(1, len(parameters)):
        for neuron in range(parameters[line]):
            for previous in range(parameters[line - 1]):
                weight = weights[line][neuron][previous]
                weights[line][neuron][previous] = weight - step
                low = calculError(entry)
                weights[line][neuron][previous] = weight + step
                high = calculError(entry)
                weights[line][neuron][previous] = weight
                rates[0][line][neuron][previous] = (low - high) / (2 * step)
    return rates


""" Calculate the error for this entry """


def calculError(entry):
    globalError = 0
    gen = execute(entry)
    for i in range(parameters[-1]):
        globalError += (gen[i] - outputs[entry][i]) ** 2
    return globalError


""" Execute the neuronal network """


def execute(entry):
    values = [[1 for y in range(parameters[x])] for x in range(len(parameters))]
    for i in range(parameters[0]):
        values[0][i] = inputs[entry][i]

    for line in range(1, len(parameters)):
        for neuron in range(parameters[line]):
            stack = 0
            for previous in range(parameters[line - 1]):
                stack += values[line - 1][previous] * weights[line][neuron][previous]
            values[line][neuron] = sigmoid(stack)
    return values[len(parameters) - 1]


""" Apply the sigmoid function """


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


""" Calculate the average of each rate and return a new set of weights base on it """


def generate(entry):
    average = ave(entry)
    for line in range(1, len(parameters)):
        for neuron in range(parameters[line]):
            for weight in range(parameters[line - 1]):
                weights[line][neuron][weight] += average[line][neuron][weight] * learningRate


def ave(entry):
    average = [[[0 for z in range(len(entry[0][x][y]))]
              for y in range(len(entry[0][x]))] for x in range(len(entry[0]))]
    for matrix in range(len(entry)):
        for line in range(len(entry[matrix])):
            for neuron in range(len(entry[matrix][line])):
                for rate in range(len(entry[matrix][line][neuron])):
                    average[line][neuron][rate] += entry[matrix][line][neuron][rate] / len(entry)
    return average


""" Return a the smoothing version of this array """


def smoothing(array):
    new = [0 for x in range(len(array))]
    for i in range(len(array)):
        if array[i] < 0.5:
            new[i] = 0
        else:
            new[i] = 1
    return new


""" Return if this prediction match with the expected output """


def match(predict, index):
    expect = outputs[index]
    isMatching = True
    for i in range(len(predict)):
        if predict[i] != expect[i]:
            isMatching = False
    return isMatching


""" Calculate the error of the current set of weights """


def calculGlobal():
   globalError = 0
   for i in range(len(inputs)):
       gen = execute(i)
       out = outputs[i]
       for j in range(len(gen)):
           globalError += (gen[j] - out[j]) ** 2
   return globalError


# Active program

for generation in range(generationCount):
    if generation % (generationCount / 100) == 0:
        print(100 * (generation / generationCount), '% done')
    stack = []
    for i in range(len(inputs)):
        stack += calculus(i)
    generate(stack)

for i in range(len(inputs)):
    prediction = execute(index)
    print(outputs[i], match(prediction, i), smoothing(prediction))

print('Global error:', calculGlobal())
print('Weights:', weights[1:])