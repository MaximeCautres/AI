import math
import random

# Create variables

inputs = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 1]]  # All the inputs

for i in range(len(inputs)):  # Add bias
  inputs[i] += [1]

outputs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]  # All the outputs

parameters = [len(inputs[0]), len(outputs[0])]  # Length of each line of neuron

generationCount = 10 ** 5  # Set the number of time the neuronal network have to be activated


best = [[[2 * random.random() - 1 for z in range(parameters[x - 1])]
            for y in range(parameters[x])] for x in range(len(parameters))]
bestError = math.inf

""" Apply the sigmoide function """


def sigmoide(x):
    return 1 / (1 + math.exp(-x))


""" Execute the neuronal network """


def execute(entry, matrix):
    values = [[1 for y in range(parameters[x])] for x in range(len(parameters))]
    for i in range(parameters[0]):
        values[0][i] = inputs[entry][i]

    for line in range(1, len(parameters)):
        for neuron in range(parameters[line]):
            stack = 0
            for previous in range(parameters[line - 1]):
                stack += values[line - 1][previous] * matrix[line][neuron][previous]
            values[line][neuron] = sigmoide(stack)
    return values[-1]


""" Calculate the error of the current set of weights """


def calculGlobal(matrix):
   globalError = 0
   for i in range(len(inputs)):
       gen = execute(i, matrix)
       out = outputs[i]
       for j in range(len(gen)):
           globalError += (gen[j] - out[j]) ** 6
   return globalError


""" Return a the smoothing version of this array """


def smoothing(array):
    new = [0 for x in range(len(array))]
    for i in range(len(array)):
        if array[i] < 0.5:
            new[i] = 0
        else:
            new[i] = 1
    return new


# Active program

for generation in range(generationCount):
    if generation % (generationCount / 100) == 0:
        print(100 * (generation / generationCount), '% done')
    current = [[[2 * random.random() - 1 for z in range(parameters[x - 1])]
            for y in range(parameters[x])] for x in range(len(parameters))]
    error = calculGlobal(current)
    if error < bestError:
        bestError = error
        best = current

for i in range(len(inputs)):
    prediction = execute(i, best)
    print(outputs[i], smoothing(prediction), prediction)

print('Global error:', bestError)
print('Weights:', best[1:])