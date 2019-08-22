""" A genetic algorithm """

import json
import random
import neural_network as nn
from operator import attrgetter

dimension = [784, 10]
mutation_rate = 10 ** -1

print('Loading data...')

file = open('test.txt', 'r')
data = json.loads(file.read())
testing = random.sample(data, 100)
file.close()

print('Loading completed.')


def evolve(generation_count, population_size):
    size = int(population_size / 5)
    population = create_population(population_size)
    ranking = []

    for generation in range(generation_count):
        for individual in population:
            individual.fitness = individual.compute_fitness(testing)
            print(individual.fitness)
        ranking = sorted(population, key=attrgetter("fitness"))
        parents = ranking[-size:]
        mutated_parents = mutate(parents)
        children = cross_over(parents + mutated_parents, population_size - 2 * size)
        population = parents + mutated_parents + children
        print('Generation {0} : {1} %'.format(generation + 1, ranking[-1].fitness * 100 / len(testing)))
    ranking[-1].save_data()


def mutate(parents):
    children = []

    for parent in parents:
        child = nn.Network(dimension)

        for layer in range(1, len(dimension)):
            for neuron in range(dimension[layer]):
                child.biases[layer].matrix[0][neuron] = parent.biases[layer].matrix[0][neuron]
                if random.random() < mutation_rate:
                    child.biases[layer].matrix[0][neuron] += (random.random() - 0.5)
                for previous in range(dimension[layer - 1]):
                    child.weights[layer].matrix[neuron][previous] = parent.weights[layer].matrix[neuron][previous]
                    if random.random() < mutation_rate:
                        child.weights[layer].matrix[neuron][previous] += (random.random() - 0.5)

        children.append(child)

    return children


def cross_over(parents, count):
    children = []

    for _ in range(count):
        mum, dad = random.sample(parents, 2)
        child = nn.Network(dimension)

        for layer in range(1, len(dimension)):
            for neuron in range(dimension[layer]):
                if random.random() < 0.5:
                    child.biases[layer].matrix[0][neuron] = mum.biases[layer].matrix[0][neuron]
                else:
                    child.biases[layer].matrix[0][neuron] = dad.biases[layer].matrix[0][neuron]
                for previous in range(dimension[layer - 1]):
                    if random.random() < 0.5:
                        child.weights[layer].matrix[neuron][previous] = mum.weights[layer].matrix[neuron][previous]
                    else:
                        child.weights[layer].matrix[neuron][previous] = dad.weights[layer].matrix[neuron][previous]

        children.append(child)

    return children


def create_population(size):
    new = []

    for current in range(size):
        new.append(nn.Network(dimension))

    return new
