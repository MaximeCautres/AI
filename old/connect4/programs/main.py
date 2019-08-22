""" The main program """

import time
import random
import neural_network as nn

generations = 100
step = 10 ** -1

width = 7
height = 6

dim = (width * height, 13, 23, width)

brain_a = nn.Network(dim)
brain_b = nn.Network(dim)

# brain_a = nn.restore('test_a.txt')
# brain_b = nn.restore('test_b.txt')

brains = (brain_a, brain_b)

path = [{'x': 1, 'y': 1}, {'x': 1, 'y': 0}, {'x': 1, 'y': -1}, {'x': 0, 'y': -1}]


def play(a, b):

    count = 0
    index = 0
    winner = None

    players = (a, b)
    array = [0.5 for _ in range(width * height)]

    while count < width * height:

        count += 1

        x = players[index].generate(array)
        y = -1

        while y < height - 1 and array[x + width * (y + 1)] == 0.5:
            y += 1

        if y != -1:
            array[x + width * y] = index

            for i in range(len(path)):
                dx = path[i]['x']
                dy = path[i]['y']
                score = 0
                for unit in range(-1, 2, 2):
                    vec = unit
                    pos = x + dx * vec + width * (y + dy * vec)
                    while 0 <= pos < width * height and array[pos] == index:
                        score += 1
                        vec += unit
                        pos = x + dx * vec + width * (y + dy * vec)
                if score == 3:
                    winner = index
                    break

            index = 1 - index
        else:
            winner = 1 - index

        if winner is not None:
            break

    return winner


def best_brain():

    w1 = play(brain_a, brain_b)
    w2 = play(brain_b, brain_a)

    if w1 == w2 and w1 is not None:
        best = w1
    else:
        best = random.randint(0, 1)

    return best


year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
print("[{0} / {1} / {2} at {3} : {4}] Beginning of the training...".format(day, month, year, hour, minute))
begin = time.time()

for generation in range(generations):
    for layer in range(1, len(dim)):

        for neuron in range(dim[layer]):
            current = best_brain()
            brains[1 - current].biases[layer].matrix[0][neuron] += step
            if current == best_brain():
                brains[1 - current].biases[layer].matrix[0][neuron] -= step * 2

            for previous in range(dim[layer - 1]):
                current = best_brain()
                brains[1 - current].weights[layer].matrix[neuron][previous] += step
                if current == best_brain():
                    brains[1 - current].weights[layer].matrix[neuron][previous] -= step * 2

    print("Generation {0} / {1} done".format(generation + 1, generations))

delta = time.gmtime(time.time() - begin)
print("Finished in {0} hour(s) {1} minute(s) {2} second(s).".format(delta.tm_hour, delta.tm_min, delta.tm_sec))

brain_a.save_data('test_a.txt')
brain_b.save_data('test_b.txt')
