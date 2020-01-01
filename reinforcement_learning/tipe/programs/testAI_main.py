import numpy as np
import pickle
import matplotlib.pyplot as plt
from convolutional_neural_network import *
from random import *

def show_stats(stats, t):
    plt.plot(t, stats, color='red')
    plt.ylabel("success rate")
    plt.xlabel("Iteration")
    plt.show()


def create_mini_batches(x, y, size):
    """
    Generate an array of couple (features, labels) shuffled

    Take :
    x -- features (n0, m)
    y -- labels (nL, m)
    size -- scalar

    Return :
    couples -- array of couple
    """

    couples = []
    m = x.shape[3]

    permutation = list(np.random.permutation(np.arange(m, dtype=np.int16)))

    shuffled_x = x[:, :, :, permutation]
    shuffled_y = y[:, permutation]

    num_complete_mini_batches = int(m / size)

    for k in range(num_complete_mini_batches):
        dx = shuffled_x[:, :, :, k * size: (k+1) * size]
        dy = shuffled_y[:, k * size: (k+1) * size]
        couple = (dx, dy)
        couples.append(couple)

    if m % size != 0:
        dx = shuffled_x[:, :, :,  num_complete_mini_batches * size: m]
        dy = shuffled_y[:, num_complete_mini_batches * size: m]
        couple = (dx, dy)
        couples.append(couple)

    return couples

data_base_name = 'mnist'

image_dimension = (28, 28, 1)

epoch_count = 32
batch_length = 64
alpha = 10**(-3)

parameters = {'Lc': 0, 'idi': image_dimension[:2] + (1,),
              'lt1': 'c', 'kd1': (3, 3, 4),
              'lt2': 'p', 'ss2': (3, 3), 'pd2': (4, 4)}
dnn_topology = (784, 150, 10)

parameters = initialize_parameters(parameters, dnn_topology)

data = pickle.load(open(data_base_name, 'rb'))
data_x, data_y = data['train_x'], data['train_y']
data_test_x, data_test_y = data['test_x'], data['test_y']
print(data_x.shape)

k = 0
success = []



for epoch in range(epoch_count):
    batchs = create_mini_batches(data_x, data_y, batch_length)
    p = 0
    for batch in batchs:
        p += 1
        gradients = backward(parameters, batch[0], batch[1])
        parameters = update_parameters(parameters, gradients, alpha)
    c = 0
    y = np.argmax(forward(parameters, data_test_x), axis=0)
    for i in range(len(y)):
        if data_test_y[y[i], i]:
            c += 1

    print("Epoch {} : {} %".format(epoch + 1, 100 * c/len(y)))
    success.append(c/len(data_test_x[0, 0, 0]))
show_stats(success, np.arange(1, epoch_count + 1))



