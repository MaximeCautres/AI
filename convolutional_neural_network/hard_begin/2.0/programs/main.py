""" Main program -- manage UI and convolutional neural network """

import pickle
import numpy as np
import convolutional_neural_network as cnn


" Set hyper-parameters "

inputs_dimensions = (28, 28, 1)
cnn_topology = {'Lc': 1, 'kc0': inputs_dimensions[2], 'kc1': 3, 'kd1': np.array([3, 3]),
                'sc1': (2, 2), 'afc1': 'relu'}
dnn_topology = {'Ld': 1, 'nc0': 588, 'nc1': 10, 'afd1': 'softmax'}

epoch_count = 3
mini_batch_size = 64
alpha = 0.01
training_count = 50000
testing_count = 10000

" Get parameters "

arg = (cnn_topology, dnn_topology)
parameters = cnn.initialize_parameters(*arg)

data = pickle.load(open('data.p', 'rb'))
training = (data['train_x'], data['train_y'])
testing = (data['test_x'], data['test_y'])

(train_x, train_y) = training
training = (train_x[:training_count].astype(np.float32), train_y[:, :training_count])

(test_x, test_y) = testing
testing = (test_x[:testing_count].astype(np.float32), test_y[:, :testing_count])

arg = (parameters, epoch_count, mini_batch_size, alpha, training, testing)
parameters = cnn.train(*arg)

" Manage UI "

print("Groin")
