""" Main program -- manage UI and convolutional neural network """

import pickle
import convolutional_neural_network as cnn


" Set hyper-parameters "

inputs_dimensions = (28, 28, 1)
layer_count = {'cnn': 1, 'dnn': 2}
CNN_topology = {'kc1': 3}
CNN_functions = {'pf1': 'max', 'af1': 'relu'}
DNN_topology = (588, 32, 10)
DNN_functions = ('', 'relu', 'softmax')

epoch_count = 6
mini_batch_size = 64
alpha = 0.01
training_count = 1000
testing_count = 100

" Get parameters "

arg = (inputs_dimensions, layer_count, CNN_topology, CNN_functions, DNN_topology, DNN_functions)
parameters = cnn.initialize_parameters(*arg)

data = pickle.load(open('data.p', 'rb'))
training = (data['train_x'], data['train_y'])
testing = (data['test_x'], data['test_y'])

(train_x, train_y) = training
training = (train_x[:, :, :, :training_count], train_y[:, :training_count])

(test_x, test_y) = testing
testing = (test_x[:, :, :, :testing_count], test_y[:, :testing_count])

arg = (parameters, epoch_count, mini_batch_size, alpha, training, testing)
parameters = cnn.train(*arg)

" Manage UI "

print("Groin")
