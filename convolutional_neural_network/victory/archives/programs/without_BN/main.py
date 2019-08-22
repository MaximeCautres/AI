""" Main program -- manage UI and convolutional neural network """

import pickle
import without_BN.convolutional_neural_network as cnn

" Set hyper-parameters "

save_name = ''
save_type = 'test'

cnn_topology = {'Lc': 2,
                'kc1': 5, 'kc2': 11,
                'kd1': (6, 6), 'kd2': (2, 2),
                'sc1': (4, 4), 'sc2': (2, 2),
                'afc1': 'relu', 'afc2': 'relu'
                }
dnn_topology = {'Ld': 2,
                'nc1': 48, 'nc2': 15,
                'afd1': 'relu', 'afd2': 'softmax',
                'dor0': 0, 'dor1': 0
                }

epoch_count = 16
mini_batch_size = 128
optimizer = 'adadelta'
alpha = 0.01
beta = 0.9
gamma = 0.9
rho = 0.95
lambda2C = 0.01
lambda2D = 0.1
training_count = 15000
testing_count = 3000

inputs_dimensions = (32, 32, 3)
goal = 6
iteration_count = 32
galpha = 1
refresh_rate = 0.05

" Get parameters "

if save_name == '':
    arg = (cnn_topology, dnn_topology, inputs_dimensions)
    parameters = cnn.initialize_parameters(*arg)

    print('node_count_dnn_0:', parameters['nc0'])

    data = pickle.load(open('cifar_15_sorted_inverted', 'rb'))
    training = (data['train_x'], data['train_y'])
    testing = (data['test_x'], data['test_y'])

    (train_x, train_y) = training
    training = (train_x[:, :, :, :training_count], train_y[:, :training_count])

    (test_x, test_y) = testing
    testing = (test_x[:, :, :, :testing_count], test_y[:, :testing_count])

    arg = (parameters, epoch_count, mini_batch_size, optimizer, alpha, beta, gamma, rho, lambda2C, lambda2D, training, testing, save_type)
    parameters, name, best_cost = cnn.train(*arg)

    hyper_parameters = {'cnn_topology': cnn_topology, 'dnn_topology': dnn_topology,
                        'epoch_count': epoch_count, 'mini_batch_size': mini_batch_size,
                        'optimizer': optimizer, 'alpha': alpha, 'beta': beta,
                        'gamma': gamma, 'rho': rho, 'lambda2C': lambda2C, 'lambda2D': lambda2D,
                        'best_cost': best_cost, 'training_count': training_count,
                        'testing_count': testing_count, 'save_type': save_type}
    save = {'parameters' : parameters, 'hyper_parameters' : hyper_parameters}
    pickle.dump(save, open(name, 'wb'))

else:
    save = pickle.load(open(save_name, 'rb'))
    parameters = save['parameters']
    print(save['hyper_parameters'])

    cnn.generate(parameters, inputs_dimensions, goal, iteration_count, galpha, refresh_rate)

" Manage UI "

print("Groin")
