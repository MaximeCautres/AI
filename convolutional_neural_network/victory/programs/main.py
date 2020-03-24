""" Main program -- manage UI and convolutional neural network """

import pickle
import numpy as np
import matplotlib.pyplot as plt
import convolutional_neural_network as cnn

" Set hyper-parameters "

save_name = ''
save_type = 'test'
data_set_name = 'cifar_15'
count = 15
inputs_dimensions = (32, 32, 3)

cnn_topology = {'Lc': 2,
                'kc1': 6, 'kc2': 12,
                'kd1': (4, 4), 'kd2': (3, 3),
                'sc1': (3, 3), 'sc2': (2, 2),
                'afc1': 'relu', 'afc2': 'relu',
                }
dnn_topology = {'Ld': 2,
                'nc1': 48, 'nc2': count,
                'afd1': 'relu', 'afd2': 'softmax',
                'dor0': 0.2, 'dor1': 0.2
                }

epoch_count = 16
mini_batch_size = 256
optimizer = 'momentum'
alpha = '0.6 + 0.4 * math.sin(math.pi * t)'
beta = '0.95 - 0.05 * math.sin(math.pi * t)'
gamma = '0.9'
rho = '0.9'
lambda2C = 0.01
lambda2D = 0.1
training_count = 15000
testing_count = 1500

data = pickle.load(open(data_set_name, 'rb'))
training = (data['train_x'][:, :, :, :training_count], data['train_y'][:, :training_count])
testing = (data['test_x'][:, :, :, :training_count], data['test_y'][:, :training_count])
labels = data['labels']

" Get parameters "

if save_name == '':

    arg = (cnn_topology, dnn_topology, inputs_dimensions)
    parameters = cnn.initialize_parameters(*arg)

    print('node_count_dnn_0:', parameters['nc0'])

    arg = (parameters, epoch_count, mini_batch_size, optimizer, alpha, beta, gamma, rho, lambda2C, lambda2D, training, testing, save_type)
    parameters, name, best_cost = cnn.train(*arg)

    hyper_parameters = {'cnn_topology': cnn_topology, 'dnn_topology': dnn_topology,
                        'epoch_count': epoch_count, 'mini_batch_size': mini_batch_size,
                        'optimizer': optimizer, 'alpha': alpha, 'beta': beta,
                        'gamma': gamma, 'rho': rho, 'lambda2C': lambda2C, 'lambda2D': lambda2D,
                        'best_cost': best_cost, 'training_count': training_count,
                        'testing_count': testing_count, 'save_type': save_type}
    save = {'parameters': parameters, 'hyper_parameters': hyper_parameters}
    pickle.dump(save, open(data_set_name+'_'+name, 'wb'))

else:

    save = pickle.load(open(save_name, 'rb'))
    parameters = save['parameters']
    print(save['hyper_parameters'])

    test_x, test_y = testing
    permutation = list(np.random.permutation(np.arange(testing_count, dtype=np.int16)))
    shuffled_x = test_x[:, :, :, permutation]
    shuffled_y = test_y[:, permutation]

    for i in range(testing_count):
        img = shuffled_x[:, :, :, i].reshape(inputs_dimensions + (1, ))
        goal = list(shuffled_y[:, i]).index(1)
        prediction = cnn.predict(parameters, img)

        result = {labels[k]: round(float(prediction[k]) * 100, 2) for k in range(count)}

        print(sorted(result.items(), key=lambda z: z[1], reverse=True))
        print(labels[goal])

        plt.close()
        plt.imshow(np.squeeze(img))
        plt.show()
