""" Main program -- manage UI and encoder decoder network """

import pickle
import numpy as np
import matplotlib.pyplot as plt
import encoder_decoder_network as edn

" Set hyper-parameters "

save_name = ''
save_type = 'test'
data_set_name = 'mnist_edn'
img_dims = (28, 28, 1)
count = np.prod(img_dims)

encoder = [count, 128, 32, 10]
decoder = [10, 32, 128, count]

epoch_count = 1
mini_batch_size = 128
alpha = 0.6
training_count = 10000
testing_count = 1000

data = pickle.load(open(data_set_name, 'rb'))
training = (data['train_x'][:, :, :, :training_count], data['train_y'][:, :, :, :training_count])
testing = (data['test_x'][:, :, :, :training_count], data['test_y'][:, :, :, :training_count])

" Get parameters "

if save_name == '':

    parameters = edn.initialize_parameters(encoder, decoder)

    arg = (parameters, epoch_count, mini_batch_size, alpha, training, testing, save_type)
    parameters, name, best_cost = edn.train(*arg)

    hyper_parameters = {'epoch_count': epoch_count, 'mini_batch_size': mini_batch_size,
                        'alpha': alpha, 'best_cost': best_cost,
                        'training_count': training_count, 'testing_count': testing_count,
                        'save_type': save_type}
    save = {'parameters': parameters, 'hyper_parameters': hyper_parameters}
    pickle.dump(save, open(data_set_name+'_'+name, 'wb'))

else:

    save = pickle.load(open(save_name, 'rb'))
    parameters = save['parameters']
    print(save['hyper_parameters'])

    test_x, test_y = testing
    permutation = list(np.random.permutation(np.arange(testing_count, dtype=np.int16)))
    shuffled_x = test_x[:, :, :, permutation]
    shuffled_y = test_y[:, :, :, permutation]
    predictions = edn.predict(parameters, shuffled_x)

    for k in range(testing_count):
        img = shuffled_x[:, :, 0, k]
        goal = shuffled_y[:, :, 0, k]
        prediction = predictions[:, :, 0, k]

        plt.close()
        plt.imshow(np.concatenate((img, goal, prediction), axis=1))
        plt.show()
