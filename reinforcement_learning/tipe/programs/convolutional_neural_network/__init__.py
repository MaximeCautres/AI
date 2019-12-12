from deep_neural_network import *


def initialize_parameters_cnn(topology):
    parameters = {'layers': topology, 'L': len(topology)}
    for l in range(1, parameters['L']):
        cond = l < parameters['L'] - 1
        parameters['af' + str(l)] = 'relu' * cond + 'softmax' * (not cond)
        parameters['w' + str(l)] = np.random.randn(*topology[l-1:l+1][::-1]) * 10**-1
        parameters['b' + str(l)] = np.zeros((topology[l], 1))
    return parameters
