import pickle
from environment_prime import *


file_name = 'parameters_0'
import_file = False
begin_with = False

color_map = {'obst': 1/3, 'goal': 2/3, 'self': 1}
actions = np.array([[i, j] for i in [-1, 0, 1] for j in [-1, 0, 1]])
dimension = (25, 25)

environment = Environment(dimension, color_map, actions, 12)

if import_file:
    parameters = pickle.load(open(file_name, 'rb'))
    environment.play(parameters, 36, 40)

else:
    if begin_with:
        parameters = pickle.load(open(file_name, 'rb'))
    else:
        parameters = {'Lc': 3, 'idi': dimension + (2,),
                      'lt1': 'c', 'kd1': (3, 3, 4),
                      'lt2': 'p', 'ss2': (2, 2), 'pd2': (3, 3),
                      'Ld': 3, 'tod': (13 * 13 * 4, 64, len(actions))}
        parameters = initialize_parameters(parameters)

    parameters = environment.train(parameters, 10**0, 0.9, 128, 64, 2)
    pickle.dump(parameters, open(file_name, 'wb'))
