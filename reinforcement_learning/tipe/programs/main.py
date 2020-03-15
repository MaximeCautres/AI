import pickle
from environment_prime import *


file_name = 'parameters_0'
import_file = True
begin_with = ''

color_map = {'obst': 1/3, 'goal': 2/3, 'self': 1}
actions = np.array([[i, j] for i in [-1, 0, 1] for j in [-1, 0, 1]])
dimension = (25, 25, 1)

simulation = Simulation(dimension, color_map, actions, 12)

if file_name != '':
    parameters = pickle.load(open(file_name, 'rb'))
    simulation.play(parameters, 40, 36)

else:
    parameters = {'Lc': 3, 'idi': dimension[:2] + (2,),
                  'lt1': 'c', 'kd1': (3, 3, 4),
                  'lt2': 'p', 'ss2': (2, 2), 'pd2': (3, 3),
                  'Ld': 3, 'tod': (13 * 13 * 4, 64, len(actions))}

    if begin_with != '':
        parameters = pickle.load(open(begin_with, 'rb'))
    else:
        parameters = initialize_parameters(parameters)

    parameters = simulation.train(parameters, 10**-1, 0.94, 128, 2048, 1)
    pickle.dump(parameters, open(file_name, 'wb'))
