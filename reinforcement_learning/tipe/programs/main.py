import pickle
from environment import *


file_name = 'parameters_0'
import_file = True
begin_with = ''

color_map = {'obst': 1/3, 'goal': 2/3, 'self': 1}
actions = np.array([[i, j] for i in [-1, 0, 1] for j in [-1, 0, 1]])
dimension = (25, 25, 1)

simulation = Simulation(dimension, color_map, actions, 12, 40)

if import_file:
    parameters = pickle.load(open(file_name, 'rb'))
    simulation.play(parameters, 36)

else:
    """
    parameters = {'Lc': 5, 'idi': dimension[:2] + (2,),
                  'lt1': 'c', 'kd1': (6, 6, 4),
                  'lt2': 'p', 'ss2': (3, 3), 'pd2': (4, 4),
                  'lt3': 'c', 'kd3': (4, 4, 6),
                  'lt4': 'p', 'ss4': (3, 3), 'pd4': (4, 4)}
    dnn_topology = (6 * 6 * 6, 24, len(actions))
    """
    parameters = {'Lc': 3, 'idi': dimension[:2] + (2,),
                  'lt1': 'c', 'kd1': (3, 3, 4),
                  'lt2': 'p', 'ss2': (2, 2), 'pd2': (3, 3)}
    dnn_topology = (13 * 13 * 4, 64, len(actions))

    if begin_with != '':
        parameters = pickle.load(open(begin_with, 'rb'))
    else:
        parameters = initialize_parameters(parameters, dnn_topology)

    parameters = simulation.train(parameters, 10**-1, 64, 128, 8)
    pickle.dump(parameters, open(file_name, 'wb'))
