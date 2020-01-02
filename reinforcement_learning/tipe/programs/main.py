import pickle
from environment import *


file_name = '66.84CNN25x25_8mouv'
import_file = True
begin_with = '66.84CNN25x25_8mouv'

color_map = {'obst': 1/3, 'goal': 2/3, 'self': 1}
actions = np.array([[i, j] for i in [-1, 0, 1] for j in [-1, 0, 1] if not (i == 0 and j == 0)])
dimension = (25, 25, 1)

simulation = Simulation(dimension, color_map, actions, 12, 36)

if import_file:
    parameters = pickle.load(open(file_name, 'rb'))
    simulation.play(parameters, 32)

else:
    parameters = {'Lc': 3, 'idi': dimension[:2] + (2,),
                  'lt1': 'c', 'kd1': (3, 3, 4),
                  'lt2': 'p', 'ss2': (2, 2), 'pd2': (3, 3)}
    dnn_topology = (13 * 13 * 4, 64, len(actions))

    if begin_with != '':
        parameters = pickle.load(open(begin_with, 'rb'))
    else:
        parameters = initialize_parameters(parameters, dnn_topology)

    parameters = simulation.train(parameters, 10**-1, 256, 12288, 2**9)
    pickle.dump(parameters, open(file_name, 'wb'))

