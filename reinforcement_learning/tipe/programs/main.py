import pickle
from environment import *


file_name = 'parameters_sample_1'
import_file = True
begin_with = False

color_map = {'obst': 1/3, 'goal': 2/3, 'self': 1}
actions = np.array([[i, j] for i in [-1, 0, 1] for j in [-1, 0, 1]])
dimension = (25, 25)
max_move_count = 12

optimizer = 'adadelta'
alpha = 10**0  # learning rate
beta = 0.9  # momentum rate
gamma = 0.85  # rms-prop rate
rho = 0.8  # adadelta rate
xp_discount = 1
epoch_count = 16384
batch_size = 128
print_length = 64

environment = Environment(dimension, color_map, actions, max_move_count)

if import_file:
    parameters = pickle.load(open(file_name, 'rb'))
    environment.play(parameters, 36, 40, True)

else:
    if begin_with:
        parameters = pickle.load(open(file_name, 'rb'))
    else:
        parameters = {'Lc': 3, 'idi': dimension + (2,),
                      'lt1': 'c', 'kd1': (3, 3, 4),
                      'lt2': 'p', 'ss2': (2, 2), 'pd2': (3, 3),
                      'Ld': 3, 'tod': (13*13*4, 64, len(actions))}
        # parameters = {'Lc': 5, 'idi': dimension + (2,),
        #               'lt1': 'c', 'kd1': (4, 4, 5),
        #               'lt2': 'p', 'ss2': (3, 3), 'pd2': (4, 4),
        #               'lt3': 'c', 'kd3': (2, 2, 10),
        #               'lt4': 'p', 'ss4': (2, 2), 'pd4': (3, 3),
        #               'Ld': 3, 'tod': (9*9*10, 128, len(actions))}
        parameters = initialize_parameters(parameters, 42)

    arg = (parameters, optimizer, alpha, beta, gamma, rho, xp_discount, epoch_count, batch_size, print_length)
    parameters = environment.train(*arg)
    pickle.dump(parameters, open(file_name, 'wb'))
