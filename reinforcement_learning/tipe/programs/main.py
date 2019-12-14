import pickle
from environment import *


file_name = 'parameters_1'
import_file = True

color_map = {'obst': 1/3, 'goal': 2/3, 'self': 1}
actions = [Vec2(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]]
dimension = Vec2(25, 25)
rectangles = [Rectangle(Vec2(5, 19), Vec2(7, 5)),
              Rectangle(Vec2(17, 19), Vec2(19, 5))]
goal_pos = Vec2(12, 12)
goal_rad = 2

arguments = (dimension, rectangles, goal_pos, goal_rad, color_map, actions, 12, 36)
simulation = Simulation(*arguments)

if import_file:
    parameters = pickle.load(open(file_name, 'rb'))
    simulation.play(parameters, 24)
else:
    topology = (dimension.x * dimension.y * 2, 128, len(actions))
    parameters = initialize_parameters_dnn(topology)
    parameters = simulation.train(parameters, 128, 2**9, 2**4)
    pickle.dump(parameters, open(file_name, 'wb'))

# else:
#     parameters = {'L': 3, 'id': [25, 25, 2],
#                   'lt1': 'c', 'kd1': (3, 3, 4),
#                   'lt2': 'p', 'ss2': (2, 2), 'pd2': (3, 3)}
#     parameters = initialize_parameters_cnn(parameters)
#     parameters = simulation.train(parameters, 128, 2**9, 2**4)
#     pickle.dump(parameters, open(file_name, 'wb'))
