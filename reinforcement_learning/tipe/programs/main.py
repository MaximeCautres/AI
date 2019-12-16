import pickle
from environment import *


file_name = 'parameters_0'
import_file = False

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
    simulation.play(parameters, 12)

else:
    parameters = {'Lc': 3, 'idi': (dimension.x, dimension.y, 2),
                  'lt1': 'c', 'kd1': (3, 3, 4),
                  'lt2': 'p', 'ss2': (2, 2), 'pd2': (3, 3)}
    dnn_topology = (13 * 13 * 4, 64, len(actions))
    parameters = initialize_parameters(parameters, dnn_topology)

    # simulation.play(parameters, 24)

    parameters = simulation.train(parameters, 6, 2**7, 2**3)
    pickle.dump(parameters, open(file_name, 'wb'))
