import pickle
from environment import *


file_name = 'parameters_3'
import_file = True

color_map = {'obst': 1/3, 'goal': 2/3, 'self': 1}
actions = [Vec2(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]]
dimension = Vec2(25, 25)
rectangles = [Rectangle(Vec2(0, dimension.y-1), Vec2(dimension.x-1, 0), False),
              Rectangle(Vec2(5, 19), Vec2(7, 5)),
              Rectangle(Vec2(17, 19), Vec2(19, 5))]
goal_pos = Vec2(12, 12)
goal_rad = 2

arguments = (dimension, rectangles, goal_pos, goal_rad, color_map, actions, 12, 36)
simulation = Simulation(*arguments)

if import_file:
    parameters = pickle.load(open(file_name, 'rb'))
    simulation.play(parameters, 12)
else:
    topology = (dimension.x * dimension.y * 2, 128, len(actions))
    parameters = initialize_parameters(topology)
    parameters = simulation.train(parameters, 128, 2**17, 2**10)
    pickle.dump(parameters, open(file_name, 'wb'))