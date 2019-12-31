from environment import *
import numpy as np
import pickle
import matplotlib.pyplot as plt

color_map = {'obst': 1/3, 'goal': 2/3, 'self': 1}
actions = np.array([[i, j] for i in [-1, 0, 1] for j in [-1, 0, 1]])
dimension = (25, 25, 1)

simulation = Simulation(dimension, color_map, actions, 12, 36)
