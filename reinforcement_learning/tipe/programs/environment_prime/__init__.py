import time
import pygame
import matplotlib.pyplot as plt
from convolutional_neural_network import *


def generate_map(w, h, obs_count=2, goal_count=1):
    """
    [x-, y-, x+, y+]
    obs -- (4, p, n)
    goal -- (4, q, n)
    pos -- (2, 1, n)
    vel -- (2, 1, n)
    """

    obs = np.array([[4, 18],
                    [5, 5],
                    [6, 20],
                    [19, 19]], dtype=int)

    goal = np.array([[10],
                    [10],
                    [14],
                    [14]], dtype=int)

    pos = ()
    vel = np.zeros((2, 1, 1), dtype=int)

    return obs, goal, pos, vel

class Environment:

    def __init__(self, dims, color_map, actions, life_time):

        self.dims = dims
        self.color_map = color_map
        self.actions = actions
        self.life_time = life_time

        self.map = generate_map(*dims)

    def train(self, parameters, alpha, gamma, batch_size, epoch_count, print_length):

        games = [Game(self) for _ in batch_size]

        for epoch in range(epoch_count):
            while 'there exists a game not ended':
                for game in games:
                    game.update()

    def play(self, parameters, count, unit):

        pass


class Game:

    def __init__(self, mother):

        self.mother = mother
        self.log = []
        self.pos = ()

    def update(self):

        pass
