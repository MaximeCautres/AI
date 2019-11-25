import random
import pygame
import numpy as np

RED = 1 / 3
GREEN = 1
BLUE = 2 / 3

topology = (24*24*2, 128, 9)
parameters = {}


def initialize_parameters():
    parameters['layers'] = topology
    parameters['L'] = len(topology)
    for l in range(1, parameters['L']):
        cond = l < parameters['L'] - 1
        parameters['af' + str(l)] = 'relu' * cond + 'softmax' * (not cond)
        parameters['w' + str(l)] = np.random.randn(*topology[l-1:l+1][::-1]) * 10**-2
        parameters['b' + str(l)] = np.zeros((topology[l], 1))


def forward(x):
    a = x
    for l in range(1, parameters['L']):
        z = np.dot(parameters['w' + str(l)], a) + parameters['b' + str(l)]
        if parameters['af' + str(l)] == 'relu':
            a = relu(z)
        elif parameters['af' + str(l)] == 'softmax':
            a = softmax(z)
    return a


def backward(dy):
    return dy


def relu(z):
    return np.maximum(z, 0)


def relu_prime(z):
    return z > 0


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    sigmoidz = sigmoid(z)
    return sigmoidz * (1 - sigmoidz)


def softmax(z):
    return np.divide(np.exp(z), np.sum(np.exp(z), axis=0, keepdims=True))


class Vec2:

    def __init__(self, x=0, y=0):
        self.coord = (x, y)
        self.x = x
        self.y = y

    def plus(self, v):
        return Vec2(self.x + v.x, self.y + v.y)

    def minus(self, v):
        return Vec2(self.x - v.x, self.y - v.y)

    def scale_prod(self, k):
        return Vec2(self.x * k, self.y * k)

    def dot_prod(self, v):
        return self.x * v.x + self.y * v.y

    def norm(self):
        return self.dot_prod(self)**(1/2)

    def dist(self, v):
        return self.minus(v).norm()

    def __str__(self):
        return "({}, {})".format(self.x, self.y)


class Rectangle:

    def __init__(self, tlc, brc, convex=True):
        self.tlc = tlc
        self.brc = brc
        self.convex = convex

    def collide(self, v):
        cond = self.brc.x >= v.x >= self.tlc.x and self.brc.y <= v.y <= self.tlc.y
        return cond and self.convex or not (cond or self.convex)


def simulation(dims, obstacles, p0, v0, goal, goal_radius, life_time, episode_length, epoch_count, unit=None):

    class Boat:

        def __init__(self):
            self.pos = Vec2()
            self.vel = Vec2()

        def apply_action(self, a):
            new_pos = self.pos.plus(self.vel.plus(a))
            collided = False
            for obstacle in obstacles:
                collided = collided or obstacle.collide(new_pos)
            if not collided:
                self.vel = new_pos.minus(self.pos)
                self.pos = new_pos
                return self.pos.dist(goal)

    def create_background():
        bg = np.zeros(dims.coord)
        for i in range(dims.x):
            for j in range(dims.y):
                point = Vec2(i, j)
                if point.dist(goal) < goal_radius:
                    bg[i, j] = GREEN
                for obs in obstacles:
                    if obs.convex and obs.collide(point):
                        bg[i, j] = RED
        return bg

    def create_data(cs, res):
        piece_of_data = []
        if res == "win":
            for couple in cs:
                label = np.zeros((9, 1))
                label[np.argmax(couple[1]), 0] = 1
                piece_of_data.append((couple[0], label))
        elif res == "loose":
            for couple in cs:
                label = np.ones((9, 1)) * 1/8
                label[np.argmax(couple[1]), 0] = 0
                piece_of_data.append((couple[0], label))
        return piece_of_data

    def current_frame(point):
        bg = background.copy()
        bg[point.x, point.y] = BLUE
        return bg

    def numpy_to_pygame(array, size):
        surface = pygame.surfarray.make_surface(array)
        return pygame.transform.scale(surface, size)

    def makone():
        billy.pos = p0
        billy.vel = v0
        episode_couples = []
        game_state = "in_game"
        life = 0

        if unit:
            pygame.init()
            img_size = (dims.scale_prod(unit)).coord
            screen = pygame.display.set_mode(img_size)
            while game_state == "in_game" and life < life_time:
                current = current_frame(billy.pos)
                screen.blit(numpy_to_pygame(current, img_size), (0, 0))
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        action = random.choice(actions)
                        print(action)
                        life += 1
                        res = billy.apply_action(action)
                        if res is None:
                            game_state = "loose"
                        elif res < goal_radius:
                            game_state = "win"
                pygame.display.flip()
            pygame.quit()
        else:
            previous = current_frame(billy.pos)
            while game_state == "in_game" and life < life_time:
                current = current_frame(billy.pos)
                array = np.append(np.reshape(previous, 576), np.reshape(current, 576)).reshape(1152, 1)
                a = forward(array)
                episode_couples.append((array, a))
                action = actions[int(np.argmax(a))]
                previous = current
                life += 1
                res = billy.apply_action(action)
                if res is None:
                    game_state = "loose"
                elif res < goal_radius:
                    game_state = "win"

        return game_state, episode_couples

    actions = [Vec2(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]]
    billy = Boat()
    background = create_background()

    for epoch in range(epoch_count):
        data_set = []
        for episode in range(episode_length):
            result, couples = makone()
            data_set += create_data(couples, result)
        for data in data_set:
            x, y = data
            a = forward(x)
            backward(y - a)


initialize_parameters()
dimension = Vec2(24, 24)
rectangles = [Rectangle(Vec2(0, dimension.y-1), Vec2(dimension.x-1, 0), False),
              Rectangle(Vec2(3, 17), Vec2(5, 15)),
              Rectangle(Vec2(15, 5), Vec2(17, 3))]
init_pos = Vec2(dimension.x // 3, dimension.y // 3)
goal_pos = Vec2(2 * dimension.x // 3, 2 * dimension.y // 3)
simulation(dimension, rectangles, init_pos, Vec2(), goal_pos, 6, 10, 6, 6, 3)
