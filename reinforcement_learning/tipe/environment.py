import random
import pygame
import numpy as np

RED = 1 / 3
GREEN = 2 / 3
BLUE = 1

topology = (24*24*2, 128, 9)
parameters = {}


def initialize_parameters():
    parameters['layers'] = topology
    parameters['L'] = len(topology)
    for l in range(1, parameters['L']):
        cond = l < parameters['L'] - 1
        parameters['af' + str(l)] = 'relu' * cond + 'softmax' * (not cond)
        parameters['w' + str(l)] = np.random.randn(*topology[l-1:l+1][::-1]) * 10**-1
        parameters['b' + str(l)] = np.zeros((topology[l], 1))


def forward(x, return_cache=False):
    a = x
    cache = {'a0': a}

    for l in range(1, parameters['L']):
        w = parameters['w' + str(l)]
        b = parameters['b' + str(l)]
        af = parameters['af' + str(l)]

        z = np.dot(w, a) + b
        if af == 'relu':
            a = relu(z)
        elif af == 'softmax':
            a = softmax(z)

        cache['z' + str(l)] = z
        cache['a' + str(l)] = a

    if return_cache:
        return cache
    else:
        return a


def backward(x, y):
    gradients = {}
    n = x.shape[1]
    cache = forward(x, True)
    y_hat = cache['a' + str(parameters['L']-1)]

    da = np.divide(1 - y, 1 - y_hat) - np.divide(y, y_hat)
    dz = None

    for l in reversed(range(1, parameters['L'])):
        z = cache['z' + str(l)]
        af = parameters['af' + str(l)]

        if af == 'relu':
            dz = da * relu_prime(z)
        elif af == 'softmax':
            dz = y_hat - y

        a_prev = cache['a' + str(l - 1)]
        w = parameters['w' + str(l)]

        gradients['dw' + str(l)] = (1 / n) * np.dot(dz, a_prev.T)
        gradients['db' + str(l)] = (1 / n) * np.sum(dz, axis=1, keepdims=True)

        da = np.dot(w.T, dz)

    return gradients


def update_parameters(gradients, alpha):
    for l in range(1, parameters['L']):
        parameters['w' + str(l)] -= alpha * gradients['dw' + str(l)]
        parameters['b' + str(l)] -= alpha * gradients['db' + str(l)]


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


def simulation(dims, obstacles, v0, goal, goal_radius, life_time, episode_length, epoch_count, unit=None):

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

    def type_touched(v):
        if not v:
            return 'pasouf'
        if v.dist(goal) < goal_radius:
            return 'goal'
        for obs in obstacles:
            if obs.convex and obs.collide(v):
                return 'obs'

    def create_background():
        bg = np.zeros(dims.coord)
        for i in range(dims.x):
            for j in range(dims.y):
                point = Vec2(i, j)
                type_t = type_touched(point)
                if type_t == 'obs':
                    bg[i, j] = RED
                elif type_t == 'goal':
                    bg[i, j] = GREEN
        return bg

    def get_pos():
        pos = None
        while type_touched(pos):
            pos = Vec2(np.random.randint(dims.x), np.random.randint(dims.y))
        return pos

    def create_data_set(dat, n):
        f_stack, l_stack = np.zeros((1152, n)), np.zeros((9, n))
        k = 0

        for d in dat:
            res, coups = d

            if res == "win":
                for couple in coups:
                    f_occ = couple[0][:, 0]
                    l_occ = np.zeros(9)
                    l_occ[np.argmax(couple[1])] = 1

                    f_stack[:, k] = f_occ
                    l_stack[:, k] = l_occ
                    k += 1

            elif res == "loose":
                for couple in coups:
                    f_occ = couple[0][:, 0]
                    l_occ = np.ones(9) * 1/8
                    l_occ[np.argmax(couple[1])] = 0

                    f_stack[:, k] = f_occ
                    l_stack[:, k] = l_occ
                    k += 1

        return f_stack, l_stack

    def current_frame(point):
        bg = background.copy()
        bg[point.x, point.y] = BLUE
        return bg

    def numpy_to_pygame(array, size):
        surface = pygame.surfarray.make_surface(array * 255)
        return pygame.transform.scale(surface, size)

    def makone():
        billy.pos = get_pos()
        billy.vel = v0
        episode_couples = []
        game_state = "in_game"
        life = 0
        previous = current_frame(billy.pos)
        current = previous

        if unit:
            pygame.init()
            img_size = (dims.scale_prod(unit)).coord
            screen = pygame.display.set_mode(img_size)
            while game_state == "in_game" and life < life_time:
                screen.blit(numpy_to_pygame(current, img_size), (0, 0))
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        current = current_frame(billy.pos)
                        array = np.append(np.reshape(previous, 576), np.reshape(current, 576)).reshape(1152, 1)
                        a = forward(array)
                        print(a)
                        episode_couples.append((array, a))
                        action = actions[int(np.argmax(a))]
                        previous = current
                        life += 1
                        res = billy.apply_action(action)
                        if res is None:
                            game_state = "loose"
                        elif res < goal_radius:
                            game_state = "win"
                pygame.display.flip()
            pygame.quit()
        else:
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
        data = []
        count = 0
        win_count = 0
        for episode in range(episode_length):
            result, couples = makone()
            if result != "in_game":
                count += len(couples)
                data.append((result, couples))
                if result == 'win':
                    win_count += 1
        data_set = create_data_set(data, count)
        grads = backward(*data_set)
        update_parameters(grads, 10**-1)
        print(win_count / episode_length)


initialize_parameters()
dimension = Vec2(24, 24)
rectangles = [Rectangle(Vec2(0, dimension.y-1), Vec2(dimension.x-1, 0), False),
              Rectangle(Vec2(3, 17), Vec2(5, 15)),
              Rectangle(Vec2(15, 5), Vec2(17, 3))]
goal_pos = Vec2(2 * dimension.x // 3, 2 * dimension.y // 3)
simulation(dimension, rectangles, Vec2(), goal_pos, 6, 10, 128, 128)
simulation(dimension, rectangles, Vec2(), goal_pos, 6, 10, 1, 6, 36)
