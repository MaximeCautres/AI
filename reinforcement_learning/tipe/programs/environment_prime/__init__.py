import time
import pygame
import matplotlib.pyplot as plt
from convolutional_neural_network import *


def show_stats(stats, t):
    plt.plot(t, stats['max'], color='red')
    plt.plot(t, stats['mean'], color='blue')
    plt.plot(t, stats['min'], color='green')
    plt.ylabel("Win rate")
    plt.xlabel("Iteration")
    plt.show()


def collide(areas, point):
    """
    areas -- (4, obs_count + goal_count)
    point -- (2)
    """

    return np.sum((areas[0] <= point[0]) * (point[0] <= areas[2])
                  * (areas[1] <= point[1]) * (point[1] <= areas[3]))


def generate_map(w, h, obs_count=2, goal_count=1):
    """
    [x-, y-, x+, y+]
    obs -- (4, obs_count)
    goal -- (4, goal_count)
    """

    obs = np.array([[4, 18],
                    [5, 5],
                    [6, 20],
                    [19, 19]], dtype=int)

    goal = np.array([[10],
                    [10],
                    [14],
                    [14]], dtype=int)

    return obs, goal


class Environment:

    def __init__(self, dims, color_map, actions, life_time):

        self.w, self.h = dims
        self.color_map = color_map
        self.actions = actions
        self.life_time = life_time

        self.obs, self.goal = generate_map(*dims)
        self.background = self.initialize_bg()

    def initialize_bg(self):

        img = np.zeros((self.w, self.h, 1))

        for x1, y1, x2, y2 in self.obs.T:
            img[x1:x2 + 1, y1:y2 + 1] = self.color_map['obst']

        for x1, y1, x2, y2 in self.goal.T:
            img[x1:x2 + 1, y1:y2 + 1] = self.color_map['goal']

        return img

    def in_screen(self, point):

        borders = np.array([0, 0, self.w-1, self.h-1])

        return collide(borders, point)

    def generate_pos(self):

        areas = np.append(self.obs, self.goal, axis=1)
        point = None

        while point is None or collide(areas, point):

            point = np.array([np.random.randint(self.w), np.random.randint(self.h)])

        return point

    def generate_img(self, curr, prev):

        px, py = prev; cx, cy = curr
        previous, current = self.background.copy(), self.background.copy()
        previous[px, py], current[cx, cy] = (self.color_map['self'], ) * 2

        return np.concatenate((previous, current), axis=2)

    def train(self, parameters, alpha, gamma, batch_size, epoch_count, print_length):

        games = [Game(self) for _ in range(batch_size)]

        time_begin = time.time()
        stats = {'mean': [], 'min': [], 'max': []}
        win_rates, life_acc = [], 0

        for epoch in range(epoch_count):

            for game in games:
                game.reset()

            flag = True
            while flag:
                flag = False
                for game in games:
                    if game.state == 'play':
                        game.update(parameters)
                        if game.state == 'play':
                            flag = True

            win_count = 0
            features, gradients = [], []
            for game in games:
                f, g = game.get_results(gamma)
                features += f ; gradients += g
                life_acc += game.life
                if game.state == 'win':
                    win_count += 1

            parameters = update_parameters(parameters,
                                           backward(parameters,
                                                    np.stack(features, axis=3),
                                                    np.stack(gradients, axis=1)),
                                           alpha)

            win_rates.append(win_count / batch_size)

            if not (epoch + 1) % print_length:
                a = np.array(win_rates)
                m = np.mean(a)
                stats['min'].append(np.min(a))
                stats['mean'].append(m)
                stats['max'].append(np.max(a))
                print("Epoch {} : {} % / Length mean : {}".format(epoch + 1, round(m * 100, 2),
                                                                  life_acc / (batch_size * print_length)))
                win_rates, life_acc = [], 0

        delta = time.gmtime(time.time() - time_begin)
        print("Finished in {0} hour(s) {1} minute(s) {2} second(s)."
              .format(delta.tm_hour, delta.tm_min, delta.tm_sec))
        show_stats(stats, (np.arange(epoch_count // print_length) + 1) * print_length)

        return parameters

    def play(self, parameters, count, unit):

        pass


class Game:

    def __init__(self, mother):

        self.mother = mother

        self.state = ''  # play / win / lost / draw
        self.pos, self.prev, self.vel = (None, ) * 3
        self.log_img, self.log_act = (None, ) * 2
        self.life = None

    def reset(self):

        self.state = 'play'
        self.pos = self.mother.generate_pos()
        self.prev = self.pos.copy()
        self.vel = np.zeros(2, dtype=int)
        self.log_img, self.log_act = [], []
        self.life = 0

    def update(self, parameters):

        actions = self.mother.actions

        if self.life < self.mother.life_time:

            self.life += 1

            img = self.mother.generate_img(self.pos, self.prev)
            prob = forward(parameters, img.reshape(img.shape + (1, ))).reshape(-1)
            a = np.random.choice(len(actions), p=prob) ; m = prob[a]
            # a = np.argmax(prob) ; m = np.max(prob)
            act = actions[a]

            self.prev = self.pos.copy()
            self.vel += act
            self.pos += self.vel

            if collide(self.mother.obs, self.pos) or not self.mother.in_screen(self.pos):
                self.state = 'lost'
            elif collide(self.mother.goal, self.pos):
                self.state = 'win'

            self.log_img.append(img)
            self.log_act.append((a, m))

        else:
            self.state = 'draw'

    def get_results(self, gamma):

        length = self.life
        gradients = []
        action_count = len(self.mother.actions)
        epsilon = (self.state == 'win') - (self.state == 'lost')

        for t in range(length):

            a, m = self.log_act[t]
            gain = epsilon * (1 - gamma ** (length - t)) / (1 - gamma)
            grad = np.zeros(action_count)
            grad[a] = -np.log(m) * gain
            gradients.append(grad)

        return self.log_img, gradients
