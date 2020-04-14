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


def normalize(array):

    if len(array) < 2:
        return array

    np_array = np.array(array)
    mean = np.mean(np_array)
    std = np.std(np_array)

    return list((np_array - mean) / std)


def collide(areas, point):
    """
    areas -- (4, obs_count + goal_count)
    point -- (2)
    """

    return np.sum((areas[0] <= point[0]) * (point[0] <= areas[2])
                  * (areas[1] <= point[1]) * (point[1] <= areas[3]))


def first_map_25x25():

    obs = np.array([[4, 18],
                    [5, 5],
                    [6, 20],
                    [19, 19]], dtype=int)

    goal = np.array([[10],
                    [10],
                    [14],
                    [14]], dtype=int)

    return obs, goal


def map_48x48():

    obs = np.array([[0, 8, 33, 40, 35, 41, 25, 15, 10, 8, 10],
                    [8, 5, 5, 0, 18, 16, 22, 20, 22, 25, 36],
                    [9, 16, 41, 47, 43, 47, 29, 28, 16, 12, 18],
                    [12, 10, 10, 7, 22, 41, 38, 23, 27, 37, 40]], dtype=int)

    goal = np.array([[2, 43, 14],
                     [2, 10, 30],
                     [5, 46, 17],
                     [5, 13, 33]], dtype=int)

    return obs, goal


def moving_goal_map_25x25():

    obs = np.array([[],
                    [],
                    [],
                    []], dtype=int)

    x, y = np.random.randint(1, 24, 2)

    goal = np.array([[x - 1],
                     [y - 1],
                     [x + 1],
                     [y + 1]], dtype=int)

    return obs, goal


def get_map():

    return moving_goal_map_25x25()


class Environment:

    def __init__(self, dims, color_map, actions, life_time):

        self.w, self.h = dims
        self.color_map = color_map
        self.actions = actions
        self.life_time = life_time

        self.bus = []
        self.exploration_rate = 0

    def in_screen(self, pos):

        borders = np.array([0, 0, self.w - 1, self.h - 1])

        return collide(borders, pos)

    def generate_frame(self, game, size):

        distribution = game.log_act[-1][1]
        array = np.squeeze(np.copy(game.background))

        if self.in_screen(game.pos):
            x, y = game.pos
            array[x, y] = self.color_map['self']

        surface = pygame.surfarray.make_surface(array * 255)

        new_pos = game.pos + game.vel
        ac = len(distribution)
        red = min(distribution)
        green = max(distribution)
        delta = green - red

        for k in range(ac):
            p = new_pos + self.actions[k]
            if self.in_screen(p):
                z = distribution[k]
                color = pygame.Color(int(255 * (green - z) / delta),
                                     int(255 * (z - red) / delta),
                                     0)
                surface.set_at(tuple(p), color)

        return pygame.transform.scale(surface, size)

    def terminus(self, parameters):

        probabilities = forward(parameters, np.stack(self.bus, axis=3))
        self.bus = []

        return probabilities

    def train(self, parameters, optimizer, alpha, beta, gamma, rho, xp_discount, epoch_count, batch_size, print_length):

        games = [Game(self) for _ in range(batch_size)]

        time_begin = time.time()
        stats = {'mean': [], 'min': [], 'max': []}
        win_rates, life_acc = [], 0
        self.exploration_rate = xp_discount
        self.bus = []

        for epoch in range(epoch_count):

            for game in games:
                game.reset()

            while 0 < len(self.bus):

                probabilities = self.terminus(parameters)
                index = 0

                for game in games:
                    if game.state == 'play':
                        game.take_prob(np.squeeze(probabilities[..., index]))
                        game.update()
                        index += 1

            win_count = 0
            images, grads = [], []
            for game in games:
                images += game.log_img ; grads += game.get_grad(gamma)
                life_acc += game.life
                if game.state == 'win':
                    win_count += 1

            if 0 < len(images) + len(grads):
                gradients = backward(parameters, np.stack(images, axis=3), np.stack(grads, axis=1))
                parameters = update_parameters(parameters, gradients, optimizer, alpha, beta, rho)

            win_rates.append(win_count / batch_size)
            self.exploration_rate *= xp_discount

            if not (epoch + 1) % print_length:
                a = np.array(win_rates)
                m = np.mean(a)
                stats['min'].append(np.min(a))
                stats['mean'].append(m)
                stats['max'].append(np.max(a))
                print("Epoch {} : win_rate = {} % / length_mean = {}".format(epoch + 1, round(m * 100, 2),
                                                                             life_acc / (batch_size * print_length)))
                win_rates, life_acc = [], 0

        delta = time.gmtime(time.time() - time_begin)
        print("Finished in {0} hour(s) {1} minute(s) {2} second(s)."
              .format(delta.tm_hour, delta.tm_min, delta.tm_sec))
        show_stats(stats, (np.arange(epoch_count // print_length) + 1) * print_length)

        return parameters

    def play(self, parameters, count, unit, use_sample=False):

        game = Game(self)

        pygame.init()
        img_size = (self.w * unit, self.h * unit)
        screen = pygame.display.set_mode(img_size)
        self.exploration_rate = int(use_sample)

        for _ in range(count):

            game.reset()
            game.take_prob(np.squeeze(self.terminus(parameters)))
            frame = self.generate_frame(game, img_size)

            while game.state == 'play':

                screen.blit(frame, (0, 0))

                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONDOWN:

                        game.update()
                        if 0 < len(self.bus):
                            game.take_prob(np.squeeze(self.terminus(parameters)))
                            frame = self.generate_frame(game, img_size)

                pygame.display.flip()
            print(game.state)
        pygame.quit()


class Game:

    def __init__(self, env):

        self.env = env

        self.state = ''  # play / win / lost / draw
        self.pos, self.prev, self.vel = (None, ) * 3
        self.log_img, self.log_act = (None, ) * 2
        self.obs, self.goal = (None,) * 2
        self.background = None
        self.life = None

    def initialize_bg(self):

        img = np.zeros((self.env.w, self.env.h, 1))

        for x1, y1, x2, y2 in self.obs.T:
            img[x1:x2 + 1, y1:y2 + 1] = self.env.color_map['obst']

        for x1, y1, x2, y2 in self.goal.T:
            img[x1:x2 + 1, y1:y2 + 1] = self.env.color_map['goal']

        return img

    def generate_pos(self):

        areas = np.append(self.obs, self.goal, axis=1)
        point = None

        while point is None or collide(areas, point):
            point = np.array([np.random.randint(self.env.w), np.random.randint(self.env.h)])

        return point

    def generate_img(self, prev, curr):

        px, py = prev ; cx, cy = curr
        previous, current = np.copy(self.background), np.copy(self.background)
        previous[px, py], current[cx, cy] = (self.env.color_map['self'],) * 2

        return np.concatenate((previous, current), axis=2)

    def reset(self):

        self.obs, self.goal = get_map()
        self.background = self.initialize_bg()

        self.state = 'play'
        self.pos = self.generate_pos()
        self.prev = np.copy(self.pos)
        self.vel = np.zeros(2, dtype=int)
        self.log_img, self.log_act = [], []
        self.life = 0

        self.add_img_to_bus()

    def add_img_to_bus(self):

        img = self.generate_img(self.prev, self.pos)
        self.env.bus.append(img)
        self.log_img.append(img)

    def take_prob(self, prob):

        if np.random.random() < self.env.exploration_rate:
            a = np.random.choice(len(self.env.actions), p=prob)
        else:
            a = np.argmax(prob)

        self.log_act.append((a, prob))

    def update(self):

        actions = self.env.actions

        if self.life < self.env.life_time:

            self.prev = np.copy(self.pos)
            self.vel += actions[self.log_act[-1][0]]
            self.pos += self.vel

            if collide(self.obs, self.pos) or not self.env.in_screen(self.pos):
                self.state = 'lost'
            elif collide(self.goal, self.pos):
                self.state = 'win'
            else:
                self.add_img_to_bus()

        else:
            self.state = 'draw'

        self.life += 1

    def get_grad(self, gamma):

        grad = []
        length = self.life
        action_count = len(self.env.actions)
        gain = 1 if self.state == "win" else -1

        for t in range(length):
            a, y_hat = self.log_act[t]
            y = np.zeros(action_count)
            y[a] = 1
            grad.append((y - y_hat) * gain * gamma ** (length - t))

        return grad
