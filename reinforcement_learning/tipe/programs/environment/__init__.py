import time
import pygame
from convolutional_neural_network import *
import matplotlib.pyplot as plt


def mean(array):
    return sum(array) / len(array)


def variance(array):
    array_square = [k * k for k in array]
    return mean(array_square) - mean(array) ** 2


def show_stats(stats, t):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('mean', color=color)
    ax1.plot(t, stats['mean'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('variance', color=color)
    ax2.plot(t, stats['variance'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()


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

    def __init__(self, tlc, brc):
        self.tlc = tlc
        self.brc = brc

    def collide(self, v):
        return self.tlc.x <= v.x <= self.brc.x and self.brc.y <= v.y <= self.tlc.y


class Simulation:

    def __init__(self, dims, obs, goal, goal_radius, color_map, actions, life_time, unit):
        self.dims = dims
        self.obs = obs
        self.goal = goal
        self.goal_radius = goal_radius
        self.color_map = color_map
        self.actions = actions
        self.life_time = life_time
        self.unit = unit

        self.bg = self.create_background()

        self.boat_pos = Vec2()
        self.boat_vel = Vec2()

    def create_background(self):
        bg = np.zeros(self.dims.coord)
        for i in range(self.dims.x):
            for j in range(self.dims.y):
                point = Vec2(i, j)
                type_t = self.type_touched(point)
                if type_t == 'obst':
                    bg[i, j] = self.color_map['obst']
                elif type_t == 'goal':
                    bg[i, j] = self.color_map['goal']
        return bg

    def create_data_set(self, dat, n):
        action_count = len(self.actions)
        f_stack, l_stack = np.zeros((1250, n)), np.zeros((action_count, n))
        k = 0

        for d in dat:
            res, coups = d
            if res == "win":
                for couple in coups:
                    f_occ = couple[0][:, 0]
                    l_occ = np.zeros(action_count)
                    l_occ[np.argmax(couple[1])] = 1
                    f_stack[:, k] = f_occ
                    l_stack[:, k] = l_occ
                    k += 1
            elif res == "loose":
                for couple in coups:
                    f_occ = couple[0][:, 0]
                    l_occ = np.ones(action_count) * 1 / (action_count-1)
                    l_occ[np.argmax(couple[1])] = 0
                    f_stack[:, k] = f_occ
                    l_stack[:, k] = l_occ
                    k += 1

        return f_stack, l_stack

    def numpy_to_pygame(self, source, size):
        array = source.copy()
        for a in self.actions:
            p = self.get_next_pos(a)
            if self.is_in_screen(p):
                array[p.x, p.y] = min(1, array[p.x, p.y] + 0.25)
        surface = pygame.surfarray.make_surface(array * 255)
        return pygame.transform.scale(surface, size)

    def is_in_screen(self, v):
        return 0 <= v.x < self.dims.x and 0 <= v.y < self.dims.y

    def get_spawn(self):
        pos = None
        while self.type_touched(pos):
            pos = Vec2(np.random.randint(self.dims.x), np.random.randint(self.dims.y))
        return pos

    def get_next_pos(self, a):
        return self.boat_pos.plus(self.boat_vel.plus(a))

    def apply_action(self, a):
        new_pos = self.get_next_pos(a)
        collided = not self.is_in_screen(new_pos)
        for obs in self.obs:
            collided = collided or obs.collide(new_pos)
        if not collided:
            self.boat_vel = new_pos.minus(self.boat_pos)
            self.boat_pos = new_pos
            return self.boat_pos.dist(self.goal)

    def type_touched(self, v):
        if not v:
            return 'none'
        if v.dist(self.goal) < self.goal_radius:
            return 'goal'
        for obs in self.obs:
            if obs.collide(v):
                return 'obst'

    def current_frame(self, point):
        bg = self.bg.copy()
        bg[point.x, point.y] = self.color_map['self']
        return bg

    def make_a_move(self, parameters, previous, current):
        array = np.append(np.reshape(previous, 625), np.reshape(current, 625)).reshape(1250, 1)
        a = forward(parameters, array)
        action = self.actions[int(np.argmax(a))]
        previous = current
        res = self.apply_action(action)
        current = self.current_frame(self.boat_pos)
        gs = "in_game"
        if res is None:
            gs = "loose"
        elif res < self.goal_radius:
            gs = "win"
        return previous, current, gs, (array, a)

    def make_a_game(self, parameters, visual=False):
        self.boat_pos = self.get_spawn()
        self.boat_vel = Vec2()
        episode_couples = []

        previous = self.current_frame(self.boat_pos)
        current = previous
        life = 0
        game_state = "in_game"

        if visual:
            pygame.init()
            img_size = self.dims.scale_prod(self.unit).coord
            screen = pygame.display.set_mode(img_size)
            while game_state == "in_game" and life < self.life_time:
                screen.blit(self.numpy_to_pygame(current, img_size), (0, 0))
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        previous, current, game_state, _ = self.make_a_move(parameters, previous, current)
                        life += 1
                pygame.display.flip()
            pygame.quit()

        else:
            while game_state == "in_game" and life < self.life_time:
                previous, current, game_state, couple = self.make_a_move(parameters, previous, current)
                life += 1
                episode_couples.append(couple)

        return game_state, episode_couples

    def train(self, parameters, episode_length, epoch_count, batch):

        time_point = time.time()
        stats = {'mean': [], 'variance': []}
        win_rates = []

        for epoch in range(1, epoch_count + 1):

            data = []
            count = 0
            win_count = 0

            for episode in range(episode_length):
                result, couples = self.make_a_game(parameters)
                if result != "in_game":
                    count += len(couples)
                    data.append((result, couples))
                    if result == 'win':
                        win_count += 1

            win_rates.append(win_count / episode_length)
            if data:
                data_set = self.create_data_set(data, count)
                gradients = backward(parameters, *data_set)
                parameters = update_parameters(parameters, gradients, 10**-1)
            if not epoch % batch:
                m = mean(win_rates)
                v = variance(win_rates)
                stats['mean'].append(m)
                stats['variance'].append(v)
                win_rates = []
                print("Epoch {} : mean = {} -- variance = {}".format(epoch, m, v))

        delta = time.gmtime(time.time() - time_point)
        print("Finished in {0} hour(s) {1} minute(s) {2} second(s).".format(delta.tm_hour, delta.tm_min, delta.tm_sec))
        show_stats(stats, (np.arange(epoch_count // batch) + 1) * batch)

        return parameters

    def play(self, parameters, count):
        for _ in range(count):
            result, _ = self.make_a_game(parameters, True)
            print(result)
