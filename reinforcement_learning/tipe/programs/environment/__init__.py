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


def collide(areas, p):
    return np.sum((areas[0] <= p[0]) * (p[0] <= areas[2]) * (areas[1] <= p[1]) * (p[1] <= areas[3]), axis=0)


def get_map(n, w, h, p=2, q=1):
    """
    [x-, y-, x+, y+]
    obs -- (4, p, n)
    goal -- (4, q, n)
    pos -- (2, 1, n)
    vel -- (2, 1, n)
    """

    obs_brut = np.array([[4, 5, 6, 19], [18, 5, 20, 19]], dtype=int)
    obs = np.broadcast_to(obs_brut.T.reshape(4, p, 1), (4, p, n))

    goal_brut = np.array([[[10, 10, 14, 14]]], dtype=int)
    goal = np.broadcast_to(goal_brut.T.reshape(4, q, 1), (4, q, n))

    pos = get_spawn(np.concatenate((obs, goal), axis=1), n, w, h)
    vel = np.zeros((2, 1, 1), dtype=int)

    return obs, goal, pos, vel


def get_spawn(areas, n, w, h):
    """
    areas -- (4, p + q, n)
    """

    len_l_c = n
    l_c = list(range(n))
    l_s = np.zeros((2, 1, n), dtype=int)

    while len_l_c:

        x = np.random.randint(w, size=len_l_c)
        y = np.random.randint(h, size=len_l_c)
        l_s[:, 0, l_c] = np.array([x, y])
        table = collide(areas[..., l_c], l_s[..., l_c])
        l_c = [l_c[k] for k in range(len_l_c) if table[k]]
        len_l_c = len(l_c)

    return l_s


class Simulation:

    def __init__(self, dims, color_map, actions, life_time, unit):

        self.dims = dims
        self.color_map = color_map
        self.actions = actions
        self.life_time = life_time
        self.unit = unit

        self.igc, self.obs, self.goal, self.pos, self.vel, self.bg = (None, ) * 6

    def get_bg(self):

        _, p, _ = self.obs.shape
        _, q, _ = self.goal.shape
        img = np.zeros(self.dims + (self.igc,))

        for k in range(self.igc):

            for j in range(p):
                x1, y1, x2, y2 = self.obs[:, j, k]
                img[x1:x2+1, y1:y2+1, :, k] = self.color_map['obst']

            for j in range(q):
                x1, y1, x2, y2 = self.goal[:, j, k]
                img[x1:x2+1, y1:y2+1, :, k] = self.color_map['goal']

        return img

    def get_img(self, bg):

        x, y = self.pos
        bg[x, y, :, range(self.igc)] = self.color_map['self']

        return bg

    def get_result(self, p):

        win_table = collide(self.goal, p)
        loose_table = collide(self.obs, p) + (True - self.in_screen(p))

        win = [k for k in range(self.igc) if win_table[k]]
        loose = [k for k in range(self.igc) if loose_table[k]]

        return win, loose

    def in_screen(self, p):

        w, h, _ = self.dims
        borders = np.array([0, 0, w - 1, h - 1])

        return collide(borders, p)

    def transform(self, array_in, result='None'):

        array = np.moveaxis(array_in, -1, 1).reshape(*((-1,) + array_in.shape[1:-1]))
        ac = self.actions.shape[0]
        k = array.size
        array_out = array

        if result == 'win':
            array_out = np.zeros((k, ac))
            array_out[range(k), array] = 1
        elif result == 'loose':
            array_out = np.full((k, ac), 1 / (ac - 1))
            array_out[range(k), array] = 0

        return list(array_out)

    def make_a_batch(self, parameters, n):
        """
        This function creates the experience batches used in the policy gradient algorithm.
        - w, h, d -- dimension of the map's image
        - igc -- current number of in game parties
        - goal -- (4, q, n)
        - pos -- (2, 1, n)
        - vel -- (2, 1, n)
        - bg -- background of the image (the map)
        - log_im -- the list where in_game's data will stay until treatment
        - log_act -- the list where in_game's actions will stay until treatment
        - features -- the list where finished game's data are stoked
        - labels -- the list where the label conresponding to an end are stocked
        - life_time -- maximal number of decisions in a game
        - life -- current number of decisions
        """

        w, h, d = self.dims
        self.igc = n
        self.obs, self.goal, self.pos, self.vel = get_map(n, w, h)
        self.bg = self.get_bg()
        win_count = 0
        life = 0

        init = self.get_img(self.bg.copy())
        log_img = np.concatenate((init, init), axis=2).reshape(1, w, h, 2*d, n)
        log_act = None
        features = []
        labels = []

        while life < self.life_time and 0 < self.igc:

            life += 1
            img = log_img[-1]
            prob = forward(parameters, img)
            # a = [int(np.random.choice(9, 1, p=p)) for p in prob.T]
            a = np.argmax(prob, axis=0)
            action = self.actions[a].T.reshape(2, 1, self.igc)

            new_act = np.array([a])
            if log_act is not None:
                new_act = np.concatenate((log_act, new_act), axis=0)
            log_act = new_act

            new_pos = self.pos + self.vel + action
            win, loose = self.get_result(new_pos)
            win_count += len(win)
            in_game = [i for i in range(self.igc) if not (i in win or i in loose)]
            self.igc = len(in_game)

            features += self.transform(log_img[..., win]) + self.transform(log_img[..., loose])
            labels += self.transform(log_act[:, win], 'win') + self.transform(log_act[:, loose], 'loose')

            self.obs = self.obs[..., in_game]
            self.goal = self.goal[..., in_game]
            self.vel = new_pos[..., in_game] - self.pos[..., in_game]
            self.pos = new_pos[..., in_game]
            current = self.get_img(self.bg[..., in_game].copy())
            previous = np.moveaxis(log_img[-1, ..., d:, in_game], 0, -1)

            new_img = np.concatenate((previous, current), axis=2).reshape(1, w, h, 2 * d, self.igc)
            log_img = np.concatenate((log_img[..., in_game], new_img), axis=0)
            log_act = log_act[:, in_game]

        return np.moveaxis(features, 0, -1), np.moveaxis(labels, 0, -1), win_count / n

    def train(self, parameters, alpha, batch_size, epoch_count, print_length):

        time_point = time.time()
        stats = {'mean': [], 'min': [], 'max': []}
        win_rates = []

        for epoch in range(1, epoch_count + 1):

            features, labels, win_rate = self.make_a_batch(parameters, batch_size)
            if 0 < features.size + labels.size:
                gradients = backward(parameters, features, labels)
                parameters = update_parameters(parameters, gradients, alpha)
            win_rates.append(win_rate)

            if not epoch % print_length:

                a = np.array(win_rates)
                m = np.mean(a)
                stats['min'].append(np.min(a))
                stats['mean'].append(m)
                stats['max'].append(np.max(a))
                win_rates = []
                print("Epoch {} : {} %".format(epoch, round(m * 100, 2)))

        delta = time.gmtime(time.time() - time_point)
        print("Finished in {0} hour(s) {1} minute(s) {2} second(s).".format(delta.tm_hour, delta.tm_min, delta.tm_sec))
        show_stats(stats, (np.arange(epoch_count // print_length) + 1) * print_length)

        return parameters

    def numpy_to_pygame(self, source, size, distribution):
        array = source.reshape(self.dims[:2]) * 255
        new_pos = self.pos + self.vel
        ac = self.actions.shape[0]

        red = min(distribution)
        green = max(distribution)
        delta = green - red
        map_color = lambda z: pygame.Color(int(255 * (green - z) / delta), int(255 * (z - red) / delta), 0)

        surface = pygame.surfarray.make_surface(array)

        for k in range(ac):
            p = new_pos + self.actions[k].reshape(2, 1, 1)
            if self.in_screen(p):
                x, y = p.reshape(-1)
                surface.set_at((x, y), map_color(distribution[k]))

        return pygame.transform.scale(surface, size)

    def play(self, parameters, count):

        w, h, d = self.dims
        u = self.unit
        self.igc = 1

        pygame.init()
        img_size = (w * u, h * u)
        screen = pygame.display.set_mode(img_size)

        for _ in range(count):

            self.obs, self.goal, self.pos, self.vel = get_map(1, w, h)
            self.bg = self.get_bg()
            life = 0

            previous = self.get_img(self.bg.copy())
            current = previous
            game_state = 'in_game'

            prob = forward(parameters, np.concatenate((previous, current), axis=2))
            img = self.numpy_to_pygame(current, img_size, list(prob))

            while life < self.life_time and game_state == 'in_game':

                screen.blit(img, (0, 0))

                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONDOWN:

                        life += 1
                        # a = [int(np.random.choice(9, 1, p=p)) for p in prob.T]
                        a = np.argmax(prob, axis=0)
                        action = self.actions[a].T.reshape(2, 1, 1)

                        new_pos = self.pos + self.vel + action
                        win, loose = self.get_result(new_pos)

                        if 0 < len(win):
                            game_state = 'win'
                        elif 0 < len(loose):
                            game_state = 'loose'
                        else:
                            self.vel = new_pos - self.pos
                            self.pos = new_pos
                            previous = current
                            current = self.get_img(self.bg.copy())
                            prob = forward(parameters, np.concatenate((previous, current), axis=2))
                            img = self.numpy_to_pygame(current, img_size, list(prob))

                pygame.display.flip()
            print(game_state)
        pygame.quit()
