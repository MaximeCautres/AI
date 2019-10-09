from tkinter import *
import numpy as np

knuth = 10 ** 6
unit = 34


class Mdp:

    def __init__(self, maps, actions, transitions, rewards, starting_state, alpha, delta, gamma):

        self.s_count = transitions.shape[0]
        self.a_count = actions.shape[1]
        self.dim = maps.shape
        self.start = starting_state
        self.alpha = alpha
        self.delta = delta
        self.gamma = gamma

        state = self.start
        values = np.random.random((self.s_count, self.a_count, self.s_count))

        self.agent = {'state': state, 'values': values}
        self.world = {'actions': actions, 'transitions': transitions, 'rewards': rewards}
        self.current = None
        self.t = 0

        self.canvas = self.canvas = Canvas(window, width=self.dim[1]*unit, height=self.dim[0]*unit, background='black')
        self.canvas.pack(side=LEFT, padx=1, pady=1)

        self.maps = maps
        self.grid = np.array([[self.canvas.create_rectangle(unit*x, unit*y, unit*(x+1), unit*(y+1), fill=self.get_color(y, x))
                               for x in range(self.dim[1])] for y in range(self.dim[0])])

        self.change_color(self.grid[self.state_to_coord(self.agent['state'])], 'blue')

        self.canvas.bind('<Button-1>', self.make_a_step)
        self.canvas.bind('<Button-3>', self.re_spawn)

        self.train()

    def expected(self, vs, s, a):
        return np.sum(np.multiply(self.world['transitions'][s, a], vs[s, a]))

    def policy(self, vs, s):
        decision, best = None, -knuth
        for a in range(self.a_count):
            if self.world['actions'][s, a]:
                value = self.expected(vs, s, a)
                if best < value:
                    decision, best = a, value
        return decision

    def make_a_step(self, event=None):
        if not self.t % self.delta:
            self.current = self.agent['values']
        self.t += 1

        state = self.agent['state']
        action = self.policy(self.current, state)
        state_prime = np.random.choice(self.s_count, p=self.world['transitions'][state, action])
        instant_reward = self.world['rewards'][state, action, state_prime]

        # print(self.state_to_coord(state), action, instant_reward)

        self.agent['state'] = state_prime
        action_prime = self.policy(self.current, state_prime)
        new = instant_reward + self.gamma * self.expected(self.current, state_prime, action_prime)
        self.agent['values'][state, action, state_prime] *= (1 - self.alpha)
        self.agent['values'][state, action, state_prime] += self.alpha * new

        self.update_color(state, state_prime)

    def train(self):
        for _ in range(10**3):
            for _ in range(2 * int(self.s_count ** 0.5)):
                self.make_a_step()
            self.re_spawn()

    def re_spawn(self, event=None):
        self.update_color(self.agent['state'], self.start)
        self.agent['state'] = self.start

    def state_to_coord(self, s):
        return s // self.dim[1], s % self.dim[1]

    def get_color(self, y, x):
        return 'white' * self.maps[y, x] + 'black' * (not self.maps[y, x])

    def change_color(self, tag, nc):
        self.canvas.itemconfigure(tag, fill=nc)

    def update_color(self, prev, new):
        y, x = self.state_to_coord(prev)
        self.change_color(self.grid[y, x], self.get_color(y, x))
        self.change_color(self.grid[self.state_to_coord(new)], 'blue')


def generate_world_mouse(height, width, p):
    h = 1 + height + 1
    w = 1 + width + 1

    begin = w + 1
    goal = (h - 2, w - 2)

    maps = np.array([[not (np.random.random() < p or i == 0 or i == w - 1 or j == 0 or j == h - 1)
                      for i in range(w)] for j in range(h)])
    actions_set = [(0, 1), (-1, 0), (0, -1), (1, 0)]

    states_count = h * w
    actions_count = len(actions_set)

    actions = np.zeros((states_count, actions_count))
    transitions = np.zeros((states_count, actions_count, states_count))
    rewards = np.zeros((states_count, actions_count, states_count))

    for j in range(1, h - 1):
        for i in range(1, w - 1):
            for k in range(actions_count):
                y, x = actions_set[k]
                actions[j * w + i, k] = int(maps[j + y, i + x])
                transitions[j * w + i, k, (j + y) * w + i + x] = int(maps[j + y, i + x])
                rewards[j * w + i, k, (j + y) * w + i + x] = ((j + y, i + x) == goal) * states_count - 1

    return maps, actions, transitions, rewards, begin


window = Tk()
window.title('I like trains !')

my_mdp_one = Mdp(*generate_world_mouse(24, 24, 0.2), 0.9, 6, 0.8)


window.mainloop()
