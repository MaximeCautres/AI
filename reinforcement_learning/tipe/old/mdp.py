import matplotlib.pyplot as plt
from tkinter import *
import numpy as np

training = True
knuth = 10 ** 6
unit = 32


class Mdp:

    def __init__(self, maps, actions, transitions, rewards, begend, alpha=0.9, delta=6, gamma=0.8):

        self.maps = maps
        self.s_count = transitions.shape[0]
        self.a_count = actions.shape[1]
        self.dim = maps.shape
        self.start, self.end = begend
        self.alpha = alpha
        self.delta = delta
        self.gamma = gamma

        state = self.start
        values = np.random.random((self.s_count, self.a_count, self.s_count))

        self.agent = {'state': state, 'values': values}
        self.world = {'actions': actions, 'transitions': transitions, 'rewards': rewards}
        self.current, self.life_time, self.scores = (None, ) * 3
        self.t = 0

        self.canvas, self.grid = (None, ) * 2
        self.visual()

    def visual(self):
        if training:
            self.reset()
        else:
            self.canvas = Canvas(window, width=self.dim[1]*unit, height=self.dim[0]*unit)
            self.canvas.pack(side=LEFT, padx=1, pady=1)
        
            self.grid = np.array([[self.canvas.create_rectangle(unit*x, unit*y, unit*(x+1), unit*(y+1),
                                                                outline='white', fill=self.get_color(y, x))
                                   for x in range(self.dim[1])] for y in range(self.dim[0])])
        
            self.change_color(self.grid[self.state_to_coord(self.start)], 'blue')
        
            self.canvas.bind('<MouseWheel>', self.make_a_step)
            self.canvas.bind('<Button-3>', self.re_spawn)

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

        action_prime = self.policy(self.current, state_prime)
        new = instant_reward + self.gamma * self.expected(self.current, state_prime, action_prime)
        self.agent['values'][state, action, state_prime] *= (1 - self.alpha)
        self.agent['values'][state, action, state_prime] += self.alpha * new
        self.agent['state'] = state_prime

        if not training:
            self.update_color(state, state_prime)

    def make_a_life(self):
        step_count = 0
        while step_count < self.life_time:
            step_count += 1
            self.make_a_step()
            if self.state_to_coord(self.agent['state']) == self.end:
                self.life_time = step_count
        self.scores.append(step_count)
        self.re_spawn()

    def reset(self):
        self.agent['values'] = np.random.random((self.s_count, self.a_count, self.s_count))
        self.life_time = 3 * (self.dim[0] * self.dim[1]) ** 0.5
        self.scores = []

    def re_spawn(self, event=None):
        if not training:
            self.update_color(self.agent['state'], self.start)
        self.agent['state'] = self.start

    def state_to_coord(self, s):
        return s // self.dim[1], s % self.dim[1]

    def get_color(self, y, x):
        return 'black' * self.maps[y, x] + 'white' * (not self.maps[y, x])

    def change_color(self, tag, nc):
        self.canvas.itemconfigure(tag, fill=nc)

    def update_color(self, prev, new):
        y, x = self.state_to_coord(prev)
        self.change_color(self.grid[y, x], self.get_color(y, x))
        self.change_color(self.grid[self.state_to_coord(new)], 'blue')


def generate_mouse_world(height, width, p):
    h = 1 + height + 1
    w = 1 + width + 1

    begin = w + 1
    goal = (h - 2, w - 2)

    maps = np.array([[not (np.random.random() < p or i == 0 or i == w-1 or j == 0 or j == h-1)
                      or (0 < i < 4 and 0 < j < 4) or (w-5 < i < w-1 and h-5 < j < h-1)
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

    return maps, actions, transitions, rewards, (begin, goal)


agent_trained = None
world = generate_mouse_world(32, 48, 0.2)

if training:
    mdp_train = Mdp(*world)
    # parameters = [1, 2, 4, 8, 16, 32]
    parameters = [6]
    n = len(parameters)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    graph = []
    for parameter in parameters:
        mdp_train.reset()
        mdp_train.delta = parameter
        for _ in range(2560):
            mdp_train.make_a_life()
        graph.append(mdp_train.scores)
    for l in range(n):
        plt.plot(graph[l], color=colors[l])
    plt.legend(parameters)
    plt.ylabel("Life time")
    plt.xlabel("Iteration")
    plt.show()

    agent_trained = mdp_train.agent.copy()
    training = False

window = Tk()
window.title('I like train !')

my_mdp = Mdp(*world)

if agent_trained:
    my_mdp.agent = agent_trained

window.mainloop()
