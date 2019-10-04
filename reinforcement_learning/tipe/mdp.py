import numpy as np
import time

alpha = 0.9
gamma = 0.8

epsilon = 10 ** -6
knuth = 10 ** 6

w = 10 + 2
h = 10 + 2
p = 0.3

begin = (1, 1)
goal = (h-2, w-2)

maps = np.array([[not (np.random.random() < p or i == 0 or i == w-1 or j == 0 or j == h-1)
                  for i in range(w)] for j in range(h)])
actions_set = [(0, 1), (-1, 0), (0, -1), (1, 0)]

states_count = h * w
actions_count = len(actions_set)

life_time = 60
life_count = 60
keep_count = 6


def generate_world_mouse():
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

    return {'actions': actions, 'transitions': transitions, 'rewards': rewards}


def initialize_agent_mouse():
    current = None
    values = np.random.random((states_count, actions_count, states_count))
    return {'current': current, 'values': values}


def generate_world():
    actions = np.zeros((states_count, actions_count))
    transitions = np.zeros((states_count, actions_count, states_count))
    rewards = np.zeros((states_count, actions_count, states_count))
    return {'actions': actions, 'transitions': transitions, 'rewards': rewards}


def initialize_agent():
    current = np.random.randint(states_count)
    values = np.random.random((states_count, actions_count, states_count))
    return {'current': current, 'values': values}


def expected(vs, s, a):
    return np.sum(np.multiply(world['transitions'][s, a], vs[s, a]))


def policy(vs, s):
    decision, best = None, -knuth
    for a in range(actions_count):
        if world['actions'][s, a]:
            value = expected(vs, s, a)
            if best < value:
                decision, best = a, value
    return decision


def make_a_step():
    state = agent['current']
    action = policy(current_values, state)
    state_prime = np.random.choice(states_count, p=world['transitions'][state, action])
    instant_reward = world['rewards'][state, action, state_prime]

    print(action, instant_reward)

    agent['current'] = state_prime
    action_prime = policy(current_values, state_prime)
    new = instant_reward + gamma * expected(current_values, state_prime, action_prime)
    agent['values'][state, action, state_prime] *= (1 - alpha)
    agent['values'][state, action, state_prime] += alpha * new


deb = time.time()


world = generate_world_mouse()
agent = initialize_agent_mouse()
current_values = None

print(maps)

for life in range(life_count):
    agent['current'] = begin[0] * w + begin[1]
    print('-' * 16)
    for t in range(life_time):
        if not (life * life_time + t) % keep_count:
            current_values = agent['values']
        print((agent['current'] // w, agent['current'] % w), end=' ')
        make_a_step()

end = time.time()
print(int((end-deb)//60), "minutes", int((end - deb) % 60), "secondes for a", w-2, "x", h-2, "map")
