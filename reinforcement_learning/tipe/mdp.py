import numpy as np

alpha = 0.9
gamma = 0.8

w = 5 + 2
h = 7 + 2
p = 0

begin = (1, 1)
goal = (h-2, w-2)

maps = np.array([[not (np.random.random() < p or i == 0 or i == w-1 or j == 0 or j == h-1) for i in range(w)] for j in range(h)])
actions_set = [(0, 1), (-1, 0), (0, -1), (1, 0)]

states_count = h * w
actions_count = len(actions_set)
step_count = 6
keep_count = 6


def generate_world_mouse():
    actions = np.zeros((states_count, actions_count))
    transitions = np.zeros((states_count, actions_count, states_count))
    rewards = np.zeros((states_count, actions_count, states_count))

    for j in range(1, h - 1):
        for i in range(1, w - 1):
            for k in range(actions_count):
                y, x = actions_set[k]
                actions[j * w + i, k] = maps[j + y, i + x] * 1
                transitions[j * w + i, k, (j + y) * w + i + x] = maps[j + y, i + x] * 1
                rewards[j * w + i, k, (j + y) * w + i + x] = (((j + y) * w, i + x) == goal) * states_count - 1

    return {'actions': actions, 'transitions': transitions, 'rewards': rewards}


def initialize_agent_mouse():
    current = w + 1
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
    return np.argmax([expected(vs, s, a) for a in range(actions_count) if world['actions'][s, a]])


def make_a_step():
    state = agent['current']
    action = policy(current_values, state)
    print(action)
    state_prime = np.random.choice(states_count, p=world['transitions'][state, action])
    instant_reward = world['rewards'][state, action, state_prime]

    agent['current'] = state_prime
    action_prime = policy(state_prime, current_values)
    new = instant_reward + gamma * expected(current_values, state_prime, action_prime)
    agent['values'][state, action, state_prime] *= (1 - alpha)
    agent['values'][state, action, state_prime] += alpha * new


world = generate_world_mouse()
agent = initialize_agent_mouse()
current_values = None

for t in range(step_count):
    if not t % keep_count:
        current_values = agent['values']
    v = agent['current']
    print((v // w, v % w))
    make_a_step()
