import numpy as np

states_count = 6
actions_count = 6
step_count = 6
keep_count = 6

alpha = 0.9
gamma = 0.8


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
    state_prime = np.random.choice(states_count, p=world['transitions'][state, action])
    instant_reward = world['rewards'][state, action, state_prime]

    agent['current'] = state_prime
    action_prime = policy(state_prime, current_values)
    new = instant_reward + gamma * expected(current_values, state_prime, action_prime)
    agent['values'][state, action, state_prime] *= (1 - alpha)
    agent['values'][state, action, state_prime] += alpha * new


world = generate_world()
agent = initialize_agent()
current_values = None

for t in range(step_count):
    if not t % keep_count:
        current_values = agent['values']
    make_a_step()
