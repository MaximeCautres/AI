import numpy as np

w = 5 + 2
h = 7 + 2
p = 0.1

begin = (1, 1)
goal = (h-2, w-2)

maps = np.array([[not (np.random.random() < p or i == 0 or i == w-1 or j == 0 or j == h-1) for i in range(w)] for j in range(h)])
actions_set = [(0, 1), (-1, 0), (0, -1), (1, 0)]

states_count = w * h
actions_count = len(actions_set)

actions = np.zeros((states_count, actions_count))
transitions = np.zeros((states_count, actions_count, states_count))
rewards = np.zeros((states_count, actions_count, states_count))

for j in range(1, h-1):
    for i in range(1, w-1):
        for k in range(actions_count):
            y, x = actions_set[k]
            actions[j * w + i, k] = maps[j + y, i + x]
            transitions[j * w + i, k, (j + y) * w + i + x] = maps[j + y, i + x]
            rewards[j * w + i, k, (j + y) * w + i + x] = -1

# print(maps)
# print(rewards[33])
# print(transitions[33])
# print(rewards[(h-3) * w + w - 2, 1, (h-2) * w + w - 2])
