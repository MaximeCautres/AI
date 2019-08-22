""" Main program """

import deep_Q_network as dqn

" Set hyper-parameters "

dnn_topology = {'Ld': 2,
                'nc0': 5, 'nc1': 5, 'nc2': 5,
                'afd1': 'sigmoid', 'afd2': 'softmax'
                }

reward = 'cos_theta**5'  # take x, x_p, cos_theta, sin_theta, theta_p
episode_count = 512
period_count = 256
period_length = 0.1
replay_count = 128
alpha = '10**-1'  # learning_rate(t)
beta = '0.95'  # momentum_rate(t)
gamma = 0.95  # discount factor
delta = 0.95  # Bellman factor
epsilon = 'max(1-2*t, 0)'  # greedy(t)
s0 = [0, 0, -1, 0, 0]  # [x, x_p, cos(theta), sin(theta), theta_p]
view_episode = [128, 256, 384, 512]

" Get parameters "

parameters = dqn.initialize_parameters(dnn_topology)

arg = (parameters, reward, episode_count, period_count, period_length,
       replay_count, view_episode, s0, alpha, beta, gamma, delta, epsilon)
parameters = dqn.train(*arg)

hyper_parameters = {'dnn_topology': dnn_topology, 'episode_count': episode_count}
save = {'parameters': parameters, 'hyper_parameters': hyper_parameters}
