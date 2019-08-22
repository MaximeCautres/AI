""" Function for Deep Q Network """

import numpy as np
import physic as pp
from math import cos, sin

actions = [-6, -1, 0, 1, 6]
parameter_topology = ['w', 'b']
clipping_rate = 1


def initialize_parameters(topology):
    """
    Generate parameters based on topology

    Take :
    topology -- dictionary

    Return :
    parameters -- dictionary containing all the information about the whole network
    """

    parameters = {**topology}

    for l in range(1, topology['Ld']+1):

        parameters['w'+str(l)] = np.random.randn(topology['nc' + str(l)], topology['nc' + str(l-1)])
        parameters['pw'+str(l)] = np.zeros((topology['nc' + str(l)], topology['nc' + str(l - 1)]))

        parameters['g'+str(l)] = np.zeros((topology['nc' + str(l)], 1))
        parameters['pg' + str(l)] = np.zeros((topology['nc' + str(l)], 1))

        parameters['b'+str(l)] = np.zeros((topology['nc' + str(l)], 1))
        parameters['pb' + str(l)] = np.zeros((topology['nc' + str(l)], 1))

    return parameters


def convert(s, v):
    """
    Convert a string into a lambda function

    Take :
    s -- state
    v -- variables

    Return :
    f -- lambda function
    """

    return eval('lambda ' + v + ': ' + s)


def train(parameters, reward, episode_count, period_count, period_length,
          replay_count, view_episode, s0, alpha, beta, gamma, delta, epsilon):
    """
    Train parameters in order to have good result at STICK

    Take :
    parameters -- dictionary containing all the information about the whole network
    reward -- the reward function (say what is good and what isn't)
    episode_count -- the number of episode in the training
    period_count -- the number of period in each episode
    period_length -- the length in second of each period
    replay_count -- the number of experience replay stored
    view_episode -- which episode we want to see animated
    s0 -- the initialized state
    alpha -- the learning rate
    beta -- momentum rate
    gamma -- the discount factor
    delta -- the Bellman factor
    epsilon -- the greedy rate

    Return :
    parameters -- dictionary containing all the information about the whole network
    """

    alpha = convert(alpha, 't')
    beta = convert(beta, 't')
    epsilon = convert(epsilon, 't')
    reward = convert(reward, 'x, x_p, cos_theta, sin_theta, theta_p')

    bl = period_count + replay_count
    history = {'states': np.zeros((2, len(s0), bl)), 'action': np.zeros((bl, ), dtype=np.int)}
    episodes_save = []

    for episode in range(episode_count):

        s = np.array(s0).reshape(-1, 1)
        theta = np.pi
        t = episode / episode_count
        episode_save = np.array([[s[0, 0], s[1, 0], theta, s[4, 0]]])

        for period in range(period_count):

            a_k = chose_action(parameters, epsilon(t), s)

            s_pp = np.array([s[0, 0], s[1, 0], theta, s[4, 0]])
            period_save = pp.simulate(period_length, s_pp, actions[a_k])
            episode_save = np.append(episode_save, period_save[1:], axis=0)
            s_pp = period_save[-1]
            theta = s_pp[2]
            s_p = np.array([s_pp[0], s_pp[1], cos(theta), sin(theta), s_pp[3]]).reshape(-1, 1)

            history['states'][:, :, period] = np.squeeze([s, s_p])
            history['action'][period] = a_k
            s = s_p

        k = period_count + min(episode, replay_count)
        s, s_p = history['states'][:, :, :k]
        a_ks = history['action'][:k]

        parameters = learn(parameters, s, a_ks, s_p, alpha(t), beta(t), gamma, delta, reward)

        index = np.random.randint(period_count)
        k = period_count + episode % replay_count
        history['states'][:, :, k] = history['states'][:, :, index]
        history['action'][k] = history['action'][index]

        if (episode+1) in view_episode:
            episodes_save.append((episode_save, episode+1))

    pp.show(episodes_save)

    return parameters


def chose_action(parameters, epsilon, s):
    """
    Choose a random action or the best one knowing the state s

    Take :
    parameters -- dictionary containing all the information about the whole network
    epsilon -- the greedy rate
    s -- states

    Return :
    a -- actions
    """

    if np.random.random() < epsilon:
        a_k = np.random.randint(len(actions))
    else:
        a_k = take_best_action(parameters, s)

    return a_k


def take_best_action(parameters, s):
    """
    Return the best actions to do in state s

    Take :
    parameters -- dictionary containing all the information about the whole network
    s -- states

    Return :
    a -- actions
    """

    q_v = evaluate(parameters, s)['x'+str(parameters['Ld'])]

    return np.squeeze(np.argmax(q_v, axis=0))


def evaluate(parameters, s):
    """
    Evaluate Q-value for each action in state s

    Take :
    parameters -- dictionary containing all the information about the whole network
    s -- states

    Return :
    cache -- dictionary, the log of the forward
    """

    x = s
    cache = {'x0': x}

    for l in range(1, parameters['Ld']+1):

        w, b = parameters['w'+str(l)], parameters['b'+str(l)]
        z = np.dot(w, x) + b

        af = parameters['afd' + str(l)]
        if af == 'relu':
            x = relu(z)
        elif af == 'sigmoid':
            x = sigmoid(z)
        elif af == 'softmax':
            x = softmax(z)
        else:
            x = z

        cache['z'+str(l)] = z
        cache['x'+str(l)] = x

    return cache


def learn(parameters, s, a_k, s_n, alpha, beta, gamma, delta, reward):
    """
    Backpropagate in order to optimize parameters

    Take :
    parameters -- dictionary containing all the information about the whole network
    s -- states
    a_k -- array of action indexes
    s_n -- s after a
    alpha -- the learning rate
    beta -- momentum rate
    gamma -- the discount factor
    delta -- the Bellman factor
    reward -- the reward function (say what is good and what isn't)

    Return :
    parameters -- dictionary containing all the information about the whole network
    """

    cache = evaluate(parameters, s)

    gradients = {}
    q_v = cache['x' + str(parameters['Ld'])]
    dq = bellman(parameters, reward, q_v, s_n, a_k, gamma, delta)
    dx = clip(dq, clipping_rate)

    for l in reversed(range(1, parameters['Ld']+1)):

        af = parameters['afd'+str(l)]
        z = cache['z'+str(l)]
        if af == 'relu':
            dz = dx * relu_prime(z)
        elif af == 'sigmoid':
            dz = dx * sigmoid_prime(z)
        else:
            dz = dx

        x_p = cache['x'+str(l-1)]
        w = parameters['w'+str(l)]

        gradients['dw'+str(l)] = np.dot(dz, x_p.T)
        gradients['db'+str(l)] = np.mean(dz, axis=1, keepdims=True)

        dx = np.dot(w.T, dz)

    parameters = update_parameters(parameters, gradients, alpha, beta)

    return parameters


def bellman(parameters, reward, q_v, s_n, a_k, gamma, delta):
    """
    Return the dq_v in order to learn from the bellman equation

    Take :
    parameters -- dictionary containing all the information about the whole network
    reward -- the reward function (say what is good and what isn't)
    q_v -- the Q-values
    s_n -- s after a
    a_k -- actions
    gamma -- the discount factor
    delta -- the Bellman factor

    Return :
    cache -- dictionary, the log of the forward
    """

    n = q_v.shape[1]
    iter_all = np.arange(n)
    q_v_k = q_v[a_k, iter_all]
    dq_v = np.zeros((len(actions), n))

    x, x_p, cos_theta, sin_theta, theta_p = s_n
    r = reward(x, x_p, cos_theta, sin_theta, theta_p)
    q_v_max = evaluate(parameters, s_n)['x' + str(parameters['Ld'])]

    q_v_p = delta * q_v_k + (1 - delta) * (r + gamma * np.max(q_v_max, axis=0))
    dq_v[a_k, iter_all] = q_v_k - r - gamma * np.max(q_v_max, axis=0)

    print(np.mean((q_v_p - q_v_k) ** 2))

    return dq_v


def update_parameters(parameters, gradients, alpha, beta):
    """
    Update each parameters in order to reduce cost

    Take :
    parameters -- dictionary containing all the information about the whole network
    gradients -- dictionary containing all the gradients of the whole network
    alpha -- the learning rate
    beta -- momentum rate

    Return :
    cache -- dictionary, the log of the forward
    """

    for l in range(1, parameters['Ld']+1):
        for v in parameter_topology:

            d = gradients['d' + v + str(l)]
            p = beta * parameters['p' + v + str(l)] + (1 - beta) * d
            parameters[v + str(l)] -= alpha * p
            parameters['p' + str(l)] = p

    return parameters


def clip(z, max_abs):
    """
    Clip z between -max_abs and max_abs

    Take :
    z -- numpy matrix

    Return :
    clip(z)
    """

    return np.maximum(np.minimum(z, max_abs), -max_abs)


def relu(z):
    """
    Apply the relu function on each element of z

    Take :
    z -- numpy matrix

    Return :
    relu(z)
    """

    return np.maximum(z, 0)


def relu_prime(z):
    """
    Apply the derivative of the relu function on each element of z

    Take :
    z -- numpy matrix

    Return :
    relu_prime(z)
    """

    return z > 0


def sigmoid(z):
    """
    Apply the sigmoid function on each element of z

    Take :
    z -- numpy matrix

    Return :
    sigmoid(z)
    """

    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    """
    Apply the derivative of the sigmoid function on each element of z

    Take :
    z -- numpy matrix

    Return :
    sigmoid_prime(z)
    """

    sigmoid_ = sigmoid(z)

    return sigmoid_ * (1 - sigmoid_)


def softmax(z):
    """
    Apply the softmax function on each element of z

    Take :
    z -- numpy matrix

    Return :
    softmax(z)
    """

    return np.divide(np.exp(z), np.sum(np.exp(z), axis=0, keepdims=True))
