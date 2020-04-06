from random import randrange
from convolutional_neural_network import *
import numpy as np
import pickle
import time


class Buffer:

    """
    On va stocker ici tout les time step d'une epoque, epoque en nombre d'expérience et non en nombre de partie
    """

    def __init__(self, map_dimension, size, gamma=0.99, lam=0.95):

        # Le 1 a chaque fois a la fin car on prévoit le vectorise maintenant
        
        # Initialisation du buffer pour les inputs
        self.obs_buf = np.zeros(Buffer.combined_shape(size, map_dimension), dtype=np.float32)

        # Initialisation du Buffer pour les actions
        self.act_buf = np.zeros(size, dtype=np.int8)
        
        # Initialisation du Buffer pour les avantages
        self.adv_buf = np.zeros(size, dtype=np.float32)
        
        # Initialisation du Buffer pour les rewards
        self.rew_buf = np.zeros(size, dtype=np.float32)
        
        # Initialisation du Buffer pour les la proba de l'action choisie
        self.p_buf = np.zeros(size, dtype=np.float32)
        
        # Gamma and lam to compute the advantage
        self.gamma, self.lam = gamma, lam
        # ptr: Position to insert the next tuple
        # path_start_idx Posittion of the current trajectory
        # max_size Max size of the buffer
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    @staticmethod
    def combined_shape(length, shape=None):
        if shape is None:
            return (length, )
        else:
            return (length, shape) if np.isscalar(shape) else (length, *shape)

    def store(self, obs, act, rew, p):
        """
        Ajoute un coup au buffer
        """
        
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.p_buf[self.ptr] = p
        self.ptr += 1

    def finish_path(self, last_val=0):
        self.adv_buf[self.ptr - 1] = last_val
        for ind in reversed(range(self.path_start_idx, self.ptr - 1)):
            self.adv_buf[ind] = self.adv_buf[ind + 1] * self.gamma + self.rew_buf[ind]
        self.path_start_idx = self.ptr

    def get(self):
        """
        Return le buffer en mode vectorise si il est plein
        """
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        # Si par plaisir on peut centrer et reduire la fonction avantage
        # self.adv_buf = (self.adv_buf - np.mean(self.adv_buf)) / np.std(self.adv_buf)
        obs_buf_vecto = np.swapaxes(self.obs_buf, 0, -1).reshape((*self.obs_buf.shape[1:-1], -1))
        
        return obs_buf_vecto, self.act_buf, self.adv_buf, self.p_buf
        

class Policy_Gradient():
    """
    L'implementation de l'algorithme du Policy Gradient
    """
    def __init__(self, parameters, buffer_size, pi_lr, seed):

        self.parameters = parameters  # parametres de réseaux de neuronnes de convolution: Topo + weight
        self.input_dim = self.parameters['idi']  # Dimension de la layer d'entree
        self.ac = parameters['tod'][-1]  # Action count
        self.seed = seed
        self.pi_lr = pi_lr

        # Initialisation du buffer
        self.buffer = Buffer(
            map_dimension=self.input_dim,
            size=buffer_size)

        # le learning rate
        self.pi_lr = pi_lr 

        # permet de biaiser l'aleatoire numpy de telle façon que les paramètres
        # généré soit tjrs les mêmes, nécessaire pour la reproductibilité des expérimentations
        np.random.seed(seed)

    # Donne l'action en effectuant un forward du réseaux de neurones 
    def step(self, states):
        # Take actions given the states
        # Return mu (policy without exploration), pi (policy with the current exploration) and
        # the log probability of the action chossen by pi
        output = forward(self.parameters, states)
        np.nan_to_num(output, False)
        output[0][0] += 1 - np.sum(output)
        mu, pi = np.argmax(output), np.random.choice(self.ac, p=output.reshape(-1))
        p_pi = output[pi]
        return mu, pi, p_pi

    def store(self, obs, act, rew, p):
        # Store the observation, action, reward and the log probability of the action
        # into the buffer
        self.buffer.store(obs, act, rew, p)

    def finish_path(self, last_val=0):
        self.buffer.finish_path(last_val=last_val)

    def train(self):
        # Get buffer, ici il est deja en mode ok vectorisation
        obs_buf, act_buf, adv_buf, p_buf = self.buffer.get()
        # Train the model

        da = np.zeros((self.ac, self.buffer.max_size))

        da[act_buf, np.arange(self.buffer.max_size)] = adv_buf / p_buf  # adv_buf / p_buf

        gradients = backward(self.parameters, obs_buf, da)

        self.parameters = update_parameters(self.parameters, gradients, self.pi_lr)

        # print("Loss:", np.mean(np.log(p_buf)*adv_buf)) # , end="\r"



"""
Super important ça mère de ouf:
pour passer la dimension de stack de 0 a la derniere comme on stack en zero dans le buffer et il faut la dernière pour 
la vectorisation:

Il faut ajouter la dernière dimension a 1 comme si on avait 1 seul élément dans la vectorisation
np.swapaxes(test, 0, -1).reshape((*test.shape[1:-1], -1))

(6, 2, 4, 1, 1) -> (2, 4, 1, 6)

et conserve  les relations

"""


class Environnement(object):
     
    def __init__(self):
        super(Environnement, self).__init__()
        
        self.dimension = (25, 25)
        self.action_count = 4

        """
        self.rewards = [[1, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, -1],
                       [0, 0, 0, -1, -1]] """

        def maps():
            monde = np.zeros(self.dimension)
            monde[5:20, 4:7] -= 1
            monde[5:20, 18:21] -= 1
            monde[10:15, 10:15] += 1
            return monde

        self.rewards = maps()

        def beginpos():
            y, x = randrange(self.dimension[0]), randrange(self.dimension[1])
            while self.rewards[y][x] != 0:
                y, x = randrange(self.dimension[0]), randrange(self.dimension[1])
            return [y, x]
        
        self.position = beginpos()  # y, x

    def beginpos(self):
        y, x = randrange(self.dimension[0]), randrange(self.dimension[1])
        while self.rewards[y][x] != 0:
            y, x = randrange(self.dimension[0]), randrange(self.dimension[1])
        return [y, x]

    def gen_state(self):
        state = np.copy(self.rewards)
        state[self.position[0], self.position[1]] = 2
        state = np.around((state + 1)/3, 1).reshape((*self.dimension, 1, 1))
        return state

    def step(self, action):
        
        if action == 0:  # y vers le haut
            self.position[0] += 1
        elif action == 1:  # x vers la droite
            self.position[1] += 1
        elif action == 2:  # y vers le bas
            self.position[0] -= 1
        else:  # x vers la gauche
            self.position[1] -= 1
            
        inside = (0 <= self.position[0] < self.dimension[0]) and (0 <= self.position[1] < self.dimension[1])
        
        if inside:
            rewards = self.rewards[self.position[0]][self.position[1]]
        else:
            rewards = -1

        done = False if rewards == 0 else True

        if done:
            self.position = self.beginpos()

        state = self.gen_state()

        return state, rewards, done

    def display(self):
        y = 0
        print("="*14)
        for line in self.rewards:
            x = 0
            for case in line:
                if case == -1:
                    c = "0"
                elif y == self.position[0] and x == self.position[1]:
                    c = "A"
                elif case == 1:
                    c = "T"
                else:
                    c = "-"
                print(c, end=" ")
                x += 1
            y += 1
            print()


"""Pannel de controle de l'algorithme"""


def main():
    
    file_name = 'nvdepart_0'
    import_file = False  # True => on fait une demonstration, False => on entraine
    begin_with = False  # True =>, on contine l'entrainement sur file_name, False => On en commence un nouveau
    buffer_size = 10**2
    epoch_count = 10**5

    environnement = Environnement()

    # Si l'on souhaite simplement executer un cerveau déjà entrainer pour voir ses performance
    
    if import_file:
        parameters = pickle.load(open(file_name, 'rb'))
        
        environnement.display()
        
        for k in range(100):
            action = np.argmax(forward(parameters, environnement.gen_state()))
            print(k)
            # action = randrange(0, 4)
            environnement.step(action)
            time.sleep(0.5)
            environnement.display()

    # Si l'on souhaite entrainer de nouveau un cerveau ou un nouveaux cerveau
            
    else:

        # on génère les paramètre où on les récupère
        
        if begin_with:
            parameters = pickle.load(open(file_name, 'rb'))
        else:
            parameters = {'Lc': 0, 'idi': environnement.dimension + (1, 1),
                          'lt1': 'c', 'kd1': (3, 3, 4),
                          'lt2': 'p', 'ss2': (2, 2), 'pd2': (3, 3),
                          'Ld': 3, 'tod': (25*25*1, 128, environnement.action_count)}
            parameters = initialize_parameters(parameters, 42)

        pickle.dump(parameters, open(file_name, 'wb'))

        # Policy(parameters, buffer_size, learning rate, seed)
            
        agent = Policy_Gradient(parameters, buffer_size, 0.1, 42)

        rewards = []

        b = 0
        
        for epoch in range(epoch_count):

            done = False
            state = environnement.gen_state()
            
            while not done:
                _, pi, p_pi = agent.step(state)
                n_state, reward, done = environnement.step(pi)
                agent.store(state, pi, reward, p_pi)
                b += 1
                
                state = n_state
                
                if done:
                    agent.finish_path(reward)
                    rewards.append(reward)
                    if len(rewards) > 1000:
                        rewards.pop(0)
                if b == buffer_size:
                    if not done:
                        agent.finish_path(0)
                    agent.train()
                    b = 0

            if epoch % 1000 == 0:
                print("Rewards mean:%s" % np.mean(rewards))

        pickle.dump(parameters, open(file_name, 'wb'))
        
    
if __name__ == '__main__':
    main()
