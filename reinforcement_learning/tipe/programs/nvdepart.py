from random import randrange
from convolutional_neural_network import *
import numpy as np
import pickle
import time
from copy import deepcopy

class Buffer:

    """
    On va stocker ici tout les time step d'une epoque, epoque en nombre d'expérience et non en nombre de partie
    """

    def __init__(self, map_dimension, action_dim, size, gamma = 0.99, lam = 0.95):


        # Le 1 a chaque fois a la fin car on prévoit le vectorise maintenant
        
        # Initialisation du buffer pour les inputs
        self.obs_buf = np.zeros(Buffer.combined_shape(size, obs_dim), dtype = np.float32)

        # Initialisation du Buffer pour les actions
        self.act_buf = np.zeros((size, 1), dtype=np.float32)
        
        # Initialisation du Buffer pour les avantages
        self.adv_buf = np.zeros((size, 1), dtype=np.float32)
        
        # Initialisation du Buffer pour les rewards
        self.rew_buf = np.zeros((size, 1), dtype=np.float32)
        
        # Initialisation du Buffer pour les la proba en logarithme de l'action choisie
        self.logp_buf = np.zeros((size, action_dim, 1), dtype=np.float32)
        
        # Gamma and lam to compute the advantage
        self.gamma, self.lam = gamma, lam
        # ptr: Position to insert the next tuple
        # path_start_idx Posittion of the current trajectory
        # max_size Max size of the buffer
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

        
    @staticmethod
    def combined_shape(length, shape = None):
        if shape == None:
            return (length,)
        else:
            return (length, shape) if np.isscalar(shape) else (length, *shape)

        
    def store(self, obs, act, rew, logp):
        """
        Ajoute un coup au buffer
        """
        
        assert self.ptr < max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.logp_buf[self.ptr] = logp
        self.ptr += 1


    def finish_path(self, last_val = 0):
        self.adv_buf[self.ptr - 1] = last_val
        for ind in reverse(range(self.path_start_idx, self.ptr - 1)):
            self.adv_buf[ind] = self.adv_buf[ind + 1] * self.gamma + self.rew_buf[ind]
        self.path_start_idx = self.ptr

    def get(self):
        """
        Return le buffer en mode vectorise si il est plein
        """
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        # Si par plaisir on peut centrer et reduire la fonction avantage
        self.adv_buf = (self.adv_buf - np.mean(self.adv_buf)) / np.std(self.adv_buf)
        obs_buf_vecto = np.swapaxes(self.obs_buf, 0, -1).reshape((*self.obs_buf.shape[1:-1], -1))
        act_buf_vecto = np.swapaxes(self.act_buf, 0, -1).reshape((*self.act_buf.shape[1:-1], -1))
        adv_buf_vecto = np.swapaxes(self.adv_buf, 0, -1).reshape((*self.adv_buf.shape[1:-1], -1))
        logp_buf_vecto = np.swapaxes(self.logp_buf, 0, -1).reshape((*self.logp_buf.shape[1:-1], -1))
        
        return obs_buf_vecto, act_buf_vecto, adv_buf_vecto, logp_buf_vecto
        
        

class Policy_Gradient():
    """
    L'implementation de l'algorithme du Policy Gradient
    """
    def __init__(self, parameters, buffer_size, seed):

        self.parameters = parameters # parametres de réseaux de neuronnes de convolution: Topo + weight
        self.input_dim = self.parameters['idi'] # Dimension de la layer d'entree
        self.ac = parameters['tod'][-1] # Action count
        self.seed = seed

        #Initialisation du buffer
        self.buffer = Buffer(
            obs_dim = self.input_dim,
            act_dim = self.ac,
            size = buffer_size)

        #le learning rate
        self.pi_lr = pi_lr 

        # permet de biaiser l'aleatoire numpy de telle façon que les paramètres
        # généré soit tjrs les mêmes, nécessaire pour la reproductibilité des expérimentations
        np.random.seed(42)

    # Donne l'action en effectuant un forward du réseaux de neurones 
    def step(self, states):
        # Take actions given the states
        # Return mu (policy without exploration), pi (policy with the current exploration) and
        # the log probability of the action chossen by pi
        output = forward(parameters, environnement.gen_state())
        mu, pi = np.argmax(output), np.random.choice(self.ac, p = output)
        logpi = np.log(output[pi])
        return mu, pi, logp_pi


    def store(self, obs, act, rew, logp):
        # Store the observation, action, reward and the log probability of the action
        # into the buffer
        self.buffer.store(obs, act, rew, logp)

        
    def finish_path(self, last_val=0):
        self.buffer.finish_path(last_val=last_val)


    def train(self, additional_infos={}): #TO DO il faut mettre les mains dans la merde maintenant
        # Get buffer
        obs_buf, act_buf, adv_buf, logp_last_buf = self.buffer.get()
        # Train the model
        pi_loss_list = []
        entropy_list = []

        for step in range(5):
            _, entropy, pi_loss = self.sess.run([self.train_pi, self.approx_ent, self.pi_loss], feed_dict= {
                self.tf_map: obs_buf,
                self.tf_a:act_buf,
                self.tf_adv: adv_buf
            })

            pi_loss_list.append(pi_loss)
            entropy_list.append(entropy)

        print("Entropy : %s, Loss: %s" % (np.mean(entropy_list), np.mean(pi_loss_list)), end="\r")

"""
Super important ça mère de ouf:
pour passer la dimension de stack de 0 a la derniere comme on stack en zero dans le buffer et il faut la dernière pour la vectorisation:

Il faut ajouter la dernière dimension a 1 comme si on avait 1 seul élément dans la vectorisation
np.swapaxes(test, 0, -1).reshape((*test.shape[1:-1], -1))

(6, 2, 4, 1, 1) -> (2, 4, 1, 6)

et conserve  les relations

"""











        
        

class Environnement(object):
     
    def __init__(self):
        super(Environnement, self).__init__()
        
        self.dimension = (5, 5)
        self.action_count = 4
        
        self.rewards = [[1, 1, 0, 0, 0],
                       [1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, -1],
                       [0, 0, 0, -1, -1]]
        
        self.begin_pos = [2, 2]
        
        self.position = deepcopy(self.begin_pos)  # y, x
        
         
    def gen_state(self):
        state = np.copy(self.rewards)
        state[self.position[0], self.position[1]] = 2
        state = np.around((state + 1)/3, 1).reshape((*self.dimension, 1, 1))
        return state
    
    
    def step(self, action):
        
        if action == 0: # y vers le haut
            self.position[0] += 1
        elif action == 1: # x vers la droite
            self.position[1] += 1
        elif action == 2: # y vers le bas
            self.position[0] -= 1
        else: # x vers la gauche
            self.position[1] -= 1
            
        inside =  (0 <= self.position[0] < self.dimension[0]) and (0 <= self.position[1] < self.dimension[1])
        
        if inside:
            rewards = self.rewards[self.position[0]][self.position[1]]
        else:
            rewards = -1

        done = False if rewards == 0 else True

        if done:
            self.position = deepcopy(self.begin_pos)

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
                elif (y == self.position[0] and x == self.position[1]):
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
    import_file = True # True => on fait une demonstration, False => on entraine
    begin_with = False # True =>, on contine l'entrainement sur file_name, False => On en commence un nouveau

    environnement = Environnement()

    # Si l'on souhaite simplement executer un cerveau déjà entrainer pour voir ses performance
    
    if import_file:
        parameters = pickle.load(open(file_name, 'rb'))
        
        environnement.display()
        
        for k in range(100):
            action = np.argmax(forward(parameters, environnement.gen_state()))
            print(k)
            #action = randrange(0, 4)
            environnement.step(action)
            time.sleep(0.5)
            environnement.display()

    # Si l'on souhaite entrainer de nouveau un cerveau ou un nouveaux cerveau
            
    else:

        #on génère les paramètre où on les récupère
        
        if begin_with:
            parameters = pickle.load(open(file_name, 'rb'))
        else:
            parameters = {'Lc': 0, 'idi': environnement.dimension + (1,),
                          'lt1': 'c', 'kd1': (3, 3, 4),
                          'lt2': 'p', 'ss2': (2, 2), 'pd2': (3, 3),
                          'Ld': 3, 'tod': (5*5*1, 128, environnement.action_count)}
            parameters = initialize_parameters(parameters)

        pickle.dump(parameters, open(file_name, 'wb'))

        """
        # Policy(parameters, seed, buffer_size, learning rate)
            
        agent = Policy_Gradient(parameters, 42, 10**4, 0.001)

        rewards = []
        
        for epoch in range(10000):

            done = False
            state = grid.gen_state()
            
            while not done:
                _, pi, logpi = agent.step([state])
                n_state, reward, done = grid.step(pi[0])
                agent.store(state, pi[0], reward, logpi)
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
                    #agent.train()
                    b = 0

            if epoch % 1000 == 0:
                print("Rewards mean:%s" % np.mean(rewards))

            
        
        pickle.dump(parameters, open(file_name, 'wb'))
    """
    
if __name__ == '__main__':
    main()
