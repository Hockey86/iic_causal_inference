#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 17:31:00 2020

@author: harspari
"""

import fake_sim
import numpy as np
import scipy
import scipy.optimize as opt
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import itertools
import sys
import sklearn.pipeline
import sklearn.preprocessing

from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

# def eval_regime( sim, D, reward, gamma ):
#     E = sim(D)
#     T = len(E)
#     V = np.zeros((T,))
#     V[-1] = reward(E[-1],D[-1])
#     for t in range(T-2,-1,-1):
#         V[t] = reward(E[t],D[t]) + gamma*V[t+1]
#     return V

# sim = lambda D: fake_sim.fake_simulator( fake_sim.params, fake_sim.E_init, D )[1]
# reward = lambda s,a: -s - np.linalg.norm(a)
# gamma = 0.9999

# V_curr = eval_regime(sim, fake_sim.D, reward, gamma)
# print(V_curr[0])

# V0 = lambda D: -1*eval_regime(sim, D.reshape(-1,2), reward, gamma)[0]

# res = opt.minimize( V0, fake_sim.D.reshape(-1,1), method='COBYLA' )
    
# IICopt = sim(res.x.reshape(-1,2))
# print(-V0(res.x))

# p, E, Dc = fake_sim.fake_simulator( fake_sim.params, fake_sim.E_init, fake_sim.D ) 
# p1, E1, Dc1 = fake_sim.fake_simulator( fake_sim.params, fake_sim.E_init, res.x.reshape(-1,2)) 


# fig,ax = plt.subplots(1+Dc1.shape[1],1,sharex=True,figsize=(15,10),gridspec_kw = {'height_ratios':[10]+[1 for i in range(Dc1.shape[1])]})

# ax[0].plot(np.arange(0,Dc1.shape[0]),E1, '--', c='black',label='Optimal Observed')
# ax[0].plot(np.arange(0,Dc.shape[0]),E, c='black',label='Original Observed')

# ax[0].plot(np.arange(0,Dc1.shape[0]),p1, '--', c='b',label='Optimal Probability')
# ax[0].plot(np.arange(0,Dc.shape[0]),p, c='#ff7f0e',label='Original Probability')

# # ax[0].fill_between(x=np.arange(2,T),y1=p.mean(axis=0)[2:]+p.std(axis=0)[2:],y2=p.mean(axis=0)[2:]-p.std(axis=0)[2:],alpha=0.25,color='red')
# ax[0].legend()
# ax[0].set_title('IIC Ratio')
# for i in range(1,1+Dc1.shape[1]):
#     y = Dc1[:,i-1]
#     ax[i].imshow(y[np.newaxis,:], cmap="plasma", aspect="auto")
#     ax[i].set_title('Drug-%d'%(i))
    

class Estimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self,num_actions,s_init=np.zeros((1,))):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.models = []
        for _ in range(num_actions):
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            model.partial_fit([s_init], [0])
            self.models.append(model)
    
    def predict(self, s, a=None):
        """
        Makes value function predictions.
        
        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for
            
        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.
            
        """
        # features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([s])[0] for m in self.models])
        else:
            return self.models[a].predict([s])[0]
    
    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        self.models[a].partial_fit([s], [y])
        
class Simulator():
    def drug_concentration(self, d_ts, k):
        """
        d_ts.shape = (#drug, T)
        """
        k_ts = np.array([ np.exp(-k*t) for t in range(d_ts.shape[1]) ]).T
        conc = np.array([np.convolve(d_ts[i],k_ts[i],'full') for i in range(d_ts.shape[0])])
        conc = conc[:,:d_ts.shape[1]]
        return conc

    def __init__(self, params, E_init):
        self.a0, self.a, self.b0, self.b, self.k, self.lag, self.T, self.W = params
        self.num_actions = 2**self.k
        self.E_init = E_init
        self.p = self.E_init/self.W
        self.E = self.E_init
        self.D = np.zeros( (self.lag,len(self.k)) )
    
    def step(self,d):
        self.D = np.concatenate((self.D, [d]))
        p_lag = self.p[ -self.lag : ]
        self.Dc = self.drug_concentration( np.array(self.D).T , self.k ).T
        A = self.a0 + np.dot(self.a,p_lag)
        B = self.b0 + np.dot(self.b,self.Dc[-1,:])
        p = scipy.special.expit(A)*scipy.special.expit(1 - B)
        E = np.random.binomial(self.W, p)
        self.E = np.concatenate((self.E,[E]))
        self.p = np.concatenate((self.p,[p]))
        
    def reset(self):
        self.p = self.E_init/self.W
        self.E = self.E_init
        self.D = np.zeros( (self.lag,len(self.k)) )
    
        
        

def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    
    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def q_learning(sim, estimator, episode_length, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.
    
    Args:
        sim: Simulator
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    episode_rewards = np.zeros((num_episodes,))
    
    for i_episode in range(num_episodes):
        
        # The policy we're following
        policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, sim.num_actions)
        
        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_reward = episode_rewards[i_episode - 1]
        # sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state = sim.reset()
        
        # Only used for SARSA, not Q-Learning
        next_action = None
        
        # One step in the environment
        for t in itertools.count():
                        
            # Choose an action to take
            # If we're using SARSA we already decided in the previous step
            if next_action is None:
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            else:
                action = next_action
            
            # Take a step
            next_state, reward = sim.step(action)
    
            # Update statistics
            episode_rewards[i_episode] += reward
            
            # TD Update
            q_values_next = estimator.predict(next_state)
            
            # Use this code for Q-Learning
            # Q-Value TD Target
            td_target = reward + discount_factor * np.max(q_values_next)
            
            # Use this code for SARSA TD Target for on policy-training:
            # next_action_probs = policy(next_state)
            # next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)             
            # td_target = reward + discount_factor * q_values_next[next_action]
            
            # Update the function approximator using our target
            estimator.update(state, action, td_target)
            
            print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, last_reward), end="")
                
            if t >= episode_length:
                break
                
            state = next_state
    
    return episode_rewards
