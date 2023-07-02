#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:29:05 2023

@author: aswinpaul
"""

import numpy as np
from scipy.stats import dirichlet

# %%

class z_learning_agent():
    def __init__(self, numS, numA):
        self.c = 5
        self.eta = 1
        self.epsilon = 0.1
        
        self.numS = numS
        self.numA = numA
        
        self.z = np.ones(numS)
        self.p_counts = np.ones((numS, numS))
        self.p = np.zeros((numS, numS))
        
        for i in range(numS):
            self.p[:,i] = dirichlet.mean(self.p_counts[:,i])
            
        self.model_counts = np.ones((numS, numA, numS))
        self.model = np.zeros((numS, numA, numS))
        #self.normalise_model()
        
    def decision_making(self, observation, cond=False):
        
        self.policy = np.matmul(np.reshape(self.p[:,observation],(self.numS,1)),
                                np.reshape(self.z, (1,self.numS)))
        normaliser = np.sum(self.policy)
        if(normaliser != 0):
            self.policy = self.policy/normaliser
        
        desired_next_state = np.argmax(self.policy[:,observation])
        self.q_a = self.model_counts[observation,:,desired_next_state]
        
        #Exploration-rate for e-greedy
        exploration_rate = np.random.uniform(0, 1)
        
        #Greedy-action
        if(exploration_rate > self.epsilon):
            action = np.argmax(self.q_a)
        #Random-action
        else:
            action = np.random.randint(0,self.numA)
        return(action)
    
    def update_z(self, prev_obs, reward, observation, tau, terminal=False):
        self.eta = 0.8 #self.c / (self.c + tau)
        term1 = (1 - self.eta)*self.z[prev_obs]
        exp_reward = np.clip(np.exp(reward), None, 1e+5)
        
        if(terminal):
            term2 = exp_reward
        else:
            term2 = self.eta * exp_reward * self.z[observation]
        self.z[prev_obs] =  term1 + term2
        
    def update_model(self, prev_obs, action, observation):
        self.model_counts[prev_obs, action, observation] += 1
        
    def normalise_model_and_p(self):
        for i in range(self.numS):
            for j in range(self.numA):
                self.model[i, j, :] = dirichlet.mean(self.model_counts[i, j, :])
                
        self.p = np.zeros((self.numS, self.numS))
        for i in range(self.numS):
            self.p[:,i] = dirichlet.mean(self.p_counts[:,i])
            
        for i in range(self.numS):
            for j in range(self.numS):
                if(self.model_counts[i,:,j].max() < 2):
                    self.p[j,i] = 0
                else:
                    self.p[j,i] = 1