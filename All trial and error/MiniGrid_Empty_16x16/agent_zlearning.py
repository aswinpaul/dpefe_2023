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
        
        self.c = 50
        self.eta = 1
        
        self.numS = numS
        self.numA = numA
        
        self.z = np.ones(numS)
        self.p_counts = np.zeros((numS, numS)) + 1
        self.p = np.zeros((numS, numS))
        
        for i in range(numS):
            self.p[:,i] = dirichlet.mean(self.p_counts[:,i])
            
        self.model_counts = np.zeros((numS, numA, numS)) + 1
        self.model = np.zeros((numS, numA, numS))
        self.normalise_model()
        
    def decision_making(self, observation):
        
        self.policy = np.matmul(np.reshape(self.p[:,observation], (1024,1)),np.reshape(self.z, (1,1024)))
        normaliser = np.sum(self.policy)
        self.policy = self.policy/normaliser
        
        desired_next_state = np.argmax(self.policy[:,observation])
        self.q_a = self.model[observation,:,desired_next_state]
        if(self.q_a.max()<0.001):
            action = np.random.randint(0,self.numA)
        else:
            action = np.argmax(self.q_a)
        return(action)
    
    def update_z(self, prev_obs, reward, observation, tau):
        self.eta = self.c / (self.c + tau)
        
        term1 = (1 - self.eta)*self.z[prev_obs]
        exp_reward = np.clip(np.exp(reward), None, 1e+5)
        term2 = self.eta*exp_reward*self.z[observation]

        self.z[prev_obs] =  term1 + term2
        
    def update_model(self, prev_obs, action, observation):
        self.model_counts[prev_obs, action, observation] += 100
        
    def normalise_model(self):
        for i in range(self.numS):
            for j in range(self.numA):
                self.model[i, j, :] = dirichlet.mean(self.model_counts[i, j, :])