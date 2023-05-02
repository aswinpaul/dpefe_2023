#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:52:33 2022

@author: aswinpaul
"""
# %%
import numpy as np

# %%

class dynaq_agent():
    def __init__(self, numS, numA, epsilon = 0.1, alpha = 0.5, gamma = 1, replay = 0):
        self.numS = numS
        self.numA = numA
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((numS, numA))
        self.model = np.nan*np.zeros((numS,numA,2))
        self.replay = replay
        
    def decision_making(self, observation):
        #Exploration-rate for e-greedy
        exploration_rate = np.random.uniform(0, 1)

        #Greedy-action
        if(exploration_rate > self.epsilon):
            action = np.argmax(self.Q[observation,:])
        #Random-action
        else:
            action = np.random.randint(0,self.numA)
        
        return action
    
    def learning_with_replay(self, obs_prev, action, observation, reward):
        #Updating Q (Q-learning off policy)
        Target = reward + self.gamma*np.max(self.Q[observation,:])
        # update Q value
        self.Q[obs_prev,action] += self.alpha* (Target - self.Q[obs_prev,action]) 
    
        #model_update
        self.model[obs_prev,action] = [observation,reward]
        
        for j in range(self.replay):
            candidates = np.array(np.where(~np.isnan(self.model[:,:,0]))).T
            idx = np.random.choice(len(candidates))
            # Obtain the randomly selected state and action values from the candidates
            xi,actioni = candidates[idx]
            # Obtain the expected reward and next state from the model
            xpi,rewardi = self.model[xi,actioni] 
            # Q learning with memory
            Target = rewardi + self.gamma*np.max(self.Q[int(xpi),:])
            # Updaing Q
            self.Q[xi,actioni] += self.alpha*(Target - self.Q[xi,actioni])
# %%