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
    
    def __init__(self, numS, numD, numA, epsilon = 0.1, alpha = 0.5, gamma = 1, replay = 0):
        
        self.numS = numS
        self.numD = numD
        self.numA = numA
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((numS,numD,numA))
        self.model = np.nan*np.zeros((numS,numD,numA,3))
        self.replay = replay
        
    def decision_making(self, observation):
        
        state_obs = observation[0] 
        dir_obs = observation[1] 
        
        #Exploration-rate for e-greedy
        exploration_rate = np.random.uniform(0, 1)

        #Greedy-action
        if(exploration_rate > self.epsilon):
            action = np.argmax(self.Q[state_obs,dir_obs,:])
            
        #Random-action
        else:
            action = np.random.randint(0,self.numA)
        
        return action
    
    def learning_with_replay(self, obs_prev, action, observation, reward, episode_num = 1, total_number_of_episodes = 1000):
        state_prev = obs_prev[0]
        dir_prev = obs_prev[1]
        
        state_obs = observation[0] 
        dir_obs = observation[1] 
        
        #Updating Q (Q-learning off policy)
        Target = reward + self.gamma * np.max(self.Q[state_obs,dir_obs,:])
        # update Q value
        self.Q[state_prev,dir_prev,action] += self.alpha * (Target - self.Q[state_prev,dir_prev,action]) 
    
        #model_update
        self.model[state_prev,dir_prev,action] = [state_obs,dir_obs,reward]
        
        for j in range(self.replay):
            
            candidates = np.array(np.where(~np.isnan(self.model[:,:,:,0]))).T
            
            idx = np.random.choice(len(candidates))
            # Obtain the randomly selected state and action values from the candidates
            xi,di,actioni = candidates[idx]
            # Obtain the expected reward and next state from the model
            xpi,dpi,rewardi = self.model[xi,di,actioni] 
            # Q learning with memory
            Target = rewardi + self.gamma * np.max(self.Q[int(xpi),int(dpi),:])
            # Updaing Q
            self.Q[xi,di,actioni] += self.alpha * (Target - self.Q[xi,di,actioni])

# %%