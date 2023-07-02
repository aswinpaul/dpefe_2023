#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 12:39:06 2022

@author: aswinpaul
"""

import gym
import numpy as np

env = gym.make('MountainCar-v0')

from dynaq_agent import dynaq_agent as dqa

def cat_obs(obs):
    
    obs[0] = obs[0]*10
    obs[1] = obs[1]*100
    
    obs[0] += 11
    obs[1] += 6
    
    obs[0] = np.around(obs[0],decimals = 0)
    obs[1] = np.around(obs[1],decimals = 0)
    
    return int(obs[0]*obs[1])

    
seedloops = 10 #trials
episodes = 1000

#Initialsing the model M(x,a,x',reward)
t_length = np.zeros((seedloops, episodes))

agent = dqa(252,3,replay=10)

for mt in range(seedloops):

    print(mt)
    
    # Changing random seeds
    np.random.seed(mt)
    
    observation, info = env.reset(seed=42)
    obs = cat_obs(observation)
    
    for trial in range(episodes):
        
        done = False
        tau = 0
        
        while(done == False):
            tau += 1
            
            action = agent.decision_making(obs)
            obs_prev = obs    
            #Fetching next-state reward from envrionment-function
            observation, reward, terminated, truncated, info = env.step(action)
            obs = cat_obs(observation)
            
            agent.learning_with_replay(obs_prev, action, obs, reward)
            
            if terminated or truncated:
                observation, info = env.reset(seed = 42)
                done == True
                break
    
        t_length[mt,trial] = tau