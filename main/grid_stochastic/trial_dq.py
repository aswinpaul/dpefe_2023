#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:06:01 2022

@author: aswinpaul
"""

import numpy as np

from GridEnv.grid_env import grid_environment as Env 
env = Env()
num_states = 50
num_actions = 4

from agent_dynaq import dynaq_agent as dqa

episodes = 100
seedloops = 100 #trials
mem_replay = 10
time_horizon = 15000

#Initialsing the model M(x,a,x',reward)
t_length = np.zeros((seedloops, episodes))

for sl in range(seedloops):
    print(sl)
    agent = dqa(num_states, num_actions, replay=mem_replay)
        
    # Changing random seeds
    rseed = sl;
    np.random.seed(rseed)
    
    for ts in range(episodes):
        
        done = False
        tau = 0
        score = 0
        observation, info = env.reset(seed=42)
        
        for t in range(time_horizon):
            tau += 1
            
            action = agent.decision_making(observation)
            
            obs_prev = observation    
            #Fetching next-state reward from envrionment-function
            observation, reward, terminated, truncated, info = env.step(action)
            
            agent.learning_with_replay(obs_prev, action, observation, reward)
            score += reward
            
            if terminated or truncated:
                break
    
        t_length[sl][ts] = t

with open('data_dynaq_10.npy', 'wb') as file:
    np.save(file, t_length)