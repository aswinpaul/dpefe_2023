#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:06:01 2022

@author: aswinpaul
"""

import numpy as np

# This is needed agents are in a diff folder
import os
import sys
from pathlib import Path

path = Path(os.getcwd())
module_path = str(path.parent) + '/'
sys.path.append(module_path)

from environments.grid_environment import grid_environment as Env 
env = Env(path = '../environments/grid10.txt', stochastic = True, end_state=37)
# Environment grid_env.grid_environment()

num_states = env.numS
num_actions = env.numA

from agents.agent_dynaq import dynaq_agent as dqa

episodes = 50
seedloops = 100
mem_replay = 10
time_horizon = 15000

#Initialsing the model M(x,a,x',reward)
score_vec = np.zeros((seedloops, episodes))

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
    
        score_vec[sl][ts] = score

with open('data_dynaq_10.npy', 'wb') as file:
    np.save(file, score_vec)