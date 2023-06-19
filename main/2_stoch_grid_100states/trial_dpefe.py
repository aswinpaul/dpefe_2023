#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:38:48 2022

@author: aswinpaul
"""

import numpy as np
np.random.seed(10)

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

# agent
# agent
# agent
from agents.agent_dpefe import dpefe_agent as dpefe_agent
from pymdp.utils import random_A_matrix, obj_array_zeros

# %%

# Generative model
s1_size = env.numS
s1_actions = ['Left', 'Right', 'Up', 'Down']

o1_size = s1_size

# Hidden states
num_states = [s1_size]
num_factors = len(num_states)

# Controls
num_controls = [len(s1_actions)]

#Observations
num_obs = [o1_size]
num_modalities = len(num_obs)

# %%

A = random_A_matrix(num_obs, num_states)
A[0] = np.eye(s1_size)

goal_state = env.end_state
C = obj_array_zeros(num_obs)
C[0][goal_state] = 100

# %%

# Trial
m_trials = 100
n_trials = 50
time_horizon = 15000

score_vec = np.zeros((m_trials, n_trials))

for mt in range(m_trials):
    print(mt)
    
    N = 80
    a = dpefe_agent(num_states = num_states, 
                    num_obs = num_obs, 
                    num_controls = num_controls, 
                    A = A,
                    planning_horizon = N, 
                    action_precision = 1024,
                    C = C)
    
    for trial in range(n_trials):
        a.plan_using_dynprog()    
        
        obs, info = env.reset(seed=42)
        a.tau = 0
        score = 0
        
        for t in range(time_horizon):
            
            action  = a.step([obs], learning=True)
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward  
            
            #Checking for succesful episode
            if terminated or truncated:
                action = a.step([obs], learning=True)
                break

        score_vec[mt,trial] = score
        
with open('data_dpefe.npy', 'wb') as file:
    np.save(file, score_vec)
