#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:38:48 2022

@author: aswinpaul

"""

# This is needed agents are in a diff folder
import os
import sys
from pathlib import Path

path = Path(os.getcwd())
module_path = str(path.parent) + '/'
sys.path.append(module_path)

from grid_env import grid_environment as Env 
env = Env()
# Environment grid_env.grid_environment()

mtrial = int(sys.argv[1])
n_trials = int(sys.argv[2])
print("mtrial:", mtrial, "ntrial:", n_trials)
import numpy as np
np.random.seed(mtrial)

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

A = random_A_matrix(num_obs, num_states)*0 
A[0] = np.eye(s1_size)

# %%

# Trial
time_horizon = 15000

score_vec = np.zeros((n_trials))
  
T = 80
a = dpefe_agent(num_states = num_states, 
                num_obs = num_obs, 
                num_controls = num_controls, 
                A = A,
                planning_precision = 1,
                action_precision = 1,
                planning_horizon = T)
a.lr_pB = 1024

for trial in range(n_trials):

    rs = True if trial%25 == 0 else False
    obs, info = env.reset(seed=42, randomise_goal=rs)

    goal_state = env.end_state
    C = obj_array_zeros(num_obs)
    C[0][goal_state] = 100
    a.C = C
    a.plan_using_dynprog()
  
    score = 0
    a.tau = 0
    
    for t in range(time_horizon):
        
        a.alpha = 1e-5 if(t>500) else 1e5
        action  = a.step([obs])
        obs, reward, terminated, truncated, info = env.step(action)                
        score += reward  
                        
        #Checking for succesful episode
        if terminated or truncated:
            action  = a.step([obs])
            break

    score_vec[trial] = score
        
file_name = 'data_dpefe/data_dpefe_' + str(mtrial) + '.npy'
with open(file_name, 'wb') as file:
    np.save(file, score_vec)
