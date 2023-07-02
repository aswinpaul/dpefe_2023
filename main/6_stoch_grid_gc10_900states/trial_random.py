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

from environments.grid_environment import grid_environment as Env 
time_horizon = 40000
env = Env(path = '../environments/grid30.txt', stochastic = True, end_state=369, epi_length_limit = time_horizon, p_rew = 10, n_rew = -0.25e-3)
# Environment grid_env.grid_environment()

import numpy as np
np.random.seed(10)

# Trial
m_trials = 100
n_trials = 50

score_vec = np.zeros((m_trials, n_trials))

for mt in range(m_trials):
    print(mt)
    
    
    for trial in range(n_trials):
        if(trial%10 == 0):
            end_state = np.random.randint(0,env.numS)
            end_state = 369 if(trial == 0) else end_state
            env = Env(path = '../environments/grid30.txt', stochastic = True, end_state=end_state, epi_length_limit = time_horizon, p_rew = 10, n_rew = -0.25e-3)
        
        #print(mt, "trial: ", trial)
            
        obs, info = env.reset(seed=42)
        obs_list = [obs]
        prev_obs_list = obs_list
        
        score = 0
        
        for t in range(time_horizon):
            
            action  = np.random.randint(0,4)
            obs, reward, terminated, truncated, info = env.step(action)
            
            prev_obs_list = obs_list
            obs_list = [obs]
            
            score += reward  
            
            #Checking for succesful episode
            if terminated or truncated:
                break

        score_vec[mt,trial] = score
        
with open('data_random.npy', 'wb') as file:
    np.save(file, score_vec)
