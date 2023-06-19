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
env = Env(path = '../environments/grid20.txt', 
          stochastic = True, end_state=150, epi_length_limit=15000)
# Environment grid_env.grid_environment()

import numpy as np
np.random.seed(10)

# Trial
m_trials = 10
n_trials = 50
time_horizon = 15000

score_vec = np.zeros((m_trials, n_trials))

for mt in range(m_trials):
    print(mt)
    for trial in range(n_trials):
        obs, info = env.reset(seed=42)
        obs_list = [obs]
        prev_obs_list = obs_list
        
        score = 0
        st = []
        st.append(obs)
        
        for t in range(time_horizon):
            
            action  = np.random.choice([0,1,2,3], size=None, replace=True, 
                                       p=[0.25,0.25,0.25,0.25])
            obs, reward, terminated, truncated, info = env.step(action)
            st.append(obs)
            
            prev_obs_list = obs_list
            obs_list = [obs]
            
            score += reward  
            
            #Checking for succesful episode
            if terminated or truncated:
                break

        score_vec[mt,trial] = score
        
with open('data_random.npy', 'wb') as file:
    np.save(file, score_vec)
