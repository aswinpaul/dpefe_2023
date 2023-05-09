#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:38:48 2022

@author: aswinpaul
"""

from GridEnv.grid_env import grid_environment as Env 
env = Env()
# Environment grid_env.grid_environment()

import numpy as np
np.random.seed(10)


# Trial
m_trials = 100
n_trials = 100
time_horizon = 15000

t_length = np.zeros((m_trials, n_trials))

for mt in range(m_trials):
    print(mt)
    
    
    for trial in range(n_trials):
        
        #print(mt, "trial: ", trial)
            
        obs, info = env.reset(seed=42)
        obs_list = [obs]
        prev_obs_list = obs_list
        
        score = 0

        for t in range(time_horizon):
            
            action  = np.random.choice([0,1,2,3], size=None, replace=True, p=[0.25,0.25,0.25,0.25])
            obs, reward, terminated, truncated, info = env.step(action)
            
            prev_obs_list = obs_list
            obs_list = [obs]
            
            score += reward  
            
            #Checking for succesful episode
            if terminated or truncated:
                break

        t_length[mt,trial] = t
        
with open('data_random.npy', 'wb') as file:
    np.save(file, t_length)
