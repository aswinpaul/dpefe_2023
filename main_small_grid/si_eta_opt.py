#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:21:12 2023

@author: aswinpaul
"""

from GridEnv.grid_env import grid_environment as Env 
env = Env()
# Environment grid_env.grid_environment()

import numpy as np
np.random.seed(10)

# agent
import agent_soph_inf as helper
from agent_soph_inf import agent as dpefe_agent

# %%

# Generative model
s1_size = 50
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

A = helper.random_A_matrix(num_obs, num_states)*0 
A[0] = np.eye(s1_size)
# %%

# %%

def eta_optimiser(eta):
    
    # Trial
    m_trials = 10
    n_trials = 100
    time_horizon = 15000
    
    t_length = np.zeros((m_trials, n_trials))
    
    for mt in range(m_trials):
        # print(mt)
        
        N = 1
        
        a = dpefe_agent(num_states=num_states, num_obs=num_obs, 
                        num_controls=num_controls, 
                        a = A,
                        planning_precision = 1,
                        action_precision = 16,
                        planning_horizon = N,
                        episode_horizon = 100,
                        search_threshold = 1/2, 
                        eta_par = eta)
        
        for trial in range(n_trials):
            
            obs, info = env.reset(seed=42)
            obs_list = [obs]
            prev_obs_list = obs_list
            
            score = 0
            st = []
            st.append(obs)
                
            for t in range(time_horizon):

                action  = a.step(obs_list, t)
                obs, reward, terminated, truncated, info = env.step(action)
                st.append(obs)
                
                prev_obs_list = obs_list
                obs_list = [obs]
                
                #Learning a general C to aid tree-search in soph.inf
                if(reward == -0.5):
                    r = 0
                    a.update_c(prev_obs_list, obs_list, r, terminal = False)
                    
                if(reward == 10):
                    r = 1
                    a.update_c(prev_obs_list, obs_list, r, terminal = True)
                
                score += reward  
                
                #Checking for succesful episode
                if terminated or truncated:
                    action  = a.step(obs_list, t)
                    break
    
            t_length[mt,trial] = score
            a.end_of_trial()
            
    return np.mean(t_length)

list_results = []

for i in range(500,10000,500):
    print(i)
    list_results.append((i, eta_optimiser(i)))
    
print(list)