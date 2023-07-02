#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:38:48 2022

@author: aswinpaul
"""

import gym
env = gym.make('Taxi-v3')
env.action_space.seed(42)
# Environment grid_env.grid_environment()

import numpy as np
np.random.seed(10)

# agent
import agent_soph_inf_learnc as helper
from agent_soph_inf_learnc import agent as dpefe_agent

# %%

# Generative model
s1_size = 500
s1_actions = ['move south', 'move north', 'move east', 'move west', 'pickup passenger', 'drop off passenger']

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

# Trial
m_trials = 10
n_trials = 100
time_horizon = 15000

t_length = np.zeros((m_trials, n_trials))

for mt in range(m_trials):
    print(mt)
    
    T = 2
    a = dpefe_agent(num_states=num_states, num_obs=num_obs, 
                    num_controls=num_controls, a = A, planning_horizon = T,
                    threshold=0, action_precision=10)
    
    for trial in range(n_trials):
        
        print(mt, "trial: ", trial)
            
        obs, info = env.reset(seed=42)
        obs_list = [obs]
        prev_obs_list = obs_list
        
        score = 0
        for t in range(time_horizon):
            
            action  = a.step(obs_list, t)
            obs, reward, terminated, truncated, info = env.step(action)
            prev_obs_list = obs_list
            obs_list = [obs]
            
            #Learning a general C to aid tree-search in soph.inf
            if(reward == -1):
                r = 0
            if(reward == 20):
                r = 10
            if(reward == -10):
                r = -2
            
            a.update_c(prev_obs_list, obs_list, r)
            score += reward  
            
            #Checking for succesful episode
            if terminated or truncated:
                break

        t_length[mt,trial] = score
        a.end_of_trial()
        
with open('data_si.npy', 'wb') as file:
    np.save(file, t_length)