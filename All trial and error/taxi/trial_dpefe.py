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
import agent_dpefe_strict_c as helper
from agent_dpefe_strict_c import agent as dpefe_agent

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

T = 2
a = dpefe_agent(num_states=num_states, num_obs=num_obs, num_controls=num_controls, 
                a = A, planning_horizon = T)

# %%

# Trial
m_trials = 1
n_trials = 100
time_horizon = 15000

t_length = np.zeros((m_trials, n_trials))

for mt in range(m_trials):
    print(mt)
    
    for trial in range(n_trials):
        
        print(mt, "trial: ", trial)
            
        obs, info = env.reset(seed=42)
        obs_list = [obs]
        
        score = 0
        for t in range(time_horizon):
            
            action  = a.step(obs_list, t)
            obs, reward, terminated, truncated, info = env.step(action)
            obs_list = [obs]
            
            #Learning C only for the goal-state
            if(reward == 20):
                a.update_c(obs_list)
                
            score += reward  
            
            #Checking for succesful episode
            if terminated or truncated:
                break

        t_length[mt,trial] = score
        a.end_of_trial()