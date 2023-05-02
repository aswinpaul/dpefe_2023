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

# agent
import agent_dpefe as helper
from agent_dpefe import agent as dpefe_agent

# %%

# Generative model
s1_size = 470
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

A = helper.random_A_matrix(num_obs, num_states)*0 
A[0] = np.eye(s1_size)

goal_state = env.end_state

C = helper.obj_array_zeros(num_obs)
C[0] = 100*helper.onehot(goal_state, num_states[0])

T = 50

# %%

# Trial
m_trials = 10
n_trials = 100
time_horizon = 15000

t_length = np.zeros((m_trials, n_trials))

for mt in range(m_trials):
    print(mt)
    
    a = dpefe_agent(num_states=num_states, 
                    num_obs=num_obs, 
                    num_controls=num_controls, 
                    a = A,
                    planning_precision = 1,
                    action_precision = 16,
                    planning_horizon = T, 
                    c = C)
    seen_goal = False
    for trial in range(n_trials):
        #print(mt, "trial: ", trial)
            
        obs, info = env.reset(seed=42)
        obs_list = [obs]
        
        score = 0
        for t in range(time_horizon):

            action  = a.step(obs_list, t)
            obs, reward, terminated, truncated, info = env.step(action)
            obs_list = [obs]
                
            score += reward  
            if(reward == 10):
                seen_goal = True
            #Checking for succesful episode
            if terminated or truncated:
                action  = a.step(obs_list, t)
                break

        t_length[mt,trial] = t
        
        # Turning off planning after every episode to save time and run simulations
        yes_planning = False if (trial > 5 and seen_goal == True) else True
        a.end_of_trial(planning = yes_planning)
        
with open('data_dpefe.npy', 'wb') as file:
    np.save(file, t_length)