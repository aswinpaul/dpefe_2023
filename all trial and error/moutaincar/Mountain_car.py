#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 11:25:50 2022

@author: aswinpaul
"""

import gym 
import numpy as np
from dpefeagent import dpefe_agent as agent

env = gym.make('MountainCar-v0')
# env.observation_space.low = array([-1.2 , -0.07], dtype=float32)
# env.observation_space.high = array([0.6 , 0.07], dtype=float32)
# env.action_space = Discrete(3)

def cat_obs(obs):
    
    obs[0] = obs[0]*10
    obs[1] = obs[1]*100
    
    obs[0] += 12
    obs[1] += 7
    
    obs[0] = np.around(obs[0],decimals=0)
    obs[1] = np.around(obs[1],decimals=0)
    
    return [int(obs[0]), int(obs[1])]
    
# Generative model

# Hidden states
s1_size = 19
s2_size = 15

# Control
s1_actions = ['Accelerate to the left', 'Don’t accelerate', 'Accelerate to the right']
s2_actions = ['Do nothing']

# Observations
o1_size = s1_size
o2_size = s2_size

# As parameters

# Hidden states
num_states = [s1_size, s2_size]
num_factors = len(num_states)

# Controls
num_controls = [len(s1_actions), len(s2_actions)]

#Observations
num_obs = [o1_size, o2_size]
num_modalities = len(num_obs)

# Planning horizon
T = 150

A = agent.random_A_matrix(num_obs, num_states)*0 
for i in range(num_states[1]):
    A[0][:,:,i] = np.eye(s1_size)
for i in range(num_states[0]):
    A[1][:,i,:] = np.eye(s2_size)

goal_obs = 17
C = agent.obj_array_zeros(num_obs)
C[0][goal_obs] = 1
C[1] += 1 / num_obs[1]

# Create an agent
a = agent(num_states, num_obs, num_controls, T = T, a = A, c = C)

# %%
m_trials = 1
n_trials = 20
time_horizon = 15000

# %%
t_length = np.zeros((m_trials, n_trials))
t_best = 150

for mt in range(m_trials):
    print(mt)
    
    for trial in range(n_trials):
        
        print(mt, "trial: ", trial)
        if(trial == 0 or t_length[mt, trial-1] > t_best):
            print("Planning")
            a.dpefe_plan()
            
        obs, info = env.reset()
        obs = cat_obs(obs)
        tau = 0
        for t in range(time_horizon):
            
            action = a.step(obs, tau)
            tau += 1
            
            #print(action)
            observation, reward, terminated, truncated, info = env.step(action)
            #print(observation)
            obs = cat_obs(observation)
            #print(obs)
            
            #Learning C
            if(reward > 0):
                a.update_c(obs)
                
            #Checking for succesful episode
            if(terminated or truncated):
                break
            # if(truncated):
            #     tau = 0
            #     obs, info = env.reset()
            #     obs = cat_obs(obs)
                
        t_length[mt,trial] = t
        a.learn_params_endoftrial()
        
        if(t < t_best):
            t_best = t
