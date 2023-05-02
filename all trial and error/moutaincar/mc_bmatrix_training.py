#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:12:14 2022

@author: aswinpaul
"""

import gym 
import numpy as np
from dpefeagent import dpefe_agent as agent

env = gym.make('MountainCar-v0')
# env.observation_space.low = array([-1.2 , -0.07], dtype=float32)
# env.observation_space.high = array([0.6 , 0.07], dtype=float32)
# env.action_space = Discrete(3)

def cat_obs(obse):
    
    obse[0] = obse[0]*10
    obse[1] = obse[1]*100
    
    obse[0] += 12
    obse[1] += 7
    
    obse[0] = np.around(obse[0],decimals=0)
    obse[1] = np.around(obse[1],decimals=0)
    
    return [int(obse[0]), int(obse[1])]
    
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

# Create an agent
a = agent(num_states, num_obs, num_controls, T = T, a = A)

# %%
m_trials = 10
n_trials = 100000
time_horizon = 300

# %%

for mt in range(m_trials):
    print(mt)
    
    for trial in range(n_trials):

        obs, info = env.reset()
        obs = cat_obs(obs)
        a.infer_hiddenstate(obs)
        
        tau = 0
        for t in range(time_horizon):
            
            action = env.action_space.sample()
            tau += 1
            
            #print(action)
            observation, reward, terminated, truncated, info = env.step(action)
            #print(observation)
            obs = cat_obs(observation)
            a.infer_hiddenstate(obs)
            #print(obs)
            a.update_b([action, 0])
            #Checking for succesful episode
            
            if(terminated or truncated):
                if(terminated):
                    print("Succesful episode")
                break
            
        a.learn_B()