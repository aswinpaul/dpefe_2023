#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:15:26 2022

@author: aswinpaul
"""

import time
# Start time of program
st = time.process_time()

import numpy as np
np.random.seed(10)

from wgwt_det_environment import Environment as Env, StartandGoal, ImportDynamics

import agent_dpefe_backup as helper
from agent_dpefe import agent as dpefe_agent

start_state, end_state = StartandGoal()
Td = ImportDynamics()

# %%

EPS_VAL = 1e-15

# Generative model
s1_size = 70

s1_actions = ['0','1','2','3','4','5','6','7']

o1_size = s1_size

# Hidden states
num_states = [s1_size]
num_factors = len(num_states)

# Controls
num_controls = [len(s1_actions)]

#Observations
num_obs = [o1_size]
num_modalities = len(num_obs)

#Planning horizon
T = 9

m_trials = 1
n_trials = 100
time_horizon = 15000

t_length = np.zeros((m_trials, n_trials))

for mt in range(m_trials):
    
    print(mt)

    A = helper.random_A_matrix(num_obs, num_states)*0 
    A[0] = np.eye(s1_size)
    
    B = helper.random_B_matrix(num_states, num_controls)*0
    
    for i in range(s1_size):
        for j in range(8):
            B[0][:,i,j] = Td[i,j,:] + EPS_VAL
    
    D = helper.obj_array_zeros(num_states)
    D[0] = helper.onehot(start_state, num_states[0])
    
    C = helper.obj_array_zeros(num_obs)
    C[0][end_state] = 1
    
    a = dpefe_agent(num_states, num_obs, num_controls, a = A,
                    planning_horizon = T, action_precision = 100,
                    MDP = True)

    for trial in range(n_trials):
        
        print(mt, "trial: ", trial)
        obs = start_state
        
        for t in range(time_horizon):
            
            action = a.step([obs], t)
            x, y, reward, obs = Env(obs, action)
            
            #Learning C
            if(reward == 1):
                a.update_c([obs])
                
            #Checking for succesful episode
            if (obs == end_state):
                break
            
        t_length[mt,trial] = t
        a.end_of_trial(learning = True)
            
# %%

# get the end time
et = time.process_time()

# get execution time
res = et - st
print('CPU Execution time:', res, 'seconds')