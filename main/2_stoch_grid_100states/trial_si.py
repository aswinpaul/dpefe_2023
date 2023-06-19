#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:38:48 2022

@author: aswinpaul
"""

import numpy as np
from scipy.stats import dirichlet

# This is needed agents are in a diff folder
import os
import sys
from pathlib import Path

eta = 12000 #int(sys.argv[1])

path = Path(os.getcwd())
module_path = str(path.parent) + '/'
sys.path.append(module_path)

from environments.grid_environment import grid_environment as Env 
env = Env(path = '../environments/grid10.txt', stochastic = True, end_state=37)
# Environment grid_env.grid_environment()

num_states = env.numS
num_actions = env.numA

# agent
# agent
# agent
from agents.agent_si_z_learning import si_agent_learnc as si_agent
from pymdp.utils import random_A_matrix, random_B_matrix

# %%

def ai_C_to_c(C, grid_size=10):
    c = np.zeros((grid_size, grid_size))
    for i in range(env.numS):
        [x,y] = env.statestoc(i)
        c[x][y] = C[0][i]
    return c

# Generative model
s1_size = num_states
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

A = random_A_matrix(num_obs, num_states)
A[0] = np.eye(s1_size)

EPS_VAL = 1e-5
b = random_B_matrix(num_states, num_controls)*0 + EPS_VAL
B = np.copy(b)
for i in range(len(num_states)):
    for j in range(num_states[i]):
        for k in range(num_controls[i]):
            B[i][:,j,k] = dirichlet.mean(b[i][:,j,k])

# %%

# %%
    
# Trial
m_trials = 100
n_trials = 50

time_horizon = 10000
score_vec = np.zeros((m_trials, n_trials))

for mt in range(m_trials):
    print(mt)
    
    N = 1
    
    a = si_agent(num_states=num_states, 
                 num_obs=num_obs, 
                 num_controls=num_controls, 
                 A = A,
                 B = B,
                 planning_horizon = N,
                 eta_par = eta,
                 search_threshold=1/64) #Manually optimised eta parameter
    
    a.lr_pB = 1e+10
    
    for trial in range(n_trials):
        
        obs, info = env.reset(seed=42)
        
        score = 0
        cc = a.C[0]            
        a.tau = 0  
        
        for t in range(time_horizon):
            a.alpha = 1024 if(t<100) else 1
            
            action  = a.step([obs])
            prev_obs = obs
            obs, reward, terminated, truncated, info = env.step(action)
            
            #Learning a general C to aid tree-search in soph.inf
            #if reward is not 10 it is a usual step with no additional info
            #if reward is 10, it is a terminal state i.e the goal state    
            if(reward == 10):
                r = 1
                a.update_c([prev_obs], [obs], r, terminal = True)
            else:
                r = 0
                a.update_c([prev_obs], [obs], r, terminal = False)
            
            score += reward  
            
            #Checking for succesful episode
            if terminated or truncated:
                action  = a.step([obs])
                break

        score_vec[mt,trial] = score

#print(", Eta,",eta,",Mean length of episode,",np.mean(score_vec[:,25:]))

with open('data_si.npy', 'wb') as file:
    np.save(file, score_vec)
