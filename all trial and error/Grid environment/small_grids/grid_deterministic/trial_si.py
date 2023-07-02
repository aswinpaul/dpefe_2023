#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:38:48 2022

@author: aswinpaul
"""
import matplotlib.pyplot as plt

from grid_env import grid_environment as Env 
env = Env()
# Environment grid_env.grid_environment()

import numpy as np
np.random.seed(10)

# agent
import agent_soph_inf as helper
from agent_soph_inf import agent as si_agent

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

def render_c_matrix(t, path="c_matrix_uninf.txt"):
    plt.figure(figsize=(5, 5))
    grid = np.loadtxt(path, dtype = int)
    plt.imshow(grid, cmap=plt.cm.CMRmap, interpolation='nearest')
    plt.xticks([]), plt.yticks([])
    # plt.show()
    # Figure figure when rendered in environment
    #plt.savefig(f'./img/img_{t}.png')
    plt.savefig('uninfc.png')
    
render_c_matrix(0)
    
# Trial
m_trials = 1
n_trials = 100
time_horizon = 15000

t_length = np.zeros((m_trials, n_trials))

for mt in range(m_trials):
    print(mt)
    
    N = 2
    
    a = si_agent(num_states=num_states, num_obs=num_obs, 
                    num_controls=num_controls, 
                    a = A,
                    planning_precision = 1,
                    action_precision = 1,
                    planning_horizon = N,
                    episode_horizon = 100,
                    eta_par = 13000) #Manually optimised eta parameter
    
    # render_c_matrix()
    
    for trial in range(n_trials):
        C_matrix = a.C[0]
        
        c_text = np.loadtxt("c_matrix.txt", dtype = int)
        
        for i in range(env.numS):
            [x,y] = env.statestoc(i)
            c_text[x][y] = int(10000*C_matrix[i])
            
        np.savetxt("c_matrix.txt",c_text, fmt='%i')   
        
        render_c_matrix(trial)
        
        obs, info = env.reset(seed=42)
        obs_list = [obs]
        prev_obs_list = obs_list
        
        score = 0
        st = []
        st.append(obs)
        cc = a.C[0]
        
        for t in range(time_horizon):
            a.action_precision = 1 if t > 50 else 1024
            action  = a.step(obs_list, t)
            obs, reward, terminated, truncated, info = env.step(action)
            st.append(obs)
            
            prev_obs_list = obs_list
            obs_list = [obs]
            
            #Learning a general C to aid tree-search in soph.inf
            #if reward is -0.5 it is a usual step with no additional info
            if(reward == -0.5):
                r = 0
                a.update_c(prev_obs_list, obs_list, r, terminal = False)
                
            #if reward is 10, it is a terminal state i.e the goal state    
            if(reward == 10):
                r = 1
                a.update_c(prev_obs_list, obs_list, r, terminal = True)
            
            score += reward  
            
            #Checking for succesful episode
            if terminated or truncated:
                action  = a.step(obs_list, t)
                break

        t_length[mt,trial] = t
        a.end_of_trial()
        
#with open('data_si.npy', 'wb') as file:
    #np.save(file, t_length)
