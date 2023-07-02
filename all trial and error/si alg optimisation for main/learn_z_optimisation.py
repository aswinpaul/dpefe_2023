#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 13:13:37 2023

@author: aswinpaul
"""

from GridEnv.grid_env import grid_environment as Env 
env = Env()
EPS_VAL = 1e-10
# Environment grid_env.grid_environment()

import numpy as np
from scipy.stats import dirichlet
np.random.seed(10)

n_trials = 100
time_horizon = 1000

t_length = np.zeros((n_trials))
B = env.get_trueB()

def softmax(dist):
    """ 
    Computes the softmax function on a set of values
    """
    output = dist - dist.max(axis=0)
    output = np.exp(output)
    output = output / np.sum(output, axis=0)
    return output

def update_z(prev_obs, obs, reward, t, terminal = False):
    exp_reward = np.exp(reward)
    
    c = 1000
    
    eta = c/(c+t)
    
    if(terminal == True):
        z[obs] = exp_reward
        
        z[prev_obs] = 0.9*z[obs]
        
    else:
        a = (1 - eta)*z[prev_obs]
        b = eta*exp_reward*z[obs]
        z[prev_obs] = a + b
        
    
    factor = 1
    C = softmax(factor*z)
    #C = dirichlet.mean(factor*(z + EPS_VAL))
    
    return z,C
    
s1_size = 50

z = np.zeros((s1_size))
C = np.zeros((s1_size))

for trial in range(n_trials):
        
    obs, info = env.reset(seed=42)
    obs_list = [obs]
    prev_obs_list = obs_list
    
    score = 0
    st = []
    st.append(obs)
        
    for t in range(time_horizon):
        
        action  = np.random.randint(0,4)
        obs, reward, terminated, truncated, info = env.step(action)
        st.append(obs)
        
        prev_obs_list = obs_list
        obs_list = [obs]
        
        #Learning a general C to aid tree-search in soph.inf
        if(reward == -0.5):
            r = 0
            z,C = update_z(prev_obs_list, obs_list, r, t, terminal = False)
            
        if(reward == 10):
            r = 1
            z,C = update_z(prev_obs_list, obs_list, r, t, terminal = True)
        
        score += reward  
        
        #Checking for succesful episode
        if terminated or truncated:
            break

    t_length[trial] = score