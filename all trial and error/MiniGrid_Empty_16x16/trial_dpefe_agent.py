#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:03:44 2023

@author: aswinpaul
"""

# Time of execution
import time
st = time.process_time()

# Environment Imports
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper

# Active inference agent
import agent_dpefe as helper
from agent_dpefe import agent as dpefe_agent

# Other imports
import numpy as np

env = gym.make("MiniGrid-Empty-16x16-v0", render_mode = "rgb_array")
env = FullyObsWrapper(env) 

# %%
 
# Helper functions

def observation_decoder(observation, grid_size = 16):
    
  agent_direction = observation['direction']
  x = observation['image'][:,:,0]

  num_states = grid_size*grid_size                    
  
  agent_coor = np.argwhere(x == 10)
  agent_state = (agent_coor[0][0] * grid_size) + agent_coor[0][1] - 1
  obs = agent_state + agent_direction*num_states

  return obs

# %%

# Generative model for active inference agent

EPS_VAL = 1e-15

s1_size = 1024 # (NoofStates times NoofDirection-of-agent)
s1_actions = ['Turn left', 'Turn right', 'Move forward']

o1_size = s1_size

# Hidden states
num_states = [s1_size]
num_factors = len(num_states)

# Controls
num_controls = [len(s1_actions)]

# Observations
num_obs = [o1_size]
num_modalities = len(num_obs)

# Planning horizon mentioned seperately during trials

# %%

# Trials

seedloops = 5
episodes = 50

data = np.zeros((seedloops,episodes))

for sl in range(seedloops):
    
    rseed = sl
    np.random.seed(rseed)
    print('sl:', sl)

    N = 100
    
    A = helper.random_A_matrix(num_obs, num_states)*0 
    A[0] = np.eye(s1_size)
    
    a = dpefe_agent(num_states, num_obs, num_controls, a = A,
                    planning_horizon = N, action_precision = 100, MDP = True)
    
    planning_count = 0
    seen_goal = False
    
    for ts in range(episodes):
        
        # print('episode:', ts)
        done = False
        tau = 0
        
        observation, info = env.reset(seed = rseed)
        obs = observation_decoder(observation)
        score = 0
        
        if((ts == 10 or ts == 5) and seen_goal == True):
            a.plan_using_dynprog()
            planning_count += 1
            
        if(seen_goal == True and planning_count < 1):
            a.plan_using_dynprog()
            planning_count += 1
            
        while(done == False):
            
            action = a.step([obs], tau)
            observation, reward, terminated, truncated, info = env.step(action)
            obs = observation_decoder(observation)
            
            reward = 1024 if reward > 0 else -1
            score += reward
            
            if(reward == 1024):
                a.update_c([obs], factor = 100)
                seen_goal = True
            
            if terminated or truncated:
                action = a.step([obs], tau)
                done = True
            
            tau += 1
            
        data[sl][ts] = score
        a.end_of_trial(learning = True)
            
# %%

# get the end time
et = time.process_time()

# get execution time
res = et - st
print('CPU Execution time:', res, 'seconds')

# %%
with open('data_trial_dpefe.npy', 'wb') as f:
    np.save(f, data)