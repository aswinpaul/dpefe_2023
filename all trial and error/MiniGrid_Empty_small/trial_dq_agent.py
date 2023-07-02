#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 12:45:25 2023

@author: aswinpaul
"""

import gymnasium as gym
import numpy as np

from minigrid.wrappers import FullyObsWrapper

# Dyna q agent with replay (replay = 0 is q-learning agent)
from agent_dynaq import dynaq_agent as dqa

env = gym.make("MiniGrid-Empty-5x5-v0", render_mode = "rgb_array")
env = FullyObsWrapper(env) 

# %%

def observation_decoder(observation, grid_size = 5):

  agent_direction = observation['direction']
  x = observation['image'][:,:,0]                  
  
  agent_coor = np.argwhere(x == 10)

  agent_state = (agent_coor[0][0] * grid_size) + agent_coor[0][1] - 1
  
  obs = [agent_state,agent_direction]
  
  return obs

# %%

#trials

p_trials = 2
seedloops = 10 
episodes = 50

data = np.zeros((p_trials,seedloops,episodes))

for pt in range(p_trials):
    print(pt)

    for sl in range(seedloops):
        
        print('sl:', sl)
        if(pt == 0):
            # Q-Learning agent
            agent = dqa(25,4,3,replay = 0)
            
        if(pt == 1):
            # Dyna-Q agent with memory replay = 10
            agent = dqa(25,4,3,replay = 10)

        # Changing random seeds
        rseed = sl;
        np.random.seed(rseed)
        
        for ts in range(episodes):

            done = False
            tau = 0
            observation, info = env.reset(seed = rseed)
            obs = observation_decoder(observation)
            score = 0

            while(done == False):
                tau += 1
                
                action = agent.decision_making(obs)
                #action = np.random.randint(0,3)
                
                obs_prev = obs    

                # Fetching next-state reward from envrionment-function
                observation, reward, terminated, truncated, info = env.step(action)
                obs = observation_decoder(observation)

                reward = 100 if reward > 0 else -1
                score += reward

                #print(action)
                #print(obs, obs_prev)
                agent.learning_with_replay(obs_prev, action, 
                                           obs, reward, episode_num = ts, 
                                           total_number_of_episodes = episodes)
                
                if terminated or truncated:
                  done = True
        
            data[pt][sl][ts] = score

# %%

with open('data_trial_dynaq.npy', 'wb') as f:
    np.save(f, data)
