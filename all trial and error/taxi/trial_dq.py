#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:06:01 2022

@author: aswinpaul
"""

import gym
import numpy as np

env = gym.make('Taxi-v3')
env.action_space.seed(42)

from agent_dynaq import dynaq_agent as dqa

episodes=100
seedloops = 10 #trials
mem_replay = 10

#Initialsing the model M(x,a,x',reward)
data = np.zeros((seedloops, episodes))

for sl in range(seedloops):
    print(sl)
    agent = dqa(500,6,replay=mem_replay)
        
    # Changing random seeds
    rseed = sl;
    np.random.seed(rseed)
    
    observation, info = env.reset(seed=42)
    
    
    for ts in range(episodes):
        done = False
        tau = 0
        
        score = 0
        while(done == False):
            tau += 1
            
            action = agent.decision_making(observation)
            
            obs_prev = observation    
            #Fetching next-state reward from envrionment-function
            observation, reward, terminated, truncated, info = env.step(action)
            
            agent.learning_with_replay(obs_prev, action, observation, reward)
            #if(reward == 20):
                #succ_episodes += 1
            score += reward
            
            if terminated or truncated:
                observation, info = env.reset(seed = 42)
                done == True
                break
    
        data[sl][ts] = score

with open('data_dynaq_10.npy', 'wb') as file:
    np.save(file, data)