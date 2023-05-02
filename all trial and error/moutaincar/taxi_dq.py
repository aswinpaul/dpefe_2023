#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:06:01 2022

@author: aswinpaul
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3')
env.action_space.seed(42)

from dynaq_agent import dynaq_agent as dqa

def dyna_q_agent_trail(episodes = 10000):
    
    seedloops = 10 #trials
    p_trials = 3

    #Initialsing the model M(x,a,x',reward)
    data = np.zeros((p_trials,seedloops, episodes))

    for pt in range(3):
        print(pt)

        for sl in range(seedloops):
            
            if(pt == 0):
                agent = dqa(500,6,replay=0)
            if(pt == 1):
                agent = dqa(500,6,replay=5)
            if(pt == 2):
                agent = dqa(500,6,replay=10)

            # Changing random seeds
            rseed = sl;
            np.random.seed(rseed)
            
            observation, info = env.reset(seed=42)
            
            for ts in range(episodes):
                done = False
                tau = 0
                
                while(done == False):
                    tau += 1
                    
                    action = agent.decision_making(observation)
                    
                    obs_prev = observation    
                    #Fetching next-state reward from envrionment-function
                    observation, reward, terminated, truncated, info = env.step(action)
                    
                    agent.learning_with_replay(obs_prev, action, observation, reward)
                    
                    if terminated or truncated:
                        observation, info = env.reset(seed = 42)
                        done == True
                        break
            
                data[pt][sl][ts] = tau
            
    return(data)

data = dyna_q_agent_trail(100)

# %%
def plot_result(data):
    
    sample = np.shape(data[0][0])[0]
       
    ql_rm_1 = np.mean(np.transpose(data[0]), axis=1)
    ql_rm_2 = np.mean(np.transpose(data[1]), axis=1)
    ql_rm_3 = np.mean(np.transpose(data[2]), axis=1)
    ql_rs_1 = np.std(np.transpose(data[0]), axis=1)
    ql_rs_2 = np.std(np.transpose(data[1]), axis=1)
    ql_rs_3 = np.std(np.transpose(data[2]), axis=1)
    
    x_min = np.min(data)
    plt.plot(range(sample-1),ql_rm_1[:-1])
    plt.plot(range(sample-1),ql_rm_2[:-1])
    plt.plot(range(sample-1),ql_rm_3[:-1])
    plt.fill_between(range(sample-1), ql_rm_1[:-1] + ql_rs_1[:-1], np.clip(ql_rm_1[:-1] - ql_rs_1[:-1], x_min, None), alpha=0.3)
    plt.fill_between(range(sample-1), ql_rm_2[:-1] + ql_rs_2[:-1], np.clip(ql_rm_2[:-1] - ql_rs_2[:-1],x_min, None), alpha=0.3)
    plt.fill_between(range(sample-1), ql_rm_3[:-1] + ql_rs_3[:-1], np.clip(ql_rm_3[:-1] - ql_rs_3[:-1],x_min, None), alpha=0.3)
    
    
    plt.legend(["Q-Learning","Dyna-Q (replay=5)","Dyna-Q (replay=10)"])
    plt.title("Dynamic Q-Learning agent")
    plt.axhspan(x_min-0.1, x_min+0.1, color='red', alpha=0.5)
    plt.xlabel("Episode number")
    plt.ylabel("Length of episode")
    plt.savefig('dyna_q_1', dpi=500, bbox_inches='tight');
    
    return plt

# %%

# Plotting
plot1 = plot_result(data)

# %%
#[(taxi_row, taxi_col, passenger_location, destination)]
x = list(env.decode(150))
x