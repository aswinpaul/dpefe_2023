#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:29:05 2023

@author: aswinpaul
"""
import numpy as np
from matplotlib import pyplot as plt

with open('data_trial_dynaq.npy', 'rb') as f:
    data_dynaq = np.load(f)
    
with open('data_trial_dpefe.npy', 'rb') as g:
    data_dpefe = np.load(g)

# %%

def plot_result(data):
    
    sample = np.shape(data[0][0])[0]
       
    ql_rm_1 = np.mean(np.transpose(data[0]), axis=1)
    ql_rm_2 = np.mean(np.transpose(data[1]), axis=1)
    ql_rm_3 = np.mean(np.transpose(data[2]), axis=1)
    #ql_rm_4 = np.mean(np.transpose(data[3]), axis=1)
    
    ql_rs_1 = np.std(np.transpose(data[0]), axis=1)
    ql_rs_2 = np.std(np.transpose(data[1]), axis=1)
    ql_rs_3 = np.std(np.transpose(data[2]), axis=1)
    #ql_rs_4 = np.std(np.transpose(data[3]), axis=1)
    
    x_max = 1024
    x_min = -1024
    
    plt.plot(range(sample-1),ql_rm_1[:-1])
    plt.plot(range(sample-1),ql_rm_2[:-1])
    plt.plot(range(sample-1),ql_rm_3[:-1])
    #plt.plot(range(sample-1),ql_rm_4[:-1])
    
    plt.fill_between(range(sample-1), np.clip(ql_rm_1[:-1] + ql_rs_1[:-1], None, x_max), np.clip(ql_rm_1[:-1] - ql_rs_1[:-1], x_min, None), alpha=0.3)
    plt.fill_between(range(sample-1), np.clip(ql_rm_2[:-1] + ql_rs_2[:-1], None, x_max), np.clip(ql_rm_2[:-1] - ql_rs_2[:-1], x_min, None), alpha=0.3)
    plt.fill_between(range(sample-1), np.clip(ql_rm_3[:-1] + ql_rs_3[:-1], None, x_max), np.clip(ql_rm_3[:-1] - ql_rs_3[:-1], x_min, None), alpha=0.3)
    #plt.fill_between(range(sample-1), np.clip(ql_rm_4[:-1] + ql_rs_4[:-1], None, x_max), np.clip(ql_rm_4[:-1] - ql_rs_4[:-1], x_min, None), alpha=0.3)
    
    plt.legend(["Q-Learning", "Dyna-Q (replay=10)", "DPEFE Agent (N=50)"])
    plt.title("Agents in 'MiniGrid-Empty-16x16' Environment")

    plt.xlabel("Episode number")
    plt.ylabel("Reward after episode")
    plt.savefig('plot_result', dpi=500, bbox_inches='tight');
    
    return plt

# %%

# Plotting

agents = 3
seedloops = 10 
episodes = 50

data_for_plotting = np.zeros((agents,seedloops,episodes))

data_for_plotting[0,:,:] = data_dynaq[0,:,:]
data_for_plotting[1,:,:] = data_dynaq[1,:,:]

data_for_plotting[2,:,:] = data_dpefe[:,:]

plot1 = plot_result(data_for_plotting)