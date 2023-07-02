#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:46:06 2023

@author: aswinpaul
"""
import numpy as np 
import matplotlib.pyplot as plt

with open('data_ql.npy', 'rb') as file:
    data_1 = np.load(file)
    
with open('data_dynaq_5.npy', 'rb') as file:
    data_2 = np.load(file)   
    
with open('data_dynaq_10.npy', 'rb') as file:
    data_3 = np.load(file) 
    
with open('data_si.npy', 'rb') as file:
    data_4 = np.load(file) 
    
with open('data_dpefe.npy', 'rb') as file:
    data_5 = np.load(file) 
    
agents = 5
episodes=50
seedloops = 10

data = np.zeros((agents,seedloops, episodes))
data[0,:,:] = data_1[:,0:episodes]
data[1,:,:] = data_2[:,0:episodes]
data[2,:,:] = data_3[:,0:episodes]
data[3,:,:] = data_4[:,0:episodes]
data[4,:,:] = data_5[:,0:episodes]
    
def plot_result(data):
    
    sample = np.shape(data[0][0])[0]
       
    ql_rm_1 = np.mean(np.transpose(data[0]), axis=1)
    ql_rm_2 = np.mean(np.transpose(data[1]), axis=1)
    ql_rm_3 = np.mean(np.transpose(data[2]), axis=1)
    ql_rm_4 = np.mean(np.transpose(data[3]), axis=1)
    ql_rm_5 = np.mean(np.transpose(data[4]), axis=1)
    
    ql_rs_1 = np.std(np.transpose(data[0]), axis=1)
    ql_rs_2 = np.std(np.transpose(data[1]), axis=1)
    ql_rs_3 = np.std(np.transpose(data[2]), axis=1)
    ql_rs_4 = np.mean(np.transpose(data[3]), axis=1)
    ql_rs_5 = np.mean(np.transpose(data[4]), axis=1)
    
    plt.plot(range(sample-1),ql_rm_1[:-1])
    plt.plot(range(sample-1),ql_rm_2[:-1])
    plt.plot(range(sample-1),ql_rm_3[:-1])
    plt.plot(range(sample-1),ql_rm_4[:-1])
    plt.plot(range(sample-1),ql_rm_5[:-1])
    
    plt.fill_between(range(sample-1), ql_rm_1[:-1] + ql_rs_1[:-1], ql_rm_1[:-1] - ql_rs_1[:-1], alpha=0.3)
    plt.fill_between(range(sample-1), ql_rm_2[:-1] + ql_rs_2[:-1], ql_rm_2[:-1] - ql_rs_2[:-1], alpha=0.3)
    plt.fill_between(range(sample-1), ql_rm_3[:-1] + ql_rs_3[:-1], ql_rm_3[:-1] - ql_rs_3[:-1], alpha=0.3)
    #plt.fill_between(range(sample-1), ql_rm_4[:-1] + ql_rs_4[:-1], ql_rm_4[:-1] - ql_rs_4[:-1], alpha=0.3)
    #plt.fill_between(range(sample-1), ql_rm_5[:-1] + ql_rs_5[:-1], ql_rm_5[:-1] - ql_rs_5[:-1], alpha=0.3)
    
    
    plt.legend(["Q-Learning","Dyna-Q (replay=5)","Dyna-Q (replay=10)", "SI agent", "DPEFE agent"])
    plt.title("Performace of agents in Taxi-v3 Environment")
    
    plt.xlabel("Episode number")
    plt.ylabel("Total reward")
    plt.savefig('graph.png', dpi=500, bbox_inches='tight');
    
    return plt

# %%

# Plotting
plot1 = plot_result(data)