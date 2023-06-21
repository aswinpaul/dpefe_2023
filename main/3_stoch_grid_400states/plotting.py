#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 15:53:50 2023

@author: aswinpaul
"""

import numpy as np
import matplotlib.pyplot as plt

data = {}

with open('data_ql.npy', 'rb') as file:
    data[0] = np.load(file)  
    
with open('data_dynaq_10.npy', 'rb') as file:
    data[1] = np.load(file) 

with open('data_random.npy', 'rb') as file:
    data[2] = np.load(file)
     
with open('data_dpefe.npy', 'rb') as file:
    data[3] = np.load(file)
    
agents = 4
episodes = 50

sample = np.shape(data[0][:,0:episodes][0])[0]

data_mean = {}  
for i in range(agents):
    data_mean[i] = np.mean(np.transpose(data[i][:,0:episodes]), axis=1)
    plt.plot(range(sample-1),data_mean[i][:-1])

data_std = {}    
for i in range(agents):
    fact = 0 if(i == 2) else 1
    data_std[i] = np.std(np.transpose(data[i][:,0:episodes]), axis=1)
    plt.fill_between(range(sample-1), 
                     np.clip(data_mean[i][:-1] + fact*data_std[i][:-1], None, 9.99),
                     np.clip(data_mean[i][:-1] - fact*data_std[i][:-1], -4.9, None),
                     alpha=0.3)

plt.legend(["Q-Learning agent", 
            "Dyna-Q agent (memory replay=10)",
            "Random agent",
            "DPEFE agent (T = 80)"
            ])

plt.title("Stochastic grid environment (400 states)")

plt.xlabel("Episode number")
plt.ylim(0, 10)
plt.xlim(0, 49)
plt.ylabel("Total score")
plt.savefig('perf_3', dpi=500, bbox_inches='tight');
