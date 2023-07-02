#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 15:53:50 2023

@author: aswinpaul
"""

import numpy as np
import matplotlib.pyplot as plt

with open('data_ql.npy', 'rb') as file:
    data_1 = np.load(file)  
    
with open('data_dynaq_10.npy', 'rb') as file:
    data_2 = np.load(file) 
    
with open('data_dpefe.npy', 'rb') as file:
    data_5 = np.load(file) 
    
with open('data_random.npy', 'rb') as file:
    data_6 = np.load(file) 
    
agents = 6
episodes = 50

sample = np.shape(data_1[:,0:episodes][0])[0]
ql_rm_1 = np.mean(np.transpose(data_1[:,0:episodes]), axis=1)
ql_rm_2 = np.mean(np.transpose(data_2[:,0:episodes]), axis=1)
ql_rm_5 = np.mean(np.transpose(data_5[:,0:episodes]), axis=1)
ql_rm_6 = np.mean(np.transpose(data_6[:,0:episodes]), axis=1)


plt.plot(range(sample-1),ql_rm_1[:-1])
plt.plot(range(sample-1),ql_rm_2[:-1])
plt.plot(range(sample-1),ql_rm_3[:-1])
plt.plot(range(sample-1),ql_rm_4[:-1])
plt.plot(range(sample-1),ql_rm_5[:-1])
plt.plot(range(sample-1),ql_rm_6[:-1])

ql_rs_1 = np.std(np.transpose(data_1[:,0:episodes]), axis=1)
ql_rs_2 = np.std(np.transpose(data_2[:,0:episodes]), axis=1)
ql_rs_3 = np.std(np.transpose(data_3[:,0:episodes]), axis=1)
ql_rs_4 = np.std(np.transpose(data_4[:,0:episodes]), axis=1)
ql_rs_5 = np.std(np.transpose(data_5[:,0:episodes]), axis=1)
ql_rs_6 = np.std(np.transpose(data_6[:,0:episodes]), axis=1)


# plt.fill_between(range(sample-1), ql_rm_1[:-1] + ql_rs_1[:-1], 
#                  np.clip(ql_rm_1[:-1] - ql_rs_1[:-1], 0, 999), alpha=0.3)
# plt.fill_between(range(sample-1), ql_rm_2[:-1] + ql_rs_2[:-1], 
#                  np.clip(ql_rm_2[:-1] - ql_rs_2[:-1], 0, 999), alpha=0.3)
# plt.fill_between(range(sample-1), ql_rm_3[:-1] + ql_rs_3[:-1], 
#                  np.clip(ql_rm_3[:-1] - ql_rs_3[:-1], 0, 999), alpha=0.3)
# plt.fill_between(range(sample-1), ql_rm_4[:-1] + ql_rs_4[:-1], 
#                  np.clip(ql_rm_4[:-1] - ql_rs_4[:-1], 0, 999), alpha=0.3)
# plt.fill_between(range(sample-1), ql_rm_5[:-1] + ql_rs_5[:-1], 
#                  np.clip(ql_rm_5[:-1] - ql_rs_5[:-1], 0, 999), alpha=0.3)


plt.legend(["Q-Learning agent", 
            "Dyna-Q agent (memory replay=10)", 
            "AIF agent (N = 1)" , 
            "SI agent (N = 2)", 
            "DPEFE agent (N = 80)",
            "Random agent"])

plt.title("Deterministic grid environment (220 states)")

plt.xlabel("Episode number in trial")
plt.ylim(3, 10.5)
plt.ylabel("Length of epsiode")
plt.savefig('perf_grid_deterministic.png', dpi=500, bbox_inches='tight');
