#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:29:08 2023

@author: aswinpaul
"""

import numpy as np
import matplotlib.pyplot as plt

with open('data_eta_opt_si_small_grid.npy', 'rb') as file:
    data_1 = np.load(file) 
    
plt.plot(data_1[:,0], data_1[:,1])
plt.title("Performance of SI agent vs learning parameter ($e$)")

plt.xlabel("Value of e")
plt.ylabel("Length of epsiodes in a trail (Mean)")
plt.savefig('si_optimisation.png', dpi=500, bbox_inches='tight');
