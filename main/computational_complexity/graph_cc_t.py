#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 12:11:01 2023

@author: aswinpaul
"""

from matplotlib import pyplot as plt
import numpy as np

time_space = np.linspace(1,30,num = 30)

u = 4
state_space = 100

o_c_cai = state_space*(u**time_space)
o_c_si = (state_space*u)**time_space
o_c_dpefe = state_space*u*time_space
o_c_aif_1 = state_space*u*1

plt.plot(time_space, o_c_cai)
plt.plot(time_space, o_c_si)
plt.plot(time_space, o_c_dpefe)

highlight = [1, o_c_aif_1]
plt.scatter(highlight[0], highlight[1] , marker='v', color='r')

plt.yscale("log")
plt.xlabel("Time horizon of planning ($T$)")
plt.ylabel("$\mathcal{O}$ of computational complexity (log scale)")
plt.xlim(0, None)
plt.ylim(10, None)
plt.legend(["Classical Active Inference", 
            "Sophisticated Inference", 
            "DPEFE Method",
            "AIF (T = 1)"])
plt.title("Computational complexity vs $T$ [Fixed $Card(U)$ and $Card(U)$]")
plt.savefig('Complexity4Methods_T.png', dpi=500, bbox_inches='tight')
