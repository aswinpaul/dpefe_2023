#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 12:11:01 2023

@author: aswinpaul
"""

from matplotlib import pyplot as plt
import numpy as np

state_space = np.linspace(1,5,num = 5)

u = 4
time = 2

o_c_cai = state_space*(u**time)
o_c_si = (state_space*u)**time
o_c_dpefe = state_space*u*time
o_c_aif_1 = state_space*u*time

plt.plot(state_space, o_c_cai)
plt.plot(state_space, o_c_si)
plt.plot(state_space, o_c_dpefe)
plt.plot(state_space, o_c_aif_1, ls = '--')

#highlight = [1, o_c_aif_1]
#plt.scatter(highlight[0], highlight[1] , marker='v', color='r')

#plt.yscale("log")
plt.xlabel("Dimension of state space ($Card(S)$)")
plt.ylabel("$\mathcal{O}$ of computational complexity (linear scale)")
plt.xlim(1, None)
plt.ylim(0, None)
plt.legend(["Classical Active Inference", 
            "Sophisticated Inference", 
            "DPEFE Method",
            "AIF (T = 1)"])
plt.title("Computational complexity vs $Card(S)$ [Fixed $T$ and $Card(U)$]")
plt.savefig('Complexity4Methods_S.png', dpi=500, bbox_inches='tight')