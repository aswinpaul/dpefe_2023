#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:29:08 2023

@author: aswinpaul
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
my_data = genfromtxt("eta_job-29920304.out", delimiter=',')

si_opt = np.zeros((210,2))
si_opt[:,0] = my_data[:,2]
si_opt[:,1] = my_data[:,4]

import pandas as pd
xx = pd.DataFrame(si_opt)
y = xx.groupby([0]).mean()
z = xx.groupby([0]).std()

plt.plot(y)
xa = list(range(0,210000,10000))

plt.fill_between(np.array(xa), y[1]+z[1], y[1]-z[1], alpha=0.3)

# Performance of SI agent vs learning parameter ($e$)
plt.title("Stochastic grid (100 states)")
plt.xlabel("Value of e")
plt.ylabel("Total score")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.savefig('si_optimisation.png', dpi=500, bbox_inches='tight');
