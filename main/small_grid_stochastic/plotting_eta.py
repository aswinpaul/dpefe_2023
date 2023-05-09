#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:29:08 2023

@author: aswinpaul
"""

# mdppath = "eta_job-29511653.csv"

# #storing the file as strings line by line
# mdpdata=[]

# #Saving arm true means to the array-band (indices indicates arms)
# mdp = open(str(mdppath), "r")
# for x in mdp:
#     mdpdata.append(x)
# mdp.close()
#Closing mdp file.
        
# import numpy as np
import matplotlib.pyplot as plt
# import csv

# with open("eta_job-29511666.csv") as file:
#     csv_reader = csv.reader(file, delimiter=',')

import numpy as np
from numpy import genfromtxt
my_data = genfromtxt("eta_job-29516263.out", delimiter=',')

si_opt = np.zeros((210,2))
si_opt[:,0] = my_data[:,2]
si_opt[:,1] = my_data[:,4]

xx = np.split(si_opt[:,1], np.unique(si_opt[:, 0], return_index=True)[1][1:])

plt.scatter(si_opt[:,0], si_opt[:,1])

    
# #plt.plot(data_1[:,0], data_1[:,1])
# plt.title("Performance of SI agent vs learning parameter ($e$)")

# plt.xlabel("Value of e")
# plt.ylabel("Length of epsiodes in a trail (Mean)")
plt.savefig('si_optimisation.png', dpi=500, bbox_inches='tight');
