#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 14:07:55 2023

@author: aswinpaul
"""
import os
import sys
from pathlib import Path

path = Path(os.getcwd())
module_path = str(path.parent) + '/'
sys.path.append(module_path)

import numpy as np
from environments.grid_environment import grid_environment as Env 
env = Env(path = '../environments/grid10.txt', end_state=37)

informed_c = np.zeros((11,11))
X = 6
Y = 6
for i in range(121):
    [x,y] = env.allstatestoc(i)
    informed_c[x-1][y-1] = -1*np.sqrt((x - X)**2 + (y - Y)**2)
    
# Taking a look at the structure of prior preference over the grid
env.render_c_matrix(c = informed_c, image_save=True)

sparse_c = np.zeros((11,11))
sparse_c[5][5] = 1

env.render_c_matrix(c = sparse_c, image_save=True)