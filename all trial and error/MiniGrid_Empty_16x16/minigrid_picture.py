#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 13:58:07 2023

@author: aswinpaul
"""

# Time of execution
import time
st = time.process_time()

# Environment Imports
import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper, ViewSizeWrapper

import matplotlib.pyplot as plt

env = gym.make("MiniGrid-Empty-16x16-v0", render_mode = "rgb_array")

env = RGBImgObsWrapper(env, tile_size = 16) 
#env = ViewSizeWrapper(env, agent_view_size = 8)

rseed = 10
observation, info = env.reset(seed = rseed)

plt.imshow(observation['image'])
plt.axis("off")
plt.savefig("grid.png", format = 'png')