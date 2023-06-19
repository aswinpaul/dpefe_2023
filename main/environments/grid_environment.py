#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 10:49:46 2023

@author: aswinpaul
"""

# This is needed agents are in a diff folder
import os
import sys
from pathlib import Path

path = Path(os.getcwd())
module_path = str(path) + '/'
sys.path.append(module_path)

import matplotlib.pyplot as plt
import numpy as np

class grid_environment():
    def __init__(self, path, 
                 end_state,
                 stochastic = False, 
                 epi_length_limit = 10000, p_rew = 10, n_rew = -1e-3):
        
        #Reading arguments for program from shell call
        gridpath = path
        self.path = path
        
        #storing the file as strings line by line
        griddata=[]
        
        #Saving arm true means to the array-band (indices indicates arms)
        grid = open(str(gridpath), "r")
        for x in grid:
            griddata.append(x)
        grid.close()
        #Closing mdp file
        
        #Grid Representation
        grid=[]
        for i in range(len(griddata)):    
            row=[]
            for word in griddata[i].split():
                try:
                    row.append(int(word))
                except (ValueError, IndexError):
                    pass
            grid.append(row)
        
        n=len(grid)
        
        allstates=[]
        validstates=[]
        
        for i in range(n):
            for j in range(n):
                allstates.append((i,j))
                if(grid[i][j]!=1):
                    validstates.append((i,j))
                    
        numS = len(validstates)
        self.validstates = validstates
        self.allstates = allstates
        self.numS = numS
        actions = ['North', 'South', 'East', 'West']
        numA = len(actions)
        self.numA = numA
        
        self.T = np.zeros((numS,numA,numS))
        if(stochastic == True):
            system_certainty = 0.75
        else:
            system_certainty = 0.99
            
        stochasticity = [-1, 0, +1]
        
        for ss in stochasticity:
            if(ss == 0):
                p = system_certainty
            else:
                p = (1-system_certainty)/2

            #Transitions
            #all valid states
            for i in range(numS):
                #Valid actions
                for j in range(4):
                    a=validstates[i][0]
                    b=validstates[i][1]
                    s=(a,b)
                    #North
                    if(j==0):
                        ap=a-1
                        if(ap<1):
                            ap = 1
                        bp=b-ss
                        if(bp<1):
                            bp = 1
                        for k in range(numS):
                            if(validstates[k][0]==ap and validstates[k][1]==bp):
                                sp=(ap,bp)
                                break
                            else:
                                sp=(a,b)
                        s1 = self.ctostates(s[0],s[1])
                        ac = j
                        s2 = self.ctostates(sp[0],sp[1])
                        self.T[s1,ac,s2] += p
                    #South
                    if(j==1):
                        ap=a+1
                        if(ap<1):
                            ap = 1
                        bp=b-ss
                        if(bp<1):
                            bp = 1
                        for k in range(numS):
                            if(validstates[k][0]==ap and validstates[k][1]==bp):
                                sp=(ap,bp)
                                break
                            else:
                                sp=(a,b)
                        s1 = self.ctostates(s[0],s[1])
                        ac = j
                        s2 = self.ctostates(sp[0],sp[1])
                        self.T[s1,ac,s2] += p
                    #East
                    if(j==2):
                        ap=a-ss
                        if(ap<1):
                            ap = 1
                        bp=b+1
                        if(bp<1):
                            bp = 1
                        for k in range(numS):
                            if(validstates[k][0]==ap and validstates[k][1]==bp):
                                sp=(ap,bp)
                                break
                            else:
                                sp=(a,b)
                        s1 = self.ctostates(s[0],s[1])
                        ac = j
                        s2 = self.ctostates(sp[0],sp[1])
                        self.T[s1,ac,s2] += p
                    #West
                    if(j==3):
                        ap=a-ss
                        if(ap<1):
                            ap = 1
                        bp=b-1
                        if(bp<1):
                            bp = 1
                        for k in range(numS):
                            if(validstates[k][0]==ap and validstates[k][1]==bp):
                                sp=(ap,bp)
                                break
                            else:
                                sp=(a,b)
                        s1 = self.ctostates(s[0],s[1])
                        ac = j
                        s2 = self.ctostates(sp[0],sp[1])
                        self.T[s1,ac,s2] += p
                    
        #Fixed start and goal state
        self.stochastic = stochastic
        self.end_state = end_state
        self.termination = False
        self.truncation = False
        self.info = None
        self.tau_limit = epi_length_limit
        self.p_rew = p_rew
        self.n_rew = n_rew
        
    #to get the state number corresponding to the coordinate of a valis state
    def ctostates(self,x,y):
        s=0
        for i in range(self.numS):
            if(self.validstates[i][0]==x and self.validstates[i][1]==y):
                break
            s=s+1
        return(s)
    
    def statestoc(self,s):
        return [self.validstates[s][0], self.validstates[s][1]]

    def allstatestoc(self,s):
        return [self.allstates[s][0], self.allstates[s][1]]
        
    def render(self, animation_save = False, tau = 0):
        plt.figure(figsize=(5, 5))
        grid = np.loadtxt(self.path, dtype = int)
        [gx, gy] = self.statestoc(self.end_state)
        [x,y] = self.statestoc(self.current_state)
        grid[x][y] = 4
        grid[gx][gy] = 3
        plt.imshow(grid, cmap=plt.cm.CMRmap, interpolation='nearest') #
        plt.xticks([]), plt.yticks([])
        if(animation_save == True):
            # Figure figure when rendered in environment
            plt.savefig(f'./animation/img_{tau}.png')
        else:
            plt.show()

    def render_c_matrix(self, c, image_save = False, animation_save = False, tau = 0):
        plt.figure(figsize=(5, 5))
        if c is not None:
            grid = c
        plt.imshow(grid, cmap=plt.cm.CMRmap, interpolation='nearest')
        plt.xticks([]), plt.yticks([])
        if(animation_save == True):
            # Figure figure when rendered in environment
            plt.savefig(f'./animation/img_{tau}.png')
        elif(image_save == True):
            plt.savefig("c_matrix.png")
        else:
            plt.show()
        
    def reset(self, seed = 10):
        self.start_state = np.random.randint(0,self.numS)
        while(self.start_state == self.end_state):
            self.start_state = np.random.randint(0,self.numS) 
        self.current_state = self.start_state
        self.termination = False
        self.truncation = False
        self.tau = 0
        return self.current_state, self.info
        
    def step(self, action):
        self.tau += 1
        
        if(self.stochastic == False):
            n_s = np.argmax(self.T[self.current_state, action, :])
            self.current_state = n_s
        else:
            # Transition noise
            poss_ns = list(range(0,self.numS))
            n_s = np.random.choice(poss_ns, p=self.T[self.current_state,action,:])
            self.current_state = n_s
            # Obs noise
            n_s = np.random.choice(poss_ns, p=self.T[self.current_state,action,:])
            
        if(n_s == self.end_state):
            self.termination = True
            reward = self.p_rew
        else:
            reward = self.n_rew
            
        if(self.tau > self.tau_limit):
            self.truncation = True
        return n_s, reward, self.termination, self.truncation, self.info
        
    def get_trueB(self):
        true_B = np.zeros((self.numS, self.numS, self.numA))
        for i in range(self.numS):
            for j in range(self.numA):
                true_B[:,i,j] = self.T[i,j,:]
        return true_B
