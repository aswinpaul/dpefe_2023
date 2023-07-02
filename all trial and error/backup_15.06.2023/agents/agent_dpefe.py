#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 15:56:11 2022

@author: aswinpaul
"""

# This is needed since pymdp si base agent is in a diff folder
import os
import sys
from pathlib import Path

path = Path(os.getcwd())
module_path = str(path.parent.parent) + '/'
sys.path.append(module_path)

# importing the existing classical AI agent in pymdp to reuse inference and learning
from pymdp.agent_si import si_agent
import numpy as np
from pymdp.maths import softmax, kl_div, entropy

class dpefe_agent(si_agent):
    """
    # Necessary parameters for SI agent

    # num_states
    # num_obs
    # num_controls

    # Optional parameters
    # planning_horizon (default value 1)
    # A = prior for likelihood A (same structure as pymdp.utils.random_A_matrix(num_obs, num_states))
    # B = prior for transisiton matrix B (same structure as pymdp.utils.random_B_matrix(num_obs, num_states))
    # C = prior for preference dist. C (same structure as pymdp.utils.obj_array_zeros(num_obs))
    # D = 0 prior of hidden-state
    # action_precision (precision for softmax in taking decisions) default value: 1
    # planning_precision (precision for softmax during tree search) default value: 1
    # search_threshold = 1/16 parameter pruning tree search in SI tree-search (default value 1/16)

    # Useful combination functions 
    # agent.step([obs_list], learning = False): 
    Combines Inference, planning, learning, and decision-making
    Generative model will be learned and updated over time if learning = True
    """
    def __init__(self, num_states, num_obs, num_controls,
                  planning_horizon = 1, 
                  A = None, B = None, C = None, D = None, 
                  action_precision = 1,
                  planning_precision = 1):
        
        super().__init__(num_states = num_states,
                         num_obs = num_obs, 
                         num_controls = num_controls,
                         planning_horizon = planning_horizon, 
                         A = A, B = B, C = C, D = D, 
                         action_precision = action_precision,
                         planning_precision = planning_precision)
        self.N = planning_horizon
        
   # Planning with dynamic programming
    def plan_using_dynprog(self, modalities = False):
        self.G = np.zeros((self.N-1, self.numA, self.numS)) + self.EPS_VAL
        self.Q_actions = np.zeros((self.N-1, self.numA, self.numS)) + 1/self.numA
        T = self.N
        if(modalities == False):
            moda = list(range(self.num_modalities))
        else:
            moda = modalities
        
        for mod in moda:
            Q_po = np.zeros((self.A[mod].shape[0], self.numS, self.numA))
            
            for i in range(self.numS):
                for j in range(self.numA):
                    Q_po[:,i,j] = self.A[mod].dot(self.B[0][:,i,j])
    
            for k in range(T-2,-1,-1):
                for i in range(self.numA):
                    for j in range(self.numS):
                        
                        if(k == T-2):
                            self.G[k,i,j] += kl_div(Q_po[:,j,i],self.C[mod]) + np.dot(
                                self.B[0][:,j,i],entropy(self.A[mod]))
                        else:
                            self.G[k,i,j] += kl_div(Q_po[:,j,i],self.C[mod]) + np.dot(
                                self.B[0][:,j,i],entropy(self.A[mod]))
                            
                            # Dynamic programming backwards in time
                            self.G[k,i,j] += np.sum(np.matmul(np.reshape(np.multiply(
                                self.Q_actions[k+1,:,:],self.G[k+1,:,:]), 
                                (self.numA,self.numS)),np.reshape(self.B[0][:,j,i], 
                                                                  (self.numS,1))))
                        
                # Distribution for action-selection
                for l in range(self.numS):
                    self.Q_actions[k,:,l] = softmax(-1*self.gamma*self.G[k,:,l])               
    # Decision making
    def take_decision(self):
        
        # Making sure self.tau is never greater than T-2
        tau = self.N-2 if self.tau > self.N-2 else self.tau
        
        p1 = np.matmul(self.G[tau,:,:], self.qs[0])
        p = softmax(-1*self.alpha*p1)
        
        action = np.random.choice(list(range(0, self.numA)), size = None, 
                                  replace = True, p = p)
        self.action = np.array([action])
        
        return(action)    
        
   
    def step(self, obs_list, learning = True):
        """
        Agent step combines the following agent functions:
        Combines Inference, Planning, Learning, and decision-making.
        This function represents the agent-environment loop in behaviour where an "environment" feeds observations
        to an "Agent", then the "Agent" responds with actions to control the "environment".
        Usage: agent.step([obs_list])
        Returns: Action(s) from agent to environment
        """
        if(self.tau == 0):

            # Inference
            self.infer_states(obs_list)
            
            # Decision making
            self.take_decision()
            self.tau += 1
            
            # Learning D
            if(learning == True):
                self.update_D(self.qs)

        else:
            # Inference
            self.qs_prev = np.copy(self.qs)
            self.infer_states(obs_list)

            # Learning model parameters
            if(learning == True):
                # Updating b
                self.update_B(self.qs_prev)
                # Updating a
                self.update_A(obs_list)
            
            # Decision making
            self.take_decision()
            self.tau += 1

        return(self.action[0])
