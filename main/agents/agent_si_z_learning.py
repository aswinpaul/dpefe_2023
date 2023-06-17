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
from scipy.stats import dirichlet

class si_agent_learnc(si_agent):
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
                  planning_precision = 1, 
                  search_threshold = 1/16, 
                  eta_par = 5000):
        
        super().__init__(num_states = num_states,
                         num_obs = num_obs, 
                         num_controls = num_controls,
                         planning_horizon = planning_horizon, 
                         A = A, B = B, C = C, D = D,
                         action_precision = action_precision,
                         planning_precision = planning_precision, 
                         search_threshold = search_threshold)
        
        self.eta_par = eta_par
        self.trial_tau = 0
        self.c = np.copy(self.C)

    def update_c(self, prev_obs, obs, reward, moda = False, terminal = False):
        if(moda == False):
            for mod in range(self.num_modalities):
                exp_reward = np.exp(reward)
                
                eta = self.eta_par/(self.eta_par + self.trial_tau)
                
                if(terminal == True):
                    self.c[mod][obs[mod]] = exp_reward
                    b = eta*self.c[mod][obs[mod]]
                    self.c[mod][prev_obs[mod]] = b
                    
                else:
                    a = (1 - eta)*self.c[mod][prev_obs[mod]]
                    b = eta*exp_reward*self.c[mod][obs[mod]]
                    self.c[mod][prev_obs[mod]] = a + b

                self.c += self.EPS_VAL
                self.C[mod] = dirichlet.mean(self.c[mod])
    
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
            self.qs_prev = np.copy(self.qs)
            # Planning
            self.plan_tree_search()
            
            # Decision making
            self.take_decision()
            self.tau += 1
            self.trial_tau += 1
            
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
            
            # Planning
            self.plan_tree_search()
            
            # Decision making
            self.take_decision()
            self.tau += 1
            self.trial_tau += 1

        return(self.action)
