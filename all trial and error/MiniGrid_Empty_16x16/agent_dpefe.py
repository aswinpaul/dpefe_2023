#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 15:56:11 2022

@author: aswinpaul
"""

import numpy as np
from scipy.stats import dirichlet
np.random.seed(2022)

# Helper functions from pymdp: https://github.com/infer-actively/pymdp

def log_stable(arr):
    """
    Adds small epsilon value to an array before natural logging it
    """
    EPS_VAL = 1e-10
    return np.log(arr + EPS_VAL)

def kl_div(P,Q):
    """
    Parameters
    ----------
    P : Categorical probability distribution
    Q : Categorical probability distribution

    Returns
    -------
    The KL-DIV of P and Q

    """          
    dkl = 0 
    for i in range(len(P)):
        dkl += (P[i]*log_stable(P[i]))-(P[i]*log_stable(Q[i]))
    return(dkl)

def entropy(A):
    """ 
    Compute the entropy of a set of condition distributions, 
    i.e. one entropy value per column 
    """
    H_A = - (A * log_stable(A)).sum(axis=0)
    return H_A

def onehot(value, num_values):
    arr = np.zeros(num_values)
    arr[value] = 1.0
    return arr

def obj_array(num_arr):
    """
    Creates a generic object array with the desired number of sub-arrays, 
    given by `num_arr`
    """
    return np.empty(num_arr, dtype=object)

def obj_array_zeros(shape_list):
    """ 
    Creates a numpy object array whose sub-arrays are 1-D vectors
    filled with zeros, with shapes given by shape_list[i]
    """
    arr = obj_array(len(shape_list))
    for i, shape in enumerate(shape_list):
        arr[i] = np.zeros(shape)
    return arr

def norm_dist(dist):
    """ Normalizes a Categorical probability distribution (or set of them) 
    assuming sufficient statistics are stored in leading dimension"""
    
    if dist.ndim == 3:
        new_dist = np.zeros_like(dist)
        for c in range(dist.shape[2]):
            new_dist[:, :, c] = np.divide(dist[:, :, c], dist[:, :, c].sum(axis=0))
        return new_dist
    else:
        return np.divide(dist, dist.sum(axis=0))

def random_A_matrix(num_obs, num_states):
    """ Generates a random A-matrix i.e liklihood
    matrix using number of state and observation modalitiles
    """
    if type(num_obs) is int:
        num_obs = [num_obs]
    if type(num_states) is int:
        num_states = [num_states]
    num_modalities = len(num_obs)

    A = obj_array(num_modalities)
    for modality, modality_obs in enumerate(num_obs):
        modality_shape = [modality_obs] + num_states
        modality_dist = np.random.rand(*modality_shape)
        A[modality] = norm_dist(modality_dist)
    return A

def random_B_matrix(num_states, num_controls):
    """Generates a random B matrix i.e one step dynamics matrix using the number of
    (hidden states) and number of controls in each hidden states.
    Minimum number of controls equal to one i.e markov chain with action: 'Do nothing'.
    """
    if type(num_states) is int:
        num_states = [num_states]
    if type(num_controls) is int:
        num_controls = [num_controls]
    num_factors = len(num_states)
    assert len(num_controls) == len(num_states)

    B = obj_array(num_factors)
    for factor in range(num_factors):
        factor_shape = (num_states[factor], num_states[factor], num_controls[factor])
        factor_dist = np.random.rand(*factor_shape)
        B[factor] = norm_dist(factor_dist)
    return B


def softmax(dist):
    """ 
    Computes the softmax function on a set of values
    """
    output = dist - dist.max(axis=0)
    output = np.exp(output)
    output = output / np.sum(output, axis=0)
    return output

# DPEFE AGENT CLASS (Author: Aswin Paul)

class agent():
    
    # Definition of variables
    def __init__(self, num_states, num_obs, num_controls, planning_horizon, 
                 a = 0, b = 0, c = 0, d = 0, action_precision = 1, MDP = False):
        
        self.numS = 1
        self.numA = 1
        for i in num_states:
            self.numS *= i
        for i in num_controls:
            self.numA *= i
            
        self.num_states = [self.numS]
        self.num_factors = len(self.num_states)
        self.num_controls = [self.numA]
        
        self.num_obs = num_obs
        self.num_modalities = len(num_obs)
        
        self.EPS_VAL = 1e-16
        self.a = random_A_matrix(self.num_obs, self.num_states)*0 + self.EPS_VAL
        
        if(type(a) != int):
            for i in range(len(self.num_obs)):
                self.a[i] = a[i].reshape(num_obs[i], self.numS) 
        
        self.b = random_B_matrix(self.num_states, self.num_controls)*0 + self.EPS_VAL
        if(type(b) != int):
            bb = 1
            for i in range(len(num_states)):
                bb = np.kron(bb, b[i])
            self.b[0] = bb
            
        if(type(c) != int):
            self.c = c
        else:
            self.c = obj_array_zeros(num_obs)
            for idx in range(len(num_obs)):
                self.c[idx] += (1/num_obs[idx])
            
        if(type(d) != int):
            self.d = d
        else:
            self.d = obj_array_zeros(self.num_states)
            for idx in range(len(self.num_states)):
                self.d[idx] += 1 / self.num_states[idx]
                
        self.a += self.EPS_VAL
        self.b += self.EPS_VAL
            
        self.A = random_A_matrix(self.num_obs, self.num_states)*0 + self.EPS_VAL
        self.B = random_B_matrix(self.num_states, self.num_controls)*0 + self.EPS_VAL
        self.C = obj_array_zeros(self.num_obs)
        self.D = obj_array_zeros(self.num_states)
        
        self.learn_parameters()
        
        self.T = planning_horizon
        self.tau = 0

        self.G = np.zeros((self.T-1, self.numA, self.numS)) + self.EPS_VAL
        self.Q_actions = np.zeros((self.T-1, self.numA, self.numS)) + (1 / self.numA)      
            
        self.qs = obj_array_zeros(self.num_states)
        self.qs = np.copy(self.D)
        
        self.qs_prev = obj_array_zeros(self.num_states)
        self.qs_prev = np.copy(self.qs)
        
        self.action = 0
        self.action_precision = action_precision
        self.MDP = MDP
    
    # Inference using belief propogation (BP)
    def infer_hiddenstate(self, obs):
        self.qs_prev = np.copy(self.qs)
        
        for i in range(len(self.num_states)):
            # Likelihood
            term_2 = 0
            for j in range(len(self.num_obs)):
                term_2 += log_stable(np.matmul(np.transpose(self.A[j]),
                                               onehot(obs[j],self.num_obs[j])))
            
            if(self.MDP == True):
                # Only likelihood
                self.qs[i] = softmax(term_2)
            else:
                # Prior when POMDP
                if(self.tau == 0):
                    term_1 = log_stable(self.D[i] + self.EPS_VAL)
                else:
                    term_1 = log_stable(np.matmul(self.B[i][:,:,self.action],
                                                  self.qs_prev[i]))
                
                #Equal-weightage for prior and likelihood
                self.qs[i] = softmax(term_1 + term_2)

    # Planning with dynamic programming
    def plan_using_dynprog(self, modalities = False):
        print("Planning")
        
        T = self.T
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
                    self.Q_actions[k,:,l] = softmax(-1*self.action_precision*self.G[k,:,l])
                    
    # Decision making
    def take_decision(self):
        # Making sure self.tau is never greater than T-2
        tau = 0 if self.tau > self.T-2 else self.tau
        
        p = np.matmul(self.Q_actions[tau], self.qs[0])
        #p = softmax(self.action_precision*p)
        
        action = np.random.choice(list(range(0, self.numA)), size = None, replace = True, p = p)
        self.action = action
        return(action)

    # Learning model parameters
    # Updating parameter priors
    def update_a(self, obs):
        qss = 1
        for i in range(len(self.num_states)):
            qss = np.kron(self.qs[i],qss)
        for i in range(len(self.num_obs)):
            self.a[i] += np.kron(qss,onehot(obs[i],self.num_obs[i]).reshape((-1,1)))
        
    def update_b(self):
        action_list = [self.action]
        action = np.array(action_list)
        for i in range(len(self.num_states)):
            self.b[i][:,:,action[i]] += np.kron(self.qs_prev[i],self.qs[i].reshape((-1,1)))
    
    def update_c(self, obs, moda = False, factor = 1):
        if(moda == False):
            for mod in range(self.num_modalities):
                self.c[mod] += factor*onehot(obs[mod], self.num_obs[mod])
        else:
            for mod in moda:
                self.c[mod] += factor*onehot(obs[mod], self.num_obs[mod])
            
    def update_d(self):
        for i in range(len(self.num_states)):
            self.d[i] += self.qs[i]
    
    # Learning parameters A,B,C,D using a,b,c,d
    def learn_parameters(self, factor=1):
        for i in range(self.num_modalities):
            for k in range(self.num_states[0]):
                self.A[i][:,k] = dirichlet.mean(factor*self.a[i][:,k])
        
        for i in range(len(self.num_states)):
            for j in range(self.num_states[i]):
                for k in range(self.num_controls[i]):
                    self.B[i][:,j,k] = dirichlet.mean(factor*self.b[i][:,j,k])
                
        for i in range(len(self.num_states)):
            self.D[i] = softmax(self.d[i])
            
        for mod in range(self.num_modalities):
            self.C[mod] = softmax(self.c[mod])
        
    # Step Function for interaction with environment combining above functions
    
    def step(self, obs_list, tau):
        if(tau == 0):
            self.tau = 0
            
            # Inference
            self.infer_hiddenstate(obs_list)
            
            # Learning model parameters
            self.update_d()
            
            # Decision making
            self.take_decision()
            self.tau += 1
                
        else:
            # Inference
            self.infer_hiddenstate(obs_list)
            
            # Learning model parameters
            # Updating b
            self.update_b()
            # Updating a
            self.update_a(obs_list)
                
            # Decision making
            self.take_decision()
            self.tau += 1
            
        return(self.action)

    # End of trial
    def end_of_trial(self, learning = True, factor = 1):
        if(learning == True):
            self.learn_parameters(factor)
