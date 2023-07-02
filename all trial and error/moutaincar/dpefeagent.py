#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 15:56:11 2022

@author: aswinpaul
"""

from scipy.stats import dirichlet
from pymdp import inference
from pymdp import utils
from pymdp import maths
import numpy as np

np.random.seed(2022)
EPS_VAL = 1e-16

def kl_div(P,Q):
    """
    Parameters
    ----------
    P : A Categorical distribution/vector P
    Q : A Categorical distribution/vector Q

    Returns
    -------
    KL_Divergence of P and Q: KL(P || Q).
    """
    n = len(P)
    for i in range(n):
        if(P[i] == 0):
            P[i] += EPS_VAL
        if(Q[i] == 0):
            Q[i] += EPS_VAL
    dkl=0
    for i in range(n):
        dkl += (P[i]*np.log(P[i]))-(P[i]*np.log(Q[i]))
        
    return(dkl)

def log_stable(arr):
    """
    Adds small epsilon value to an array before natural logging it
    """
    return np.log(arr + EPS_VAL)

def entropy(A):
    """ 
    Compute the entropy of a set of condition distributions, i.e. one entropy value per column 
    """
    H_A = - (A * log_stable(A)).sum(axis=0)
    return H_A

def spm_norm(A):
    """ 
    Returns normalization of Categorical distribution, 
    stored in the columns of A.
    """
    A = A + EPS_VAL
    normed_A = np.divide(A, A.sum(axis=0))
    return normed_A

class dpefe_agent():
    
    def __init__(self, num_states, num_obs, num_controls, T, a = None, b = None, c = None, d = None, action_precision = 10):
        
        self.num_states = num_states
        self.num_factors = len(num_states)
        self.num_obs = num_obs
        self.num_modalities = len(num_obs)
        self.num_controls = num_controls
        
        self.numS = 1
        self.numA = 1
        for i in num_states:
            self.numS *= i
        for i in num_controls:
            self.numA *= i
            
        if(a != None):
            self.a = a
        else:
            self.a = utils.random_A_matrix(self.num_obs, self.num_states)*0 + EPS_VAL
            
        self.A = utils.random_A_matrix(self.num_obs, self.num_states)*0
        self.A = np.copy(a)
            
        if(b != None):
            self.b = b
        else:
            self.b = utils.random_B_matrix(num_states, num_controls)*0 + EPS_VAL

        self.B = utils.random_B_matrix(num_states, num_controls)*0 + EPS_VAL
        self.B = np.copy(self.b)
        self.learn_B()
        
        if(c != None):
            self.c = c
        else:
            self.c = utils.obj_array_zeros(num_obs)
            for idx in range(len(num_obs)):
                self.c[idx] += (1/num_obs[idx])
            
        self.C = utils.obj_array_zeros(num_obs)
        self.C = np.copy(self.c)
            
        if(d != None):
            self.d = d
        else:
            self.d = utils.obj_array_zeros(num_states)
            for idx in range(len(num_states)):
                self.d[idx] += 1 / num_states[idx]
            
        self.D = utils.obj_array_zeros(num_states)
        self.D = np.copy(self.d)
        
        self.T = T
        self.tau = 0

        self.G = np.zeros((T-1, self.numA, self.numS)) + EPS_VAL
        self.Q_actions = np.zeros((T-1, self.numA, self.numS)) + (1 / self.numA)      
            
        self.qs = utils.obj_array_zeros(self.num_states)
        self.qs = np.copy(self.D)
        
        self.qs_prev = utils.obj_array_zeros(self.num_states)
        self.qs_prev = np.copy(self.qs)
        
        self.action = 0
        self.action_precision = action_precision
        
    def infer_hiddenstate(self, obs, MDP = True):
        self.qs_prev = np.copy(self.qs)
        
        obs_array = utils.obj_array(self.num_modalities)
        for i in range(self.num_modalities):
            obs_array[i] = self.onehot(obs[i], self.num_obs[i])
        qs = inference.update_posterior_states(self.A, obs_array)
        
        self.qs = qs
        
    def dpefe_plan(self, modalities = False):
        
        T = self.T
        
        if(modalities == False):
            moda = list(range(self.num_modalities))
        else:
            moda = modalities
    
        new_num_states = [self.numS]
    
        new_A = utils.random_A_matrix(self.num_obs, new_num_states) 
        new_B = utils.random_B_matrix(1, 1) 
    
        for i in range(self.num_modalities):
            new_A[i] = np.reshape(self.A[i], [self.A[i].shape[0], self.numS])
    
        for i in range(self.num_factors):
            new_B[0] = np.kron(new_B[0],self.B[i])
        
        for mod in moda:
    
            Q_po = np.zeros((self.A[mod].shape[0], self.numS, self.numA))
    
            for i in range(self.numS):
                for j in range(self.numA):
                    Q_po[:,i,j] = new_A[mod].dot(new_B[0][:,i,j])
    
            for k in range(T-2,-1,-1):
                for i in range(self.numA):
    
                        if(k == T-2):
                            for j in range(self.numS):
                                self.G[k,i,j] += kl_div(Q_po[:,j,i],self.C[mod]) + np.dot(new_B[0][:,j,i],entropy(new_A[mod]))
    
                        else:
                            for j in range(self.numS):
                                self.G[k,i,j] += kl_div(Q_po[:,j,i],self.C[mod]) + np.dot(new_B[0][:,j,i],entropy(new_A[mod]))
                            
                                for jj in range(self.numS):
                                    for kk in range(self.numA):
                                        self.G[k,i,j] += self.Q_actions[k+1,kk,jj] * new_B[0][jj,j,i] * self.G[k+1,kk,jj] 
    
                #Distribution for action-selection
                for l in range(self.numS):
                    self.Q_actions[k,:,l] = maths.softmax(self.action_precision*(-1*self.G[k,:,l]))
                    
    def get_action(self,tau):
        
        tau = self.T-2 if tau > self.T-2 else tau
        qss = 1
        for i in range(self.num_factors):
            qss = np.kron(qss, self.qs[i])
        pp = np.matmul(self.Q_actions[tau], qss)
        action = np.random.choice(list(range(0, self.numA)), size = None, replace = True, p = pp)
        self.action = action
        
        return(action)
    
    def update_a(self, obs):
        qss = 1
        for i in range(len(self.num_states)):
            qss = np.kron(self.qs[i],qss)
        for i in range(len(self.num_obs)):
            self.a[i] += np.kron(qss,utils.onehot(obs[i],self.num_obs[i]).reshape((-1,1)))
        
    def update_b(self, action_list):
        action = np.array(action_list)
        for i in range(len(self.num_states)):
            self.b[i][:,:,action[i]] += np.kron(self.qs_prev[i],self.qs[i].reshape((-1,1)))
    
    def update_c(self, obs):
        for mod in range(self.num_modalities):
            self.c[mod] += utils.onehot(obs[mod], self.num_obs[mod])
            
    def update_d(self):
        for i in range(len(self.num_states)):
            self.d[i] += self.qs[i]
        
    def learn_A(self):
        for i in range(self.num_modalities):
            self.A[i] = spm_norm(self.a[i])
        
    def learn_B(self):
        for i in range(len(self.num_states)):
            for j in range(self.num_states[i]):
                for k in range(self.num_controls[i]):
                    self.B[i][:,j,k] = dirichlet.mean(self.b[i][:,j,k])
                    
    def learn_C(self):
        for mod in range(self.num_modalities):
            self.C[mod] = maths.softmax(100*self.c[mod])
            
    def learn_D(self):
        for i in range(len(self.num_states)):
            self.D[i] = maths.softmax(100*self.d[i])

    def learn_params_endoftrial(self):
        # self.learn_A()
        self.learn_B()
        self.learn_C()
        self.learn_D()
            
    def step(self, obs_list, tau, MDP = True):
        self.tau = tau
        if(tau == 0):
            self.infer_hiddenstate(obs_list, MDP)
            self.update_d()
            action = self.get_action(tau)
        else:
            self.infer_hiddenstate(obs_list, MDP)
            #Learning B
            action_list = [self.action, 0]
            self.update_b(action_list)
            #Learning A
            # self.update_a(obs_list)
            action = self.get_action(tau)
        return(action)
    
    def random_A_matrix(num_obs, num_states):
        return utils.random_A_matrix(num_obs, num_states)

    def random_B_matrix(num_states, num_controls):
        return utils.random_B_matrix(num_states, num_controls)

    def obj_array_zeros(num_states):
        return utils.obj_array_zeros(num_states)

    def onehot(self, state, num_states):
        return utils.onehot(state, num_states)