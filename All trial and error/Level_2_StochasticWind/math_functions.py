EPS_VAL = 1e-16 #negligibleconstant

import numpy as np
import math

def softmax(dist):
    """ 
    Computes the softmax function on a set of values
    """

    output = dist - dist.max(axis=0)
    output = np.exp(output)
    output = output / np.sum(output, axis=0)
    return output

def log_stable(arr):
    """
    Adds small epsilon value to an array before natural logging it
    """
    return np.log(arr + EPS_VAL)

def kl_div(P,Q):
    n=len(P)
    for i in range(n):
        if(P[i]==0):
            P[i]+=EPS_VAL
        if(Q[i]==0):
            Q[i]+=EPS_VAL
            
    dkl=0
    for i in range(n):
        dkl+=(P[i]*math.log(P[i]))-(P[i]*math.log(Q[i]))
    return(dkl)

def obj_array(num_arr):
    """
    Creates a generic object array with the desired number of sub-arrays, given by `num_arr`
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


def onehot(value, num_values):
    arr = np.zeros(num_values)
    arr[value] = 1.0
    return arr