from math_functions import obj_array,obj_array_zeros
from wgwt_det_environment import ImportDynamics as DetDyn
from wgwt_environment_first import ImportDynamics as Dyn1
from wgwt_environment_first import StartandGoal,StatetoCordinates,ctostates
import numpy as np 

#Number of states and number of controls/action
num_states=[70]
num_controls=[8]
num_obs=[9,12]
num_factors=len(num_states)
num_control_factors=len(num_controls)
num_modalities = len(num_obs) 

numS=num_states[0]
numA=num_controls[0]
numO1=num_obs[0]
numO2=num_obs[1]

#Deterministic Environment
B_det=np.zeros((num_states[0], num_states[0], num_controls[0]))

T_dyn_det=DetDyn()
for i in range(numS):
    for j in range(numA):
        for k in range(numS):
            B_det[k,i,j]=T_dyn_det[i,j,k]
            
#Prior outcome dynamics deterministic environment
A = obj_array(num_modalities)

# for Modality-1: down_cordinate
A[0]=np.zeros((num_obs[0],num_states[0]));
for i in range(num_states[0]):
    [x,y]=StatetoCordinates(i)
    A[0][x,i]=1

#for Modality-2: side_cordinate
A[1]=np.zeros((num_obs[1],num_states[0]));
for i in range(num_states[0]):
    [x,y]=StatetoCordinates(i);
    A[1][y,i]=1

#=====================================================================

#True parameters of the environment to evaluate learning
#Transition-dynamics
T_dynamics1=Dyn1()
# T_dynamics2=Dyn2()
Btrue_1=np.zeros((num_states[0], num_states[0], num_controls[0]))
# Btrue_2=np.zeros((num_states[0], num_states[0], num_controls[0]))

for i in range(numS):
    for j in range(numA):
        for k in range(numS):
            Btrue_1[k,i,j]=T_dynamics1[i,j,k]
#             Btrue_2[k,i,j]=T_dynamics2[i,j,k]
            
#True_Likelihood
A_true=obj_array(num_modalities)
A_true[0]=np.zeros((num_obs[0],num_states[0]));
for i in range(num_states[0]):
    [x,y]=StatetoCordinates(i)
    A_true[0][x,i]=0.7
    A_true[0][x+1,i]=0.15
    A_true[0][x-1,i]=0.15

#for Modality-2: side_cordinate
A_true[1]=np.zeros((num_obs[1],num_states[0]));
for i in range(num_states[0]):
    [x,y]=StatetoCordinates(i);
    A_true[1][y,i]=0.7
    A_true[1][y+1,i]=0.15
    A_true[1][y-1,i]=0.15

def trueA():
    return A_true

def trueB():
    return Btrue_1

def det_dyn():
    return B_det

def det_A():
    return A