from math_functions import log_stable
from math_functions import softmax
from math_functions import kl_div
import numpy as np
import math

#System properties
num_states=[70]
num_obs=[9,12]
num_controls=[8]

numS=num_states[0]
numA=num_controls[0]
numO1=num_obs[0]
numO2=num_obs[1]

EPS_VAL = 1e-16 #negligibleconstant

#Active inference algorithm
def action_dist(A,B,C,T,sm_par):
    
    Bo=np.zeros((numO1,numO2,numO1,numO2,numA))

    Bo1_i=np.zeros((numO1,numS,numA))
    Bo2_i=np.zeros((numO2,numS,numA))

    Bo1=np.zeros((numO1,numO1,numA))
    Bo2=np.zeros((numO2,numO2,numA))

    for i in range(numS):
        for j in range(numA):
            Bo1_i[:,i,j]=A[0].dot(B[0][:,i,j])

    for i in range(numS):
        for j in range(numA):
            Bo2_i[:,i,j]=A[1].dot(B[0][:,i,j])

    Q_po=np.zeros((numO1,numO2,numS,numA))
    for i in range(numO1):
        for j in range(numO2):
            for k in range(numS):
                for l in range(numA):
                    Q_po[i,j,k,l]=Bo1_i[i,k,l]*Bo2_i[j,k,l]
    C_po=np.zeros((numO1,numO2))
    for i in range(numO1):
        for j in range(numO2):
            C_po[i,j]=C[0][i]*C[1][j]

    #planning horizon
    G=np.zeros((T-1,numA,numS))
    Qpi=np.zeros((T-1,numA,numS))

    for k in range(T-2,-1,-1):
        for i in range(numA):
            for j in range(numS):

                if(k==T-2):
                    G[k,i,j]=kl_div(np.array(Q_po[:,:,j,i]).flatten(),np.array(C_po).flatten())

                else:
                    G[k,i,j]=kl_div(np.array(Q_po[:,:,j,i]).flatten(),np.array(C_po).flatten())
                    for jj in range(numS):
                        for kk in range(numA):
                            G[k,i,j]+=Qpi[k+1,kk,jj]*B[0][jj,j,i]*G[k+1,kk,jj]
                            
                            
                            
                            
                            

        #Distribution for action-selection
        for ppp in range(numS):
            Qpi[k,:,ppp]=softmax(sm_par*(-1*G[k,:,ppp]))
    
    return(Qpi)