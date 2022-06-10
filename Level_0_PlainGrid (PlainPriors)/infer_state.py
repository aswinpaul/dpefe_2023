from math_functions import log_stable
from math_functions import softmax
from math_functions import onehot
import numpy as np

#System properties
num_states=[70]
num_obs=[9,12]
num_controls=[8]

numS=num_states[0]
numA=num_controls[0]
numO1=num_obs[0]
numO2=num_obs[1]

def infer_state(prior,A,obs_idx):
    sm_par=1
    term_1=log_stable(prior)
    o1_term=log_stable(np.matmul(np.transpose(A[0]),onehot(obs_idx[0],numO1)))
    o2_term=log_stable(np.matmul(np.transpose(A[1]),onehot(obs_idx[1],numO2)))
    term_2=0.5*(o1_term+o2_term)
    #Equal-weightage for prior and likelihood
    posterior=softmax(sm_par*(term_1+term_2))
    
    return(posterior)

